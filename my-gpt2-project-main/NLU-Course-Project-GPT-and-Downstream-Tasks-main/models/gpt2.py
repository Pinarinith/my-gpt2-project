import torch
from torch import nn
from transformers import GPT2Model as HFModel

from config import GPT2Config
from models.base_gpt import GPTPreTrainedModel
from modules.gpt2_layer import GPT2Layer
from utils import get_extended_attention_mask

class GPT2Model(GPTPreTrainedModel):
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.config = config

        # 1) Ensure intermediate_size is set (GPT-2 uses 4*hidden_size by default)
        if self.config.intermediate_size is None:
            self.config.intermediate_size = self.config.hidden_size * 4

        # Embeddings
        self.word_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.pos_embedding = nn.Embedding(
            config.max_position_embeddings,
            config.hidden_size
        )
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids is constant [1, max_pos]
        position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
        self.register_buffer('position_ids', position_ids)

        # Transformer blocks
        self.gpt_layers = nn.ModuleList([
            GPT2Layer(config) for _ in range(config.num_hidden_layers)
        ])

        # Pooler / final norm
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_af = nn.Tanh()
        self.final_layer_norm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )

        self.init_weights()

    def embed(self, input_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Token + positional embeddings + dropout
        """
        bs, seq_len = input_ids.size()
        token_embeds = self.word_embedding(input_ids)           # [bs, seq_len, hidden]
        pos_ids = self.position_ids[:, :seq_len]                # [1, seq_len]
        pos_embeds = self.pos_embedding(pos_ids)                # [1, seq_len, hidden]
        return self.embed_dropout(token_embeds + pos_embeds)

    def encode(self, hidden_states: torch.FloatTensor, attention_mask: torch.LongTensor) -> torch.FloatTensor:
        """
        Pass through GPT2Layer stack.
        """
        extended_mask = get_extended_attention_mask(attention_mask, self.dtype)
        for block in self.gpt_layers:
            hidden_states = block(hidden_states, extended_mask)
        return hidden_states

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor):
        """
        Returns:
          last_hidden_state: [bs, seq_len, hidden]
          last_token:        [bs, hidden]  (final non-padded token)
        """
        embeds = self.embed(input_ids)
        seq_out = self.encode(embeds, attention_mask)
        seq_out = self.final_layer_norm(seq_out)

        last_idx = attention_mask.sum(dim=1) - 1
        last_token = seq_out[torch.arange(seq_out.size(0)), last_idx]
        return {'last_hidden_state': seq_out, 'last_token': last_token}

    def hidden_state_to_token(self, hidden_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Weight-tying: logits = hidden_state @ E^T
        """
        return torch.matmul(hidden_state, self.word_embedding.weight.t())

    @classmethod
    def from_pretrained(cls, model: str, d: int, l: int, num_heads: int):
        hf = HFModel.from_pretrained(model, local_files_only=True).eval()
        hf_cfg = hf.config

        # 1. 构造等价的 GPT2Config
        our_cfg = GPT2Config(
            vocab_size=hf_cfg.vocab_size,
            hidden_size=d,
            num_hidden_layers=l,
            num_attention_heads=num_heads,
            intermediate_size=getattr(hf_cfg, "n_inner", hf_cfg.hidden_size * 4),
            max_position_embeddings=getattr(hf_cfg, "n_ctx", 1024),
            pad_token_id=hf_cfg.pad_token_id,
            hidden_dropout_prob=hf_cfg.embd_pdrop,
            attention_probs_dropout_prob=hf_cfg.attn_pdrop,
            layer_norm_eps=hf_cfg.layer_norm_epsilon,
        )

    # 2. 实例化我们自己的模型并拷贝嵌入
        model = cls(our_cfg).eval()
        model.word_embedding.load_state_dict(hf.wte.state_dict())
        model.pos_embedding.load_state_dict(hf.wpe.state_dict())
        sd = hf.state_dict()          # 只取一次
        for i in range(l):
            layer = model.gpt_layers[i]

            # Q K V
            W = sd[f'h.{i}.attn.c_attn.weight']   # 形状 [in=768 , out=3*768]
            b = sd[f'h.{i}.attn.c_attn.bias']     # 形状 [3*768]

            layer.self_attention.query.weight.data.copy_(W[:, :d].T)      # 列切片 + 转置
            layer.self_attention.query.bias.data .copy_(b[:d])

            layer.self_attention.key.weight.data .copy_(W[:, d:2*d].T)
            layer.self_attention.key.bias.data   .copy_(b[d:2*d])

            layer.self_attention.value.weight.data.copy_(W[:, 2*d:3*d].T)
            layer.self_attention.value.bias.data .copy_(b[2*d:3*d])

            # 注意力输出投影
            layer.attention_dense.weight.data.copy_(sd[f'h.{i}.attn.c_proj.weight'].T)
            layer.attention_dense.bias.data  .copy_(sd[f'h.{i}.attn.c_proj.bias'])

            # MLP
            layer.interm_dense.weight.data.copy_(sd[f'h.{i}.mlp.c_fc.weight'].T)
            layer.interm_dense.bias.data   .copy_(sd[f'h.{i}.mlp.c_fc.bias'])

            layer.out_dense.weight.data.copy_(sd[f'h.{i}.mlp.c_proj.weight'].T)
            layer.out_dense.bias.data   .copy_(sd[f'h.{i}.mlp.c_proj.bias'])

            # LayerNorm
            layer.attention_layer_norm.weight.data.copy_(sd[f'h.{i}.ln_1.weight'])
            layer.attention_layer_norm.bias.data  .copy_(sd[f'h.{i}.ln_1.bias'])
            layer.out_layer_norm.weight.data.copy_(sd[f'h.{i}.ln_2.weight'])
            layer.out_layer_norm.bias.data  .copy_(sd[f'h.{i}.ln_2.bias'])
    
        # 最终 LayerNorm（放循环外）
        model.final_layer_norm.weight.data.copy_(sd['ln_f.weight'])
        model.final_layer_norm.bias.data  .copy_(sd['ln_f.bias'])


        return model
