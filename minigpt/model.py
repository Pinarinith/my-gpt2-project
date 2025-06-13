import os
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm  # 进度条工具

# =============================================
# 自注意力模块（Causal Self-Attention）
# 类似 GPT-1 的因果自注意力机制
# =============================================
class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0  # 确保 embedding 维度可以被头数整除

        # 合并 key, query, value 的线性投影层（一次性计算所有头）
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)

        # 输出投影层
        self.c_proj = nn.Linear(n_embd, n_embd)

        # Dropout 正则化
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 注册一个下三角矩阵作为因果掩码（只能关注当前及之前的 token）
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                     .view(1, 1, block_size, block_size))

        # 保存参数用于 forward 计算
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        B, T, C = x.size()  # 输入维度：batch size, sequence length, embedding dim

        # 一次性生成 q, k, v
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # 将每个 head 的维度分离出来 (B, nh, T, hs)，其中 hs = C // n_head
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 计算注意力得分
        att = (q @ k.transpose(-2, -1)) * (1.0 / (k.size(-1)**0.5))  # 缩放点积
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # 应用因果掩码
        att = F.softmax(att, dim=-1)  # softmax 归一化
        att = self.attn_dropout(att)  # 应用 dropout

        # 加权求和得到输出
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)

        # 最终输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


# =============================================
# Transformer Block 模块
# 包含一个自注意力层 + 一个前馈网络
# =============================================
class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)  # 第一层 LayerNorm
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)  # 注意力模块
        self.ln_2 = nn.LayerNorm(n_embd)  # 第二层 LayerNorm
        self.mlp = nn.Sequential(  # 前馈网络
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差连接 + 自注意力
        x = x + self.mlp(self.ln_2(x))   # 残差连接 + 前馈网络
        return x


# =============================================
# MiniGPT 模型类
# 一个简化版的 GPT 模型
# =============================================
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size  # 上下文最大长度
        self.vocab_size = vocab_size  # 词汇表大小

        # 构建模型组件
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),      # Token Embedding 层
            wpe = nn.Embedding(block_size, n_embd),      # Positional Embedding 层
            drop = nn.Dropout(dropout),                  # Dropout 层
            h = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]),  # 多个 Transformer Block
            ln_f = nn.LayerNorm(n_embd),                 # 最后的 LayerNorm
        ))

        # 输出头，用于预测下一个字符
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 共享权重（Token Embedding 和 Output Head 使用相同权重）
        self.transformer.wte.weight = self.lm_head.weight

        # 初始化模型参数
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """初始化模型参数"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        """
        前向传播
        :param idx: 输入序列 [B, T]
        :param targets: 目标序列 [B, T]
        :return: logits 和 loss（如果提供了 targets）
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"输入序列长度 {t} 超过最大限制 {self.block_size}"

        # 位置编码
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # 获取 token embeddings 和 position embeddings
        tok_emb = self.transformer.wte(idx)  # [b, t, n_embd]
        pos_emb = self.transformer.wpe(pos)  # [t, n_embd]

        # 合并并应用 dropout
        x = self.transformer.drop(tok_emb + pos_emb)

        # 依次通过各个 Transformer Block
        for block in self.transformer.h:
            x = block(x)

        # 最后一层 LayerNorm
        x = self.transformer.ln_f(x)

        # 输出 logits
        logits = self.lm_head(x)

        # 如果提供 targets，则计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        根据现有上下文生成新文本
        :param idx: 当前上下文 [1, T]
        :param max_new_tokens: 要生成的最大 token 数量
        :param temperature: 控制采样随机性的温度系数
        :param top_k: 只考虑概率最高的 top-k 个词
        :return: 生成的新序列
        """
        for _ in range(max_new_tokens):
            # 截断过长的上下文以适应 block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]

            # 前向传播获取 logits
            logits, _ = self(idx_cond)

            # 取最后一个时间步的 logits 并缩放
            logits = logits[:, -1, :] / temperature

            # 可选 top-k 截断
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)

            # 随机采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)

            # 拼接进已有序列
            idx = torch.cat((idx, idx_next), dim=1)

        return idx



def create_model_from_config(vocab_size, block_size, config, device):
    """根据配置创建模型"""
    config = config.get('config')
    model = MiniGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        dropout=config['dropout']
    ).to(device)
    return model


def load_hyperparam_configs(config_file):
    """加载超参数搜索配置"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            configs = json.load(f)
        return configs
    else:
        # 默认配置
        return [
            {"n_layer": 6, "n_head": 6, "n_embd": 384, "dropout": 0.2, "learning_rate": 5e-4},
        ]



def save_model_and_config(model, config, save_dir, model_name):
    """保存模型和配置"""
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}.pth")
    config_path = os.path.join(save_dir, f"{model_name}_config.json")

    torch.save(model.state_dict(), model_path)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    return model_path, config_path