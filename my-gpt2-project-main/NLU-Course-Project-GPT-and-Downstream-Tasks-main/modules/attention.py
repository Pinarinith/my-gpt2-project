import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#transoformer函数把状态拆分成multi-headed
  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    #attention_mask 是来自输入的musk info
    # 1. 计算 Q·K^T 并进行缩放：[..., seq_len, seq_len]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
           / (self.attention_head_size ** 0.5)
    scores = scores + attention_mask
    # 2. 对最后一维做 softmax 得到注意力权重，然后做 dropout
    attn_probs = torch.softmax(scores, dim=-1)
    attn_probs = self.dropout(attn_probs)
    # 3. 用注意力权重加权 V，得到上下文表示
    context = torch.matmul(attn_probs, value)  # [bs, num_heads, seq_len, head_size]
    # 4. 重排回 [bs, seq_len, num_heads, head_size]
    context = rearrange(context, 'b h t d -> b t (h d)')
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states : [bs, seq_len, hidden]
    attention_mask: [bs, 1, 1, seq_len]  (pad-mask: 0 或 -inf)
    """
    # 生成 Q K V 
    key_layer   = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)

    # 构造因果(下三角)掩码并叠加，通常是一个很小的数inf，防止模型在训练的时候在因果位置偷看未来，降低训练效果
    seq_len = hidden_states.size(1)
    causal = torch.triu(
        torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=hidden_states.dtype),
        1
    )
    # 上三角(不含对角)=1，因为每一个token完全自相关
    causal = causal.masked_fill(causal == 1, float('-inf')).unsqueeze(0).unsqueeze(0)
    full_mask = attention_mask + causal  # [bs, 1, seq, seq]

    #计算注意力
    attn_value = self.attention(key_layer, query_layer, value_layer, full_mask)
    return attn_value
