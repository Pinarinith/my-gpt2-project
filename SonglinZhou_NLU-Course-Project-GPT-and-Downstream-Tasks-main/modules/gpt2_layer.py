from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, residual, sublayer_out, dense_layer, dropout):
    """
    GPT-2 的 “Dropout-Dense-Residual” 步骤
      residual     : 子层输入 (即残差)             [bs, seq, hidden]
      sublayer_out : 子层输出 (未投影)             [bs, seq, ?]
      dense_layer  : nn.Linear，把子层输出映射到 hidden_size
      dropout      : nn.Dropout
    返回值          : residual + dropout(dense(sublayer_out))
    （此处 **不做 LayerNorm**，因为 GPT-2 采用 Pre-LN 结构）
    """
    projected = dense_layer(sublayer_out)   # 线性映射到 hidden_size
    projected = dropout(projected)          # dropout
    return residual + projected             # 残差相加


  def forward(self, hidden_states, attention_mask):
    """
    GPT-2 单层前向:
      1. Pre-LN → CausalSelfAttention → Dropout+Residual
      2. Pre-LN → Feed-Forward(GELU) → Dropout+Residual
    返回更新后的 hidden_states
    """
    # 1) Multi-Head Self-Attention
    # 1-a. 预归一化
    normed_h = self.attention_layer_norm(hidden_states)
    # 1-b. 计算注意力
    attn_out = self.self_attention(normed_h, attention_mask)   # [bs, seq, hidden]
    # 1-c. Dropout + 残差
    hidden_states = self.add(hidden_states, attn_out,
                             self.attention_dense, self.attention_dropout)

    # 2) Position-wise Feed-Forward
    # 2-a. 预归一化
    normed_h = self.out_layer_norm(hidden_states)
    # 2-b. 前馈网络：Linear → GELU
    ff_mid = self.interm_af(self.interm_dense(normed_h))       # [bs, seq, intermediate]
    # 2-c. Dropout + 残差
    hidden_states = self.add(hidden_states, ff_mid,
                             self.out_dense, self.out_dropout)

    return hidden_states

