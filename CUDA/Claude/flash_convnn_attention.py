"""
flash_convnn_attention.py

PyTorch wrapper around the CUDA FlashConvNN-Attention kernels.

Provides:
  - FlashConvNNFunction:  torch.autograd.Function fusing both phases.
  - FlashMultiHeadConvNNAttention:  nn.Module drop-in replacement for
    MultiHeadConvNNAttention / FastMultiHeadConvNNAttention with
    convolution_type='depthwise'.

Mathematical equivalence (depthwise):
    y[b, q, m] = sum_{k=0..K-1}
                  softmax(s_q)_k *
                  V[b, topk_idx[b,q,k], m] *
                  conv_weight[m, 0, k]
    s_q = (Q_q @ K^T) / sqrt(d_k)

The shape convention matches PyTorch's nn.Conv1d depthwise layout exactly,
so weights from MultiHeadConvNNAttention.conv.weight (shape [d_k, 1, K])
can be loaded directly without reshaping.
"""

import math
import torch
import torch.nn as nn
import numpy as np
from torch.amp import custom_fwd, custom_bwd

# Lazy import of the C++/CUDA extension so this file imports even if not built
_ext = None
def _get_ext():
    global _ext
    if _ext is None:
        import flash_convnn_cuda  # built by setup.py
        _ext = flash_convnn_cuda
    return _ext


class FlashConvNNFunction(torch.autograd.Function):
    """Fused depthwise FlashConvNN-Attention: Q,K,V,W -> output.

    Shapes:
      Q, K, V : [BH, N, d_k]  (BH = B * num_heads)
      W       : [d_k, K]      (depthwise kernel weights, w[m, k])
      output  : [BH, N, d_k]
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, Q, K, V, W):
        ext = _get_ext()
        K_top = W.shape[1]

        # Phase 1: streaming Q@K^T + online top-K + softmax
        topk_idx, topk_val = ext.topk_attn_fwd(Q, K, K_top)

        # Phase 2: gather V at top-K positions, multiply by attn weight and
        # depthwise per-channel weight w[m, k], sum over K
        out = ext.conv_aggregate_fwd(V, topk_idx, topk_val, W)

        ctx.save_for_backward(Q, K, V, W, topk_idx, topk_val)
        return out

    @staticmethod
    @custom_bwd(device_type="cuda")
    def backward(ctx, grad_out):
        ext = _get_ext()
        Q, K, V, W, topk_idx, topk_val = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        # Phase 2 backward: grad_out -> dV, dW, d_topk_val
        dV, dW, d_topk_val = ext.conv_aggregate_bwd(
            grad_out, V, topk_idx, topk_val, W
        )

        # Phase 1 backward: d_topk_val -> dQ, dK
        dQ, dK = ext.topk_attn_bwd(Q, K, topk_idx, topk_val, d_topk_val)

        return dQ, dK, dV, dW


class FlashMultiHeadConvNNAttention(nn.Module):
    """Drop-in replacement for FastMultiHeadConvNNAttention(depthwise).

    Differences from FastMultiHeadConvNNAttention:
      - Never materializes the [B, NH, N, N] attention matrix.
      - Never calls torch.topk on a dense matrix.
      - Single Python -> 4 CUDA kernel launches per attention, vs.
        Triton's 1 + several PyTorch ops + 1 fused kernel.

    Args mirror the original; only convolution_type='depthwise' is supported
    here (the standard / dense conv variant has a different memory profile
    and is left in the Triton implementation for now).
    """

    def __init__(
        self,
        d_hidden: int,
        num_heads: int,
        attention_dropout: float,
        K: int,
        seq_length: int = 197,
        convolution_type: str = "depthwise",
    ):
        super().__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"
        assert convolution_type == "depthwise", (
            "FlashMultiHeadConvNNAttention currently supports only depthwise "
            "weighting. Use FastMultiHeadConvNNAttention for the dense variant."
        )
        assert K in (4, 8, 16, 32), (
            f"K={K} not in supported set {{4, 8, 16, 32}} (compile-time templated)"
        )

        self.d_hidden = d_hidden
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.d_k = d_hidden // num_heads
        self.K = K
        self.seq_length = seq_length
        self.convolution_type = convolution_type

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)

        self.dropout = nn.Dropout(attention_dropout)

        # Match PyTorch's nn.Conv1d depthwise layout: [d_k, 1, K]
        # (so checkpoints from MultiHeadConvNNAttention load directly)
        self.conv_weight = nn.Parameter(torch.ones(self.d_k, 1, self.K))

    def _split_head(self, x):
        B, T, C = x.shape
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B, T, _ = x.shape
        assert T == self.seq_length, (
            f"Expected seq_length={self.seq_length}, got {T}"
        )

        q = self._split_head(self.W_q(x))   # [B, NH, T, d_k]
        k = self._split_head(self.W_k(x))
        v = self._split_head(self.W_v(x))

        # Fold (B, NH) -> BH for the kernel
        BH = B * self.num_heads
        q_m = q.reshape(BH, T, self.d_k).contiguous()
        k_m = k.reshape(BH, T, self.d_k).contiguous()
        v_m = v.reshape(BH, T, self.d_k).contiguous()

        # Squeeze conv weight from [d_k, 1, K] to [d_k, K] to pass to kernel
        W = self.conv_weight.squeeze(1).contiguous()

        out = FlashConvNNFunction.apply(q_m, k_m, v_m, W)
        # out: [BH, T, d_k]

        out = out.view(B, self.num_heads, T, self.d_k).transpose(1, 2).contiguous()
        out = out.view(B, T, self.d_hidden)
        out = self.dropout(out)
        return self.W_o(out)


# -----------------------------------------------------------------------------
# Optional: standalone reference forward in PyTorch for unit-testing.
# Mirrors the math the CUDA kernels implement (depthwise, no dropout).
# -----------------------------------------------------------------------------

def reference_flash_forward(Q, K, V, W):
    """Pure PyTorch reference for FlashConvNNFunction.forward.

    Args:
      Q, K, V : [BH, N, d_k]
      W       : [d_k, K]
    Returns:
      out     : [BH, N, d_k]
    """
    BH, N, d_k = Q.shape
    K_top = W.shape[1]
    scale = 1.0 / math.sqrt(d_k)

    scores = torch.matmul(Q, K.transpose(-1, -2)) * scale          # [BH, N, N]
    topk_v, topk_i = torch.topk(scores, k=K_top, dim=-1, largest=True)
    p = torch.softmax(topk_v, dim=-1)                              # [BH, N, K]

    # Gather V at top-K rows -> [BH, N, K, d_k]
    idx_exp = topk_i.unsqueeze(-1).expand(-1, -1, -1, d_k)
    V_exp   = V.unsqueeze(1).expand(-1, N, -1, -1)
    V_gath  = torch.gather(V_exp, 2, idx_exp)                      # [BH, N, K, d_k]

    # Multiply by softmax weights and per-channel kernel weights, sum over K
    out = (V_gath * p.unsqueeze(-1) * W.t().unsqueeze(0).unsqueeze(0)).sum(dim=2)
    return out, topk_i, p
