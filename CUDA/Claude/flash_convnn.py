import math
import torch
import torch.nn as nn
import flash_convnn_cuda as EXT


class _FlashConvNNFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, W, scale):
        K_top = W.size(1)
        out, tkidx, tkval = EXT.flash_dw_fwd(Q, K, V, W, K_top, scale)
        ctx.save_for_backward(Q, K, V, W, tkidx, tkval)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, W, tkidx, tkval = ctx.saved_tensors
        gV, gW, gTV_raw = EXT.flash_dw_bwd_vw(grad_out.contiguous(), V, W, tkidx, tkval)
        gQ, gK = EXT.flash_dw_bwd_qk(Q, K, tkidx, gTV_raw, ctx.scale)
        return (gQ.to(Q.dtype), gK.to(K.dtype), gV.to(V.dtype), gW.to(W.dtype), None)


class FlashCUDAMultiHeadConvNNAttention(nn.Module):
    def __init__(self, d_hidden, num_heads, attention_dropout, K, seq_length=197):
        super().__init__()
        assert d_hidden % num_heads == 0
        self.d_hidden, self.num_heads = d_hidden, num_heads
        self.d_k = d_hidden // num_heads
        self.K = K
        self.seq_length = seq_length
        assert self.d_k in (32, 64, 96, 128), f"D_K={self.d_k} not supported"
        assert K in (4, 8, 16, 32),           f"K={K} not supported"

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)
        self.dropout = nn.Dropout(attention_dropout)
        self.conv_weight = nn.Parameter(torch.ones(self.d_k, self.K))

    def _split_head(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, x):
        B, N, _ = x.shape
        q = self._split_head(self.W_q(x)).reshape(B * self.num_heads, N, self.d_k).contiguous()
        k = self._split_head(self.W_k(x)).reshape(B * self.num_heads, N, self.d_k).contiguous()
        v = self._split_head(self.W_v(x)).reshape(B * self.num_heads, N, self.d_k).contiguous()
        scale = 1.0 / math.sqrt(self.d_k)
        out = _FlashConvNNFn.apply(q, k, v, self.conv_weight, scale)
        out = out.view(B, self.num_heads, N, self.d_k).transpose(1, 2).contiguous().view(B, N, self.d_hidden)
        return self.W_o(self.dropout(out))