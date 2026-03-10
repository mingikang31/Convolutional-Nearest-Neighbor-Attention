"""
Triton implementation of fused prime+conv.

Strategy:
  Forward:  torch.topk + torch.softmax (efficient, well-optimized)
            + Triton kernel for fused gather + weighted_sum + conv
            (replaces expand -> gather -> multiply -> reshape -> conv1d chain)

  Backward: Pure PyTorch with correct cross-channel softmax backward.
            The backward ops are less memory-intensive since we have
            the saved top-K indices and softmax weights.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl


# ============================================================================
# TRITON FORWARD KERNEL
# ============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16}, num_warps=1),
        triton.Config({'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
    ],
    key=['d_k', 'K'],
)
@triton.jit
def gather_conv_kernel(
    v_ptr,          # (B_NH, d_k, SL)
    topk_idx_ptr,   # (B_NH, SL, K)
    topk_sw_ptr,    # (B_NH, SL, K)
    conv_w_ptr,     # (d_k, K)
    output_ptr,     # (B_NH, d_k, SL)
    B_NH, d_k, SL,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Grid: (B_NH, SL, cdiv(d_k, BLOCK_C))
    pid_b = tl.program_id(0)   # batch_head
    pid_t = tl.program_id(1)   # sequence position
    pid_c = tl.program_id(2)   # channel block

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)   # (BLOCK_C,)
    c_mask = c_offs < d_k

    k_offs = tl.arange(0, BLOCK_K)                      # (BLOCK_K,)
    k_mask = k_offs < K

    # Load top-K indices and softmax weights (shared across all channels)
    idx_base = pid_b * SL * K + pid_t * K
    indices = tl.load(topk_idx_ptr + idx_base + k_offs, mask=k_mask, other=0)    # (BLOCK_K,)
    sw = tl.load(topk_sw_ptr + idx_base + k_offs, mask=k_mask, other=0.0)        # (BLOCK_K,)

    # Load conv weights: (BLOCK_C, BLOCK_K)
    w = tl.load(
        conv_w_ptr + c_offs[:, None] * K + k_offs[None, :],
        mask=c_mask[:, None] & k_mask[None, :], other=0.0
    )

    # Gather v values at top-K positions: (BLOCK_C, BLOCK_K)
    v_base = pid_b * d_k * SL
    v_vals = tl.load(
        v_ptr + v_base + c_offs[:, None] * SL + indices[None, :],
        mask=c_mask[:, None] & k_mask[None, :], other=0.0
    )

    # Weighted sum: softmax * v * conv_w, reduced over K
    result = tl.sum(sw[None, :] * v_vals * w, axis=1)    # (BLOCK_C,)

    # Store output
    out_base = pid_b * d_k * SL + pid_t
    tl.store(output_ptr + out_base + c_offs * SL, result, mask=c_mask)


# ============================================================================
# AUTOGRAD FUNCTION
# ============================================================================
class PrimeConvTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, attn, conv_w, K):
        """
        Args:
            v:      (B*NH, d_k, SL)
            attn:   (B*NH, SL, SL) -- raw attention scores
            conv_w: (d_k, K)
            K:      int
        """
        # CRITICAL: Triton kernels use pointer arithmetic that assumes contiguous
        # memory layout. v often arrives non-contiguous after permute() in the
        # attention module (e.g., v.reshape(...).permute(0, 2, 1)).
        v = v.contiguous()
        attn = attn.contiguous()

        B_NH, d_k, SL = v.shape

        # Step 1: Top-K + softmax (PyTorch -- already fast)
        topk_values, topk_indices = torch.topk(attn, k=K, dim=2, largest=True)
        topk_sw = torch.softmax(topk_values, dim=-1)

        # Step 2: Fused gather + conv (Triton)
        output = torch.empty(B_NH, d_k, SL, device=v.device, dtype=v.dtype)

        BLOCK_K = triton.next_power_of_2(K)
        grid = lambda meta: (B_NH, SL, triton.cdiv(d_k, meta['BLOCK_C']))

        gather_conv_kernel[grid](
            v, topk_indices, topk_sw, conv_w, output,
            B_NH, d_k, SL,
            K=K, BLOCK_K=BLOCK_K,
        )

        ctx.save_for_backward(v, conv_w, topk_indices, topk_sw)
        ctx.K = K
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Correct backward with cross-channel softmax backward (PyTorch ops)."""
        v, conv_w, topk_indices, topk_sw = ctx.saved_tensors
        K = ctx.K
        B_NH, d_k, SL = v.shape

        # Cast everything to float32 for numerical stability under AMP
        compute_dtype = torch.float32
        grad_output = grad_output.to(compute_dtype)
        v = v.to(compute_dtype)
        conv_w = conv_w.to(compute_dtype)
        topk_sw = topk_sw.to(compute_dtype)

        # Expand for broadcasting: all (B_NH, d_k, SL, K)
        idx_exp = topk_indices.unsqueeze(1).expand(-1, d_k, -1, -1)
        sw_exp  = topk_sw.unsqueeze(1).expand(-1, d_k, -1, -1)
        w_exp   = conv_w.unsqueeze(0).unsqueeze(2).expand(B_NH, -1, SL, -1)
        go_exp  = grad_output.unsqueeze(-1).expand(-1, -1, -1, K)

        # Gather v at top-K positions: (B_NH, d_k, SL, K)
        v_exp = v.unsqueeze(-1).expand(-1, -1, -1, K)
        v_gathered = torch.gather(v_exp, 2, idx_exp)

        # grad_v: flatten last two dims so scatter_add_ works on 3D tensor
        grad_v = torch.zeros(B_NH, d_k, SL, device=v.device, dtype=compute_dtype)
        grad_v.scatter_add_(
            2,
            idx_exp.reshape(B_NH, d_k, SL * K),
            (go_exp * sw_exp * w_exp).reshape(B_NH, d_k, SL * K)
        )

        # grad_conv_w
        grad_conv_w = (go_exp * sw_exp * v_gathered).sum(dim=[0, 2])

        # grad_attn (cross-channel softmax backward)
        # ds[b,t,k] = SUM_c { grad_out[b,c,t] * v[b,c,idx[k]] * w[c,k] }
        ds = (go_exp * v_gathered * w_exp).sum(dim=1)   # (B_NH, SL, K)
        dot = (topk_sw * ds).sum(dim=-1, keepdim=True)
        grad_topk = topk_sw * (ds - dot)

        grad_attn = torch.zeros(B_NH, SL, SL, device=v.device, dtype=compute_dtype)
        grad_attn.scatter_add_(2, topk_indices, grad_topk)

        return grad_v, grad_attn, grad_conv_w, None


# ============================================================================
# DROP-IN MODULE
# ============================================================================
class FusedPrimeConvTriton(nn.Module):
    def __init__(self, d_k, K):
        super().__init__()
        self.K = K
        self.conv_weight = nn.Parameter(torch.ones(d_k, K))

    def forward(self, v, attn):
        return PrimeConvTritonFunction.apply(v, attn, self.conv_weight, self.K)




        
import torch 
import torch.nn as nn 
import numpy as np

"""Regular ConvNN Attention Implementation"""
class MultiHeadConvNNAttention_Triton(nn.Module):
    def __init__(self, 
                 d_hidden,
                 num_heads, 
                 attention_dropout, 
                 K, 
                 convolution_type='depthwise',
                 seq_length=197):

        super(MultiHeadConvNNAttention_Triton, self).__init__()
        assert d_hidden % num_heads == 0, "d_hidden must be divisible by num_heads"

        self.d_hidden = d_hidden 
        self.num_heads = num_heads 
        self.attention_dropout = attention_dropout 
        self.d_k = d_hidden // num_heads 
        self.K = K 
        self.seq_length = seq_length

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)

        self.dropout = nn.Dropout(attention_dropout)

        self.in_channels = d_hidden // num_heads
        self.out_channels = d_hidden // num_heads

        if convolution_type == 'standard': 
            self.conv = nn.Conv1d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.K,
                stride=self.K,
                padding=0,
                bias=False
            )
        elif convolution_type == 'depthwise':
            self.conv = nn.Conv1d(
                in_channels=self.in_channels, 
                out_channels=self.out_channels,
                kernel_size=self.K,
                stride=self.K,
                padding=0,
                groups=self.in_channels, 
                bias=False
            )
        elif convolution_type == 'depthwise-separable':
            self.conv = nn.Sequential(
                # Depthwise Convolution
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=self.K,
                    stride=self.K,
                    padding=0,
                    groups=self.in_channels,
                    bias=False
                ), 
                # Pointwise Convolution
                nn.Conv1d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0, 
                    bias=False
                )
            )
        self.conv.weight.data.fill_(1.0)

        self.fused_prime_conv = FusedPrimeConvTriton(d_k=self.d_k, K=self.K)
        self.fused_prime_conv.conv_weight.data.copy_(self.conv.weight.data.squeeze(1))

    def split_head(self, x):
        batch_size, seq_length, d_hidden = x.size() 
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, num_heads, seq_length, d_k)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size() 
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_hidden)

    def forward(self, x):
        B = x.shape[0]

        # Linear Projection + Split Heads 
        q = self.split_head(self.W_q(x)) # (B, NH, SL, DK)
        k = self.split_head(self.W_k(x))
        v = self.split_head(self.W_v(x))

        # Attention Matrix: (B, NH, SL, SL) - Q @ K^T
        attn_matrix = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.d_k)

        # Merge B and num_heads into dim for prime & conv 
        ## (B, NH, SL, DK) → (B*NH, DK, SL) for v and (B, NH, SL, SL) → (B*NH, SL, SL)
        v_merged = v.reshape(B * self.num_heads, self.seq_length, self.d_k).permute(0, 2, 1)
        am_merged = attn_matrix.reshape(B * self.num_heads, self.seq_length, self.seq_length)

        # Prime and Convolution 
        out = self.fused_prime_conv(v_merged, am_merged) # (B*NH, DK, SL)

        # Reshape back: (B*NH, DK, SL) → (B, NH, SL, DK)
        out = out.permute(0, 2, 1).contiguous().view(B, self.num_heads, self.seq_length, self.d_k)
        out = self.dropout(out) 

        # Combine Heads and Final Linear Projection
        output = self.W_o(self.combine_heads(out)) # (B, SL, d_hidden)
        return output