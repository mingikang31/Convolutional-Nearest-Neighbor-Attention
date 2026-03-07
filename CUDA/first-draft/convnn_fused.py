"""
convnn_fused.py — PyTorch autograd wrapper for the fused CUDA prime+conv kernel.

Usage:
    from convnn_fused import FusedPrimeConv

    # Drop-in replacement for the _prime() + self.conv() sequence
    self.fused_prime_conv = FusedPrimeConv(d_k=64, K=8)
    ...
    # In forward():
    out = self.fused_prime_conv(v_merged, am_merged)
    # instead of:
    #   prime = self._prime(v_merged, am_merged, self.K)
    #   out = self.conv(prime)
"""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

# JIT-compile the CUDA extension (cached after first run)
convnn_cuda = load(
    name='convnn_cuda',
    sources=['convnn_kernel.cu', 'convnn_binding.cpp'],
    verbose=True,
)


class PrimeConvFunction(torch.autograd.Function):
    """Custom autograd function wrapping the fused CUDA kernels."""

    @staticmethod
    def forward(ctx, v, attn, conv_w, K):
        """
        Args:
            v:      (B*NH, d_k, SL) — value tensor (channels-first)
            attn:   (B*NH, SL, SL)  — raw attention scores (pre-softmax)
            conv_w: (d_k, K)        — depthwise convolution weights
            K:      int             — number of nearest neighbors

        Returns:
            output: (B*NH, d_k, SL) — fused prime + conv result
        """
        output, saved_topk_indices, saved_softmax_weights = convnn_cuda.forward(
            v.contiguous(), attn.contiguous(), conv_w.contiguous(), K
        )
        ctx.save_for_backward(v, conv_w, saved_topk_indices, saved_softmax_weights)
        ctx.K = K
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Returns:
            grad_v, grad_attn, grad_conv_w, None (for K)
        """
        v, conv_w, saved_topk_indices, saved_softmax_weights = ctx.saved_tensors

        grad_v, grad_conv_w, grad_attn = convnn_cuda.backward(
            grad_output.contiguous(), v, conv_w,
            saved_topk_indices, saved_softmax_weights, ctx.K
        )
        return grad_v, grad_attn, grad_conv_w, None


class FusedPrimeConv(nn.Module):
    """
    Drop-in nn.Module replacement for the _prime() + conv() sequence.

    Replaces:
        prime = self._prime(v_merged, am_merged, self.K)
        out = self.conv(prime)

    With:
        out = self.fused_prime_conv(v_merged, am_merged)
    """

    def __init__(self, d_k, K):
        super().__init__()
        self.K = K
        # Depthwise conv weight: each of d_k channels has K weights
        # Initialize to 1.0 to match original conv weight init
        self.conv_weight = nn.Parameter(torch.ones(d_k, K))

    def forward(self, v, attn):
        """
        Args:
            v:    (B*NH, d_k, SL)
            attn: (B*NH, SL, SL)

        Returns:
            out:  (B*NH, d_k, SL)
        """
        return PrimeConvFunction.apply(v, attn, self.conv_weight, self.K)


# =============================================================================
# NUMERICAL GRADIENT CHECK
# =============================================================================
def test_gradcheck():
    """Verify backward pass correctness with torch.autograd.gradcheck."""
    torch.manual_seed(42)

    B_NH, d_k, SL, K = 2, 4, 8, 3

    v = torch.randn(B_NH, d_k, SL, device='cuda', dtype=torch.float64, requires_grad=True)
    attn = torch.randn(B_NH, SL, SL, device='cuda', dtype=torch.float64, requires_grad=True)
    conv_w = torch.randn(d_k, K, device='cuda', dtype=torch.float64, requires_grad=True)

    # gradcheck uses float64 for numerical differentiation
    test = torch.autograd.gradcheck(
        PrimeConvFunction.apply,
        (v, attn, conv_w, K),
        eps=1e-6,
        atol=1e-4,
        rtol=1e-3,
    )
    print(f"Gradient check passed: {test}")


def test_forward_matches_pytorch():
    """Verify forward pass matches the original PyTorch implementation."""
    torch.manual_seed(42)

    B_NH, d_k, SL, K = 4, 64, 197, 8

    v = torch.randn(B_NH, d_k, SL, device='cuda')
    attn = torch.randn(B_NH, SL, SL, device='cuda')

    # --- Original PyTorch implementation ---
    topk_values, topk_indices = torch.topk(attn, k=K, dim=2, largest=True)
    topk_values = torch.softmax(topk_values, dim=-1)

    b, c, t = v.shape
    topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
    topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)
    v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
    prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
    prime = topk_values_exp * prime
    prime = prime.view(b, c, -1)

    # Use all-ones conv weight to isolate the prime operation
    conv_w = torch.ones(d_k, K, device='cuda')
    # The conv with stride=K and kernel=K is just a dot product per group of K
    # With all-ones weights, it's a sum over each K group
    prime_reshaped = prime.view(b, c, t, K)
    pytorch_out = prime_reshaped.sum(dim=-1)  # (B_NH, d_k, SL)

    # --- Fused CUDA implementation ---
    cuda_out = PrimeConvFunction.apply(v, attn, conv_w, K)

    # Compare
    max_diff = (pytorch_out - cuda_out).abs().max().item()
    print(f"Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-4, f"Forward mismatch! Max diff: {max_diff}"
    print("Forward pass matches PyTorch implementation!")


if __name__ == '__main__':
    test_forward_matches_pytorch()
    test_gradcheck()
