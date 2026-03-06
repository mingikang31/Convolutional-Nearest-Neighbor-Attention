"""Autograd wrapper for the tiled CUDA prime+conv kernel."""
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

convnn_tiled_cuda = load(
    name='convnn_tiled_cuda',
    sources=['convnn_kernel_tiled.cu', 'convnn_binding_tiled.cpp'],
    verbose=True,
)


class PrimeConvTiledFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, attn, conv_w, K):
        output, saved_topk_idx, saved_softmax_w = convnn_tiled_cuda.forward(
            v.contiguous(), attn.contiguous(), conv_w.contiguous(), K
        )
        ctx.save_for_backward(v, conv_w, saved_topk_idx, saved_softmax_w)
        ctx.K = K
        return output

    @staticmethod
    def backward(ctx, grad_output):
        v, conv_w, saved_topk_idx, saved_softmax_w = ctx.saved_tensors
        grad_v, grad_conv_w, grad_attn = convnn_tiled_cuda.backward(
            grad_output.contiguous(), v, conv_w,
            saved_topk_idx, saved_softmax_w, ctx.K
        )
        return grad_v, grad_attn, grad_conv_w, None


class FusedPrimeConvTiled(nn.Module):
    def __init__(self, d_k, K):
        super().__init__()
        self.K = K
        self.conv_weight = nn.Parameter(torch.ones(d_k, K))

    def forward(self, v, attn):
        return PrimeConvTiledFunction.apply(v, attn, self.conv_weight, self.K)
