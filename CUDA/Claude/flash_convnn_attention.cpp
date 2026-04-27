/*
 * flash_convnn_attention.cpp
 *
 * pybind11 module exposing the CUDA kernels declared in
 * flash_convnn_attention_kernel.cu to Python.
 */

#include <torch/extension.h>
#include <vector>

// Forward declarations from the .cu file
std::vector<torch::Tensor> flash_topk_attn_fwd(
    torch::Tensor Q, torch::Tensor K, int K_top);

torch::Tensor flash_conv_aggregate_fwd(
    torch::Tensor V, torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor W);

std::vector<torch::Tensor> flash_conv_aggregate_bwd(
    torch::Tensor grad_out, torch::Tensor V,
    torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor W);

std::vector<torch::Tensor> flash_topk_attn_bwd(
    torch::Tensor Q, torch::Tensor Kmat,
    torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor d_topk_val);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashConvNN-Attention CUDA kernels (depthwise convolutional weighting).";

    m.def("topk_attn_fwd",  &flash_topk_attn_fwd,
          "Phase 1 forward: streaming Q@K^T + online top-K + softmax. "
          "Returns (topk_idx [int32], topk_val).");

    m.def("conv_aggregate_fwd", &flash_conv_aggregate_fwd,
          "Phase 2 forward: gather V at top-K positions, multiply by attn weight "
          "and depthwise per-channel kernel weight, sum over K.");

    m.def("conv_aggregate_bwd", &flash_conv_aggregate_bwd,
          "Phase 2 backward: returns (dV, dW, d_topk_val).");

    m.def("topk_attn_bwd", &flash_topk_attn_bwd,
          "Phase 1 backward: from d_topk_val (softmax-applied), produce (dQ, dK).");
}
