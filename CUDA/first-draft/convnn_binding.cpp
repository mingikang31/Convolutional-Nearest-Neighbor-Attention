// convnn_binding.cpp — PyTorch C++ binding for fused prime+conv CUDA kernels
#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
std::vector<torch::Tensor> prime_conv_forward_cuda(
    torch::Tensor v, torch::Tensor attn, torch::Tensor conv_w, int K);

std::vector<torch::Tensor> prime_conv_backward_cuda(
    torch::Tensor grad_output, torch::Tensor v, torch::Tensor conv_w,
    torch::Tensor saved_topk_indices, torch::Tensor saved_softmax_weights, int K);

// Input validation wrappers
std::vector<torch::Tensor> prime_conv_forward(
    torch::Tensor v, torch::Tensor attn, torch::Tensor conv_w, int K
) {
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(attn.is_cuda(), "attn must be a CUDA tensor");
    TORCH_CHECK(conv_w.is_cuda(), "conv_w must be a CUDA tensor");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(attn.is_contiguous(), "attn must be contiguous");
    TORCH_CHECK(conv_w.is_contiguous(), "conv_w must be contiguous");
    return prime_conv_forward_cuda(v, attn, conv_w, K);
}

std::vector<torch::Tensor> prime_conv_backward(
    torch::Tensor grad_output, torch::Tensor v, torch::Tensor conv_w,
    torch::Tensor saved_topk_indices, torch::Tensor saved_softmax_weights, int K
) {
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
    TORCH_CHECK(conv_w.is_cuda(), "conv_w must be a CUDA tensor");
    return prime_conv_backward_cuda(grad_output, v, conv_w,
                                     saved_topk_indices, saved_softmax_weights, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prime_conv_forward, "Fused Prime+Conv forward (CUDA)");
    m.def("backward", &prime_conv_backward, "Fused Prime+Conv backward (CUDA)");
}
