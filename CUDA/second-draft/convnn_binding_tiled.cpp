#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> prime_conv_forward_tiled_cuda(
    torch::Tensor v, torch::Tensor attn, torch::Tensor conv_w, int K);

std::vector<torch::Tensor> prime_conv_backward_tiled_cuda(
    torch::Tensor grad_output, torch::Tensor v, torch::Tensor conv_w,
    torch::Tensor saved_topk_idx, torch::Tensor saved_softmax_w, int K);

std::vector<torch::Tensor> prime_conv_forward_tiled(
    torch::Tensor v, torch::Tensor attn, torch::Tensor conv_w, int K
) {
    TORCH_CHECK(v.is_cuda() && attn.is_cuda() && conv_w.is_cuda(), "All tensors must be CUDA");
    TORCH_CHECK(v.is_contiguous() && attn.is_contiguous() && conv_w.is_contiguous(), "All tensors must be contiguous");
    return prime_conv_forward_tiled_cuda(v, attn, conv_w, K);
}

std::vector<torch::Tensor> prime_conv_backward_tiled(
    torch::Tensor grad_output, torch::Tensor v, torch::Tensor conv_w,
    torch::Tensor saved_topk_idx, torch::Tensor saved_softmax_w, int K
) {
    TORCH_CHECK(grad_output.is_cuda() && v.is_cuda() && conv_w.is_cuda(), "All tensors must be CUDA");
    return prime_conv_backward_tiled_cuda(grad_output, v, conv_w, saved_topk_idx, saved_softmax_w, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &prime_conv_forward_tiled, "Tiled Prime+Conv forward (CUDA)");
    m.def("backward", &prime_conv_backward_tiled, "Tiled Prime+Conv backward (CUDA)");
}
