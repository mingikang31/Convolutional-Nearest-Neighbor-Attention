// Tiled Fused Prime + Convolution (Forward + Backward)
//
// Key differences from naive kernel:
//   Forward:  Grid(B_NH, SL), Block(d_k) — top-K computed ONCE per (b,t) by thread 0,
//             broadcast via shared memory, then all d_k threads do channel work in parallel.
//   Backward: Cross-channel reduction in shared memory for correct softmax backward.
//
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// FORWARD KERNEL (TILED)
// ============================================================================
__global__ void prime_conv_forward_tiled_kernel(
    const float* __restrict__ v,              // (B*NH, d_k, SL)
    const float* __restrict__ attn,           // (B*NH, SL, SL)
    const float* __restrict__ conv_w,         // (d_k, K)
    float* __restrict__ output,               // (B*NH, d_k, SL)
    int* __restrict__ saved_topk_idx,         // (B*NH, SL, K)
    float* __restrict__ saved_softmax_w,      // (B*NH, SL, K)
    int B_NH, int d_k, int SL, int K
) {
    int b = blockIdx.x;      // batch_head index
    int t = blockIdx.y;      // sequence position
    int c = threadIdx.x;     // channel index

    if (b >= B_NH || t >= SL) return;

    // Shared memory: K ints (indices) + K floats (softmax weights)
    extern __shared__ char smem[];
    int*   s_idx = (int*)smem;
    float* s_sw  = (float*)(s_idx + K);

    // ---- Phase 1: Thread 0 finds top-K and computes softmax ----
    if (c == 0) {
        float topk_v[128];
        int   topk_i[128];

        for (int i = 0; i < K; i++) {
            topk_v[i] = -1e30f;
            topk_i[i] = 0;
        }

        const float* attn_row = attn + b * SL * SL + t * SL;
        for (int j = 0; j < SL; j++) {
            float val = attn_row[j];
            if (val > topk_v[K - 1]) {
                int pos = K - 1;
                while (pos > 0 && val > topk_v[pos - 1]) {
                    topk_v[pos] = topk_v[pos - 1];
                    topk_i[pos] = topk_i[pos - 1];
                    pos--;
                }
                topk_v[pos] = val;
                topk_i[pos] = j;
            }
        }

        float max_v = topk_v[0];
        float sum_exp = 0.0f;
        for (int i = 0; i < K; i++) {
            topk_v[i] = expf(topk_v[i] - max_v);
            sum_exp += topk_v[i];
        }

        int save_base = b * SL * K + t * K;
        for (int i = 0; i < K; i++) {
            float sw = topk_v[i] / sum_exp;
            s_idx[i] = topk_i[i];
            s_sw[i]  = sw;
            saved_topk_idx[save_base + i]  = topk_i[i];
            saved_softmax_w[save_base + i] = sw;
        }
    }

    __syncthreads();

    // ---- Phase 2: Channel-parallel gather + weighted sum ----
    if (c < d_k) {
        float result = 0.0f;
        for (int i = 0; i < K; i++) {
            int   idx   = s_idx[i];
            float sw    = s_sw[i];
            float v_val = v[b * d_k * SL + c * SL + idx];
            float w     = conv_w[c * K + i];
            result += sw * v_val * w;
        }
        output[b * d_k * SL + c * SL + t] = result;
    }
}


// ============================================================================
// BACKWARD KERNEL (TILED) — correct cross-channel softmax backward
// ============================================================================
__global__ void prime_conv_backward_tiled_kernel(
    const float* __restrict__ grad_output,       // (B*NH, d_k, SL)
    const float* __restrict__ v,                 // (B*NH, d_k, SL)
    const float* __restrict__ conv_w,            // (d_k, K)
    const int*   __restrict__ saved_topk_idx,    // (B*NH, SL, K)
    const float* __restrict__ saved_softmax_w,   // (B*NH, SL, K)
    float* __restrict__ grad_v,                  // (B*NH, d_k, SL)
    float* __restrict__ grad_conv_w,             // (d_k, K)
    float* __restrict__ grad_attn,               // (B*NH, SL, SL)
    int B_NH, int d_k, int SL, int K
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int c = threadIdx.x;

    if (b >= B_NH || t >= SL) return;

    // Shared memory: K ints + K floats + d_k*K floats (for ds reduction)
    extern __shared__ char smem[];
    int*   s_idx = (int*)smem;
    float* s_sw  = (float*)(s_idx + K);
    float* s_ds  = s_sw + K;                    // d_k * K floats

    // Load saved top-K data into shared memory
    int save_base = b * SL * K + t * K;
    if (c == 0) {
        for (int i = 0; i < K; i++) {
            s_idx[i] = saved_topk_idx[save_base + i];
            s_sw[i]  = saved_softmax_w[save_base + i];
        }
    }
    __syncthreads();

    if (c >= d_k) return;

    float g_out = grad_output[b * d_k * SL + c * SL + t];

    // ---- Phase 1: grad_v, grad_conv_w (correct per-channel), and store ds ----
    for (int i = 0; i < K; i++) {
        int   idx   = s_idx[i];
        float sw    = s_sw[i];
        float v_val = v[b * d_k * SL + c * SL + idx];
        float w_val = conv_w[c * K + i];

        atomicAdd(&grad_v[b * d_k * SL + c * SL + idx], g_out * sw * w_val);
        atomicAdd(&grad_conv_w[c * K + i], g_out * sw * v_val);

        // ds[c][i] for softmax backward (needs cross-channel aggregation)
        s_ds[c * K + i] = g_out * v_val * w_val;
    }
    __syncthreads();

    // ---- Phase 2: Cross-channel reduction + softmax backward (thread 0 only) ----
    if (c == 0) {
        float ds_total[128];

        for (int i = 0; i < K; i++) {
            float sum = 0.0f;
            for (int ch = 0; ch < d_k; ch++) {
                sum += s_ds[ch * K + i];
            }
            ds_total[i] = sum;
        }

        // Softmax backward: grad_x[i] = s[i] * (ds_total[i] - dot(s, ds_total))
        float dot = 0.0f;
        for (int i = 0; i < K; i++) {
            dot += s_sw[i] * ds_total[i];
        }

        for (int i = 0; i < K; i++) {
            float grad_val = s_sw[i] * (ds_total[i] - dot);
            atomicAdd(&grad_attn[b * SL * SL + t * SL + s_idx[i]], grad_val);
        }
    }
}


// ============================================================================
// C++ WRAPPERS
// ============================================================================

std::vector<torch::Tensor> prime_conv_forward_tiled_cuda(
    torch::Tensor v, torch::Tensor attn, torch::Tensor conv_w, int K
) {
    int B_NH = v.size(0);
    int d_k  = v.size(1);
    int SL   = v.size(2);

    auto output          = torch::zeros({B_NH, d_k, SL}, v.options());
    auto saved_topk_idx  = torch::zeros({B_NH, SL, K},
        torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
    auto saved_softmax_w = torch::zeros({B_NH, SL, K}, v.options());

    dim3 grid(B_NH, SL);
    dim3 block(d_k);
    size_t smem = K * sizeof(int) + K * sizeof(float);

    prime_conv_forward_tiled_kernel<<<grid, block, smem>>>(
        v.data_ptr<float>(), attn.data_ptr<float>(), conv_w.data_ptr<float>(),
        output.data_ptr<float>(), saved_topk_idx.data_ptr<int>(),
        saved_softmax_w.data_ptr<float>(), B_NH, d_k, SL, K
    );

    return {output, saved_topk_idx, saved_softmax_w};
}

std::vector<torch::Tensor> prime_conv_backward_tiled_cuda(
    torch::Tensor grad_output, torch::Tensor v, torch::Tensor conv_w,
    torch::Tensor saved_topk_idx, torch::Tensor saved_softmax_w, int K
) {
    int B_NH = v.size(0);
    int d_k  = v.size(1);
    int SL   = v.size(2);

    auto grad_v      = torch::zeros_like(v);
    auto grad_conv_w = torch::zeros_like(conv_w);
    auto grad_attn   = torch::zeros({B_NH, SL, SL}, v.options());

    dim3 grid(B_NH, SL);
    dim3 block(d_k);
    size_t smem = K * sizeof(int) + K * sizeof(float) + d_k * K * sizeof(float);

    prime_conv_backward_tiled_kernel<<<grid, block, smem>>>(
        grad_output.data_ptr<float>(), v.data_ptr<float>(),
        conv_w.data_ptr<float>(), saved_topk_idx.data_ptr<int>(),
        saved_softmax_w.data_ptr<float>(), grad_v.data_ptr<float>(),
        grad_conv_w.data_ptr<float>(), grad_attn.data_ptr<float>(),
        B_NH, d_k, SL, K
    );

    return {grad_v, grad_conv_w, grad_attn};
}
