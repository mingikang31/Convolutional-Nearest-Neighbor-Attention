// Fused Prime + Convolution operation (Forward + Backward)
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// ============================================================================
// FORWARD KERNEL
// ============================================================================
// Fused Kernel: top-k selection + softmax + gather + weighted sum + conv
// Eliminates multiple intermediate tensors and global memory round-trips
//
// We save topk_indices and softmax_weights for the backward pass rather than
// recomputing them, trading memory for compute.
__global__ void prime_conv_forward_kernel(
    const float* __restrict__ v,              // (B*NH, d_k, SL)
    const float* __restrict__ attn,           // (B*NH, SL, SL)
    const float* __restrict__ conv_w,         // (d_k, K) for depthwise
    float* __restrict__ output,               // (B*NH, d_k, SL)
    int* __restrict__ saved_topk_indices,     // (B*NH, SL, K) — saved for backward
    float* __restrict__ saved_softmax_weights,// (B*NH, SL, K) — saved for backward
    int B_NH, int d_k, int SL, int K
) {
    // Each block handles one (batch_head, channel) pair
    // Each thread handles one output sequence position
    int b = blockIdx.x;      // batch*num_heads index
    int c = blockIdx.y;      // channel (d_k) index
    int t_out = threadIdx.x; // output sequence position

    if (b >= B_NH || c >= d_k || t_out >= SL) return;

    // --- Shared memory for attention row ---
    // The attention row attn[b, t_out, :] is the SAME for all channels (c),
    // but each thread has a different t_out. So we can't share across channels
    // here. However, we CAN use shared memory to cache the attention row for
    // coalesced access if we restructure. For now, each thread reads its own row.
    //
    // A more advanced version would have blocks iterate over (b, t_out) and
    // use shared memory to broadcast the top-K indices to all channel threads.

    // ---- [Step 1]: Top-K selection via insertion sort (efficient for small K) ----
    float topk_vals[128];  // register storage, assuming K <= 128
    int topk_idxs[128];

    for (int i = 0; i < K; i++) {
        topk_vals[i] = -1e30f;
        topk_idxs[i] = 0;
    }

    // Single pass through attention row to find top-K
    const float* attn_row = attn + b * SL * SL + t_out * SL;
    for (int j = 0; j < SL; j++) {
        float val = attn_row[j];
        if (val > topk_vals[K - 1]) {
            // Insert into sorted position (descending order)
            int insert_pos = K - 1;
            while (insert_pos > 0 && val > topk_vals[insert_pos - 1]) {
                topk_vals[insert_pos] = topk_vals[insert_pos - 1];
                topk_idxs[insert_pos] = topk_idxs[insert_pos - 1];
                insert_pos--;
            }
            topk_vals[insert_pos] = val;
            topk_idxs[insert_pos] = j;
        }
    }

    // ---- [Step 2]: Softmax over top-K values ----
    float max_val = topk_vals[0]; // already sorted descending, so index 0 is max
    float sum_exp = 0.0f;
    for (int i = 0; i < K; i++) {
        topk_vals[i] = expf(topk_vals[i] - max_val);
        sum_exp += topk_vals[i];
    }
    for (int i = 0; i < K; i++) {
        topk_vals[i] /= sum_exp;
    }

    // ---- Save top-K indices and softmax weights for backward pass ----
    // Only need to save once per (b, t_out), not per channel.
    // We let channel 0 do the saving to avoid redundant writes.
    if (c == 0) {
        int save_base = b * SL * K + t_out * K;
        for (int i = 0; i < K; i++) {
            saved_topk_indices[save_base + i] = topk_idxs[i];
            saved_softmax_weights[save_base + i] = topk_vals[i];
        }
    }

    // ---- [Steps 3-5]: Gather from v, multiply by softmax, dot with conv weights ----
    float conv_result = 0.0f;
    for (int i = 0; i < K; i++) {
        float v_val = v[b * d_k * SL + c * SL + topk_idxs[i]];
        float primed = topk_vals[i] * v_val;
        float w = conv_w[c * K + i]; // depthwise: each channel has K weights
        conv_result += primed * w;
    }

    // Write output
    output[b * d_k * SL + c * SL + t_out] = conv_result;
}


// ============================================================================
// BACKWARD KERNEL
// ============================================================================
//
// Forward recap:
//   output[b,c,t] = SUM_i { s[i] * v[b, c, idx[i]] * w[c, i] }
//
// where:
//   idx[i]  = saved_topk_indices[b, t, i]     (top-K positions in attn row)
//   s[i]    = saved_softmax_weights[b, t, i]   (softmax over top-K attn values)
//   w[c, i] = conv_w[c, i]                     (depthwise conv weight)
//
// We need gradients w.r.t.:
//   1. grad_v[b, c, j]  — gradient of loss w.r.t. value tensor
//   2. grad_conv_w[c, i] — gradient of loss w.r.t. convolution weights
//   3. grad_attn[b, t, j] — gradient of loss w.r.t. attention matrix
//
// Derivations:
//
//   (1) grad_v[b, c, j] = SUM over t where idx[i]==j { grad_out[b,c,t] * s[i] * w[c,i] }
//       -> Use atomicAdd since multiple (t, i) pairs may map to the same j
//
//   (2) grad_conv_w[c, i] = SUM over (b, t) { grad_out[b,c,t] * s[i] * v[b,c,idx[i]] }
//       -> Use atomicAdd to accumulate across (b, t)
//
//   (3) For grad_attn, we need to backprop through softmax:
//       Let p[i] = s[i] * v[b,c,idx[i]] * w[c,i]  (the per-i contribution before sum)
//       d_loss/d_s[i] = SUM_c { grad_out[b,c,t] * v[b,c,idx[i]] * w[c,i] }
//
//       Softmax backward: if s = softmax(x), then
//         d_loss/d_x[i] = s[i] * (d_loss/d_s[i] - SUM_j { s[j] * d_loss/d_s[j] })
//
//       grad_attn[b, t, idx[i]] += d_loss/d_x[i]
//       (all other positions in grad_attn[b, t, :] are zero — top-K is sparse)

__global__ void prime_conv_backward_kernel(
    const float* __restrict__ grad_output,       // (B*NH, d_k, SL)
    const float* __restrict__ v,                 // (B*NH, d_k, SL)
    const float* __restrict__ conv_w,            // (d_k, K)
    const int* __restrict__ saved_topk_indices,  // (B*NH, SL, K)
    const float* __restrict__ saved_softmax_weights, // (B*NH, SL, K)
    float* __restrict__ grad_v,                  // (B*NH, d_k, SL)
    float* __restrict__ grad_conv_w,             // (d_k, K)
    float* __restrict__ grad_attn,               // (B*NH, SL, SL)
    int B_NH, int d_k, int SL, int K
) {
    int b = blockIdx.x;      // batch*num_heads index
    int c = blockIdx.y;      // channel index
    int t_out = threadIdx.x; // sequence position

    if (b >= B_NH || c >= d_k || t_out >= SL) return;

    float g_out = grad_output[b * d_k * SL + c * SL + t_out];
    int idx_base = b * SL * K + t_out * K;

    // ---- Preload saved top-K data for this (b, t_out) ----
    float s[128];     // softmax weights
    int idx[128];     // top-K indices
    for (int i = 0; i < K; i++) {
        s[i] = saved_softmax_weights[idx_base + i];
        idx[i] = saved_topk_indices[idx_base + i];
    }

    // ---- (1) grad_v and (2) grad_conv_w ----
    for (int i = 0; i < K; i++) {
        float v_val = v[b * d_k * SL + c * SL + idx[i]];
        float w_val = conv_w[c * K + i];

        // grad_v[b, c, idx[i]] += grad_out * s[i] * w[c, i]
        float gv = g_out * s[i] * w_val;
        atomicAdd(&grad_v[b * d_k * SL + c * SL + idx[i]], gv);

        // grad_conv_w[c, i] += grad_out * s[i] * v[b, c, idx[i]]
        float gw = g_out * s[i] * v_val;
        atomicAdd(&grad_conv_w[c * K + i], gw);
    }

    // ---- (3) grad_attn via softmax backward ----
    // First compute d_loss/d_s[i] for this channel
    float ds[128];
    for (int i = 0; i < K; i++) {
        float v_val = v[b * d_k * SL + c * SL + idx[i]];
        float w_val = conv_w[c * K + i];
        ds[i] = g_out * v_val * w_val;
    }

    // Softmax backward: d_x[i] = s[i] * (ds[i] - sum_j(s[j] * ds[j]))
    float dot = 0.0f;
    for (int i = 0; i < K; i++) {
        dot += s[i] * ds[i];
    }

    for (int i = 0; i < K; i++) {
        float grad_attn_val = s[i] * (ds[i] - dot);
        // Accumulate across channels with atomicAdd
        atomicAdd(&grad_attn[b * SL * SL + t_out * SL + idx[i]], grad_attn_val);
    }
}


// ============================================================================
// C++ WRAPPER FUNCTIONS
// ============================================================================

std::vector<torch::Tensor> prime_conv_forward_cuda(
    torch::Tensor v,       // (B*NH, d_k, SL)
    torch::Tensor attn,    // (B*NH, SL, SL)
    torch::Tensor conv_w,  // (d_k, K)
    int K
) {
    int B_NH = v.size(0);
    int d_k = v.size(1);
    int SL = v.size(2);

    auto output = torch::zeros({B_NH, d_k, SL}, v.options());
    auto saved_topk_indices = torch::zeros({B_NH, SL, K},
        torch::TensorOptions().dtype(torch::kInt32).device(v.device()));
    auto saved_softmax_weights = torch::zeros({B_NH, SL, K}, v.options());

    dim3 grid(B_NH, d_k);
    dim3 block(SL);

    prime_conv_forward_kernel<<<grid, block>>>(
        v.data_ptr<float>(),
        attn.data_ptr<float>(),
        conv_w.data_ptr<float>(),
        output.data_ptr<float>(),
        saved_topk_indices.data_ptr<int>(),
        saved_softmax_weights.data_ptr<float>(),
        B_NH, d_k, SL, K
    );

    return {output, saved_topk_indices, saved_softmax_weights};
}

std::vector<torch::Tensor> prime_conv_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor v,
    torch::Tensor conv_w,
    torch::Tensor saved_topk_indices,
    torch::Tensor saved_softmax_weights,
    int K
) {
    int B_NH = v.size(0);
    int d_k = v.size(1);
    int SL = v.size(2);

    auto grad_v = torch::zeros_like(v);
    auto grad_conv_w = torch::zeros_like(conv_w);
    auto grad_attn = torch::zeros({B_NH, SL, SL}, v.options());

    dim3 grid(B_NH, d_k);
    dim3 block(SL);

    prime_conv_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        v.data_ptr<float>(),
        conv_w.data_ptr<float>(),
        saved_topk_indices.data_ptr<int>(),
        saved_softmax_weights.data_ptr<float>(),
        grad_v.data_ptr<float>(),
        grad_conv_w.data_ptr<float>(),
        grad_attn.data_ptr<float>(),
        B_NH, d_k, SL, K
    );

    return {grad_v, grad_conv_w, grad_attn};
}
