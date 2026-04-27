/*
 * flash_convnn_attention_kernel.cu
 *
 * Fused CUDA kernels for FlashConvNN-Attention with depthwise convolutional
 * weighting. Implements:
 *
 *   Forward:
 *     Phase 1: Q,K -> (topk_idx, topk_softmax) WITHOUT materializing N x N.
 *              QK^T is streamed in K-tiles; per-query online top-K is kept
 *              in registers; softmax is applied to top-K values.
 *
 *     Phase 2: V, topk_idx, topk_softmax, depthwise conv_weight -> output.
 *              Gather V at top-K rows, multiply by attention weight,
 *              multiply by per-channel kernel weight w[d, k], accumulate.
 *
 *   Backward (matching PyTorch autograd grad_out -> dV, dQ, dK, dW):
 *     bwd Phase 2: grad_out -> d_topk_softmax, dV, dW
 *     bwd Phase 1: d_topk_softmax -> d_topk_scores -> dQ, dK
 *
 * Math:  y_{i,m} = sum_{k=0..K-1} softmax(s_i)_k * V[I_{i,k}, m] * w_{k, m}
 *        s_{i, j} = (Q_i . K_j) / sqrt(d_k)
 *        I_i = argkmax_K(s_i)
 *
 * We use fp32 accumulation throughout for numerical stability under AMP
 * (input may be fp16/bf16); outputs are cast back to the input dtype.
 *
 * Author: written for Mingi Kang's honors thesis on ConvNN-Attention.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <vector>

#ifndef FLASH_MAX_K
#define FLASH_MAX_K 32          // largest K supported at compile time
#endif

#ifndef FLASH_MAX_DK
#define FLASH_MAX_DK 128        // largest per-head dim supported
#endif

// -------------------------------------------------------------------------
// Helpers: convert any floating type to/from float for accumulation.
// We use static_cast which dispatches to c10::Half / c10::BFloat16's
// host/device operator float() and converting constructors -- portable and
// avoids the __half / at::Half ABI mismatch.
// -------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ float to_float(T x) { return static_cast<float>(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x) { return static_cast<T>(x); }

// =========================================================================
// PHASE 1 FORWARD: Streaming QK^T + online top-K + softmax
// =========================================================================
//
// One thread block handles ONE (batch_head, query). One thread per channel
// (or one thread per d in [0..Dk) for the dot product); we use a warp/block
// reduction for the dot product. We keep top-K in registers of thread 0,
// which is the simplest correct design.
//
// Layout:
//   Q, K : [BH, N, Dk]
//   topk_idx : [BH, N, K]      int32
//   topk_val : [BH, N, K]      same dtype as Q
//
// Each block:
//   - Loads its Q vector into shared memory (cooperative across threads).
//   - For each "key chunk" of size CHUNK (e.g. 32 keys at a time):
//       - Each thread t computes Q . K[chunk_base + t] using parallel
//         reduction across threads-with-same-key-id (we instead just do it
//         scalar-style: one thread per key in the chunk does the full dot).
//     The simple, correct approach: thread t handles key (chunk_base + t),
//     does the full dot product over Dk, writes into shared score buffer.
//   - Thread 0 merges shared score buffer into top-K.
//
// This is bandwidth-bound on K (must read all keys), but never materializes
// the full N x N matrix. Memory traffic is O(BH * N * Dk) per phase.
// -------------------------------------------------------------------------

template <typename scalar_t, int K_VAL, int BLOCK_THREADS>
__global__ void flash_topk_attn_fwd_kernel(
    const scalar_t* __restrict__ Q,       // [BH, N, Dk]
    const scalar_t* __restrict__ Kmat,    // [BH, N, Dk]
    int32_t*        __restrict__ topk_idx,// [BH, N, K]
    scalar_t*       __restrict__ topk_val,// [BH, N, K]  (softmax-applied)
    const int BH,
    const int N,
    const int Dk,
    const float scale)                    // = 1 / sqrt(Dk)
{
    const int bh    = blockIdx.x;
    const int q_idx = blockIdx.y;
    if (bh >= BH || q_idx >= N) return;

    const int tid = threadIdx.x;

    // Shared memory layout:
    //   q_vec        : Dk floats              (the query in fp32)
    //   chunk_scores : BLOCK_THREADS floats   (scores for current chunk)
    extern __shared__ float smem[];
    float* q_vec        = smem;
    float* chunk_scores = smem + Dk;

    // Pointers for this (bh, q)
    const scalar_t* q_ptr_global = Q    + (bh * N + q_idx) * Dk;
    const scalar_t* k_ptr_base   = Kmat + (bh * N) * Dk;

    // Cooperative load of the query into shared memory in fp32
    for (int d = tid; d < Dk; d += BLOCK_THREADS) {
        q_vec[d] = to_float<scalar_t>(q_ptr_global[d]);
    }
    __syncthreads();

    // Top-K registers held by thread 0 only
    float topk_scores_reg[K_VAL];
    int   topk_idx_reg   [K_VAL];
    int   min_pos = 0;
    if (tid == 0) {
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            topk_scores_reg[k] = -INFINITY;
            topk_idx_reg   [k] = -1;
        }
    }

    // Iterate over key chunks. CHUNK == BLOCK_THREADS so each thread handles one key.
    const int CHUNK = BLOCK_THREADS;
    for (int chunk_base = 0; chunk_base < N; chunk_base += CHUNK) {
        const int kj_local  = tid;
        const int kj_global = chunk_base + kj_local;

        float score;
        if (kj_global < N) {
            const scalar_t* k_row = k_ptr_base + kj_global * Dk;
            float acc = 0.0f;
            // Manual unroll-friendly loop. Dk is at most FLASH_MAX_DK.
            #pragma unroll 4
            for (int d = 0; d < Dk; ++d) {
                acc += q_vec[d] * to_float<scalar_t>(k_row[d]);
            }
            score = acc * scale;
        } else {
            score = -INFINITY;
        }
        chunk_scores[kj_local] = score;
        __syncthreads();

        // Thread 0 merges this chunk into top-K
        if (tid == 0) {
            const int valid = min(CHUNK, N - chunk_base);
            for (int j = 0; j < valid; ++j) {
                const float s = chunk_scores[j];
                if (s > topk_scores_reg[min_pos]) {
                    topk_scores_reg[min_pos] = s;
                    topk_idx_reg   [min_pos] = chunk_base + j;
                    // Re-find min position
                    float m = topk_scores_reg[0];
                    int   mp = 0;
                    #pragma unroll
                    for (int kk = 1; kk < K_VAL; ++kk) {
                        if (topk_scores_reg[kk] < m) {
                            m  = topk_scores_reg[kk];
                            mp = kk;
                        }
                    }
                    min_pos = mp;
                }
            }
        }
        __syncthreads();
    }

    // Thread 0 applies softmax over the top-K and writes outputs.
    if (tid == 0) {
        float maxv = topk_scores_reg[0];
        #pragma unroll
        for (int k = 1; k < K_VAL; ++k)
            maxv = fmaxf(maxv, topk_scores_reg[k]);

        float sum = 0.0f;
        float exps[K_VAL];
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            exps[k] = expf(topk_scores_reg[k] - maxv);
            sum   += exps[k];
        }
        const float inv = 1.0f / sum;

        const int out_base = (bh * N + q_idx) * K_VAL;
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            topk_idx[out_base + k] = topk_idx_reg[k];
            topk_val[out_base + k] = from_float<scalar_t>(exps[k] * inv);
        }
    }
}

// =========================================================================
// PHASE 2 FORWARD: Fused gather V + attn-weight + depthwise conv
// =========================================================================
//
//   out[b, q, m] = sum_k topk_val[b,q,k] * V[b, topk_idx[b,q,k], m] * w[m, k]
//
// One block per (bh, q). One thread per channel m (or strided if Dk > BLOCK).
// Loads top-K idx/val/weight into shared memory once.
// -------------------------------------------------------------------------

template <typename scalar_t, int K_VAL, int BLOCK_THREADS>
__global__ void flash_conv_aggregate_fwd_kernel(
    const scalar_t* __restrict__ V,         // [BH, N, Dk]
    const int32_t*  __restrict__ topk_idx,  // [BH, N, K]
    const scalar_t* __restrict__ topk_val,  // [BH, N, K]
    const scalar_t* __restrict__ W,         // [Dk, K]   (depthwise: w[m, k])
    scalar_t*       __restrict__ out,       // [BH, N, Dk]
    const int BH, const int N, const int Dk)
{
    const int bh = blockIdx.x;
    const int q  = blockIdx.y;
    if (bh >= BH || q >= N) return;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    int*   sh_idx = (int*)  smem_raw;                              // K
    float* sh_val = (float*)(sh_idx + K_VAL);                      // K

    // Load top-K idx/val into shared memory in fp32
    if (tid < K_VAL) {
        sh_idx[tid] = topk_idx[(bh * N + q) * K_VAL + tid];
        sh_val[tid] = to_float<scalar_t>(topk_val[(bh * N + q) * K_VAL + tid]);
    }
    __syncthreads();

    // Each thread handles channels m = tid, tid+BLOCK, ...
    for (int m = tid; m < Dk; m += BLOCK_THREADS) {
        float acc = 0.0f;

        // Pre-load weight row w[m, :] into registers
        float w_row[K_VAL];
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k)
            w_row[k] = to_float<scalar_t>(W[m * K_VAL + k]);

        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            const int v_idx = sh_idx[k];
            const float vv = to_float<scalar_t>(V[(bh * N + v_idx) * Dk + m]);
            acc += sh_val[k] * vv * w_row[k];
        }
        out[(bh * N + q) * Dk + m] = from_float<scalar_t>(acc);
    }
}

// =========================================================================
// PHASE 2 BACKWARD: gradients for V, conv_weight W, topk_softmax
// =========================================================================
//
//   y[b,q,m] = sum_k p[b,q,k] * V[b, I[b,q,k], m] * w[m, k]
//
//   dV[b, j, m]  += sum_{(q,k): I[b,q,k]=j} grad_y[b,q,m] * p[b,q,k] * w[m, k]
//   dw[m, k]     += sum_{b,q}  grad_y[b,q,m] * p[b,q,k] * V[b, I[b,q,k], m]
//   dp[b,q,k]    =  sum_m      grad_y[b,q,m] * V[b, I[b,q,k], m] * w[m, k]
//
// Strategy:
//   One block per (bh, q). All threads cooperate over channels m.
//   - dp[b,q,k]  : per-(b,q,k) reduction across m -> use shared-memory
//                  reduction; thread 0 writes the K outputs.
//   - dV         : we know j = I[b,q,k] (k=0..K-1). For each k, scatter into
//                  dV[b, j, m] for m in this thread's strided range using
//                  atomicAdd (fp32 only -> we accumulate dV in fp32 buffer).
//   - dw         : grad_y * p * V_gathered, accumulate across (b,q); use
//                  atomicAdd into a fp32 [Dk, K] buffer.
//
// Final post-pass casts fp32 dV and dw back to the original dtypes.
// -------------------------------------------------------------------------

template <typename scalar_t, int K_VAL, int BLOCK_THREADS>
__global__ void flash_conv_aggregate_bwd_kernel(
    const scalar_t* __restrict__ grad_out,   // [BH, N, Dk]
    const scalar_t* __restrict__ V,          // [BH, N, Dk]
    const int32_t*  __restrict__ topk_idx,   // [BH, N, K]
    const scalar_t* __restrict__ topk_val,   // [BH, N, K]
    const scalar_t* __restrict__ W,          // [Dk, K]
    float*          __restrict__ dV_fp32,    // [BH, N, Dk]
    float*          __restrict__ dW_fp32,    // [Dk, K]
    scalar_t*       __restrict__ d_topk_val, // [BH, N, K]
    const int BH, const int N, const int Dk)
{
    const int bh = blockIdx.x;
    const int q  = blockIdx.y;
    if (bh >= BH || q >= N) return;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    int*   sh_idx     = (int*)   smem_raw;                  // K
    float* sh_p       = (float*)(sh_idx + K_VAL);           // K (= softmax val)
    float* sh_dp_part = sh_p   + K_VAL;                     // BLOCK_THREADS * K (per-thread partials)

    // Load top-K idx/val
    if (tid < K_VAL) {
        sh_idx[tid] = topk_idx[(bh * N + q) * K_VAL + tid];
        sh_p  [tid] = to_float<scalar_t>(topk_val[(bh * N + q) * K_VAL + tid]);
    }
    // Each thread zeros its partial-dp slice
    #pragma unroll
    for (int k = 0; k < K_VAL; ++k) sh_dp_part[tid * K_VAL + k] = 0.0f;
    __syncthreads();

    // Strided over channels m. Each m gives:
    //   dy = grad_out[b,q,m]
    //   For each k:
    //     vk = V[b, I[k], m]
    //     w  = W[m, k]
    //     contrib_to_dV [b, I[k], m] += dy * p[k] * w     (atomic)
    //     contrib_to_dW [m, k]       += dy * p[k] * vk    (atomic)
    //     contrib_to_dp [k]          += dy * vk * w       (per-thread sum)
    //
    for (int m = tid; m < Dk; m += BLOCK_THREADS) {
        const float dy = to_float<scalar_t>(grad_out[(bh * N + q) * Dk + m]);

        float w_row[K_VAL];
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k)
            w_row[k] = to_float<scalar_t>(W[m * K_VAL + k]);

        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            const int   j  = sh_idx[k];
            const float p  = sh_p  [k];
            const float vk = to_float<scalar_t>(V[(bh * N + j) * Dk + m]);

            // dV: atomic into fp32 buffer
            atomicAdd(&dV_fp32[(bh * N + j) * Dk + m], dy * p * w_row[k]);

            // dW: atomic into fp32 buffer
            atomicAdd(&dW_fp32[m * K_VAL + k], dy * p * vk);

            // dp partial (per-thread, no atomic needed)
            sh_dp_part[tid * K_VAL + k] += dy * vk * w_row[k];
        }
    }
    __syncthreads();

    // Reduce sh_dp_part across threads to get final dp[k] per query
    if (tid < K_VAL) {
        float sum = 0.0f;
        for (int t = 0; t < BLOCK_THREADS; ++t)
            sum += sh_dp_part[t * K_VAL + tid];
        d_topk_val[(bh * N + q) * K_VAL + tid] = from_float<scalar_t>(sum);
    }
}

// =========================================================================
// PHASE 1 BACKWARD: from d_topk_softmax -> dQ, dK
// =========================================================================
//
// Let p_k = softmax(s)_k  for k in [0..K). Then
//   dp -> ds:   ds_k = p_k * (dp_k - sum_j p_j * dp_j)
// And s_k = (Q_q . K_{I_k}) / sqrt(Dk):
//   dQ_q   += sum_k ds_k * K_{I_k} / sqrt(Dk)
//   dK_{I_k} += ds_k * Q_q / sqrt(Dk)         (atomic over (q,k) collisions)
//
// One block per (bh, q). Each thread handles a strided slice of channels.
// -------------------------------------------------------------------------

template <typename scalar_t, int K_VAL, int BLOCK_THREADS>
__global__ void flash_topk_attn_bwd_kernel(
    const scalar_t* __restrict__ Q,           // [BH, N, Dk]
    const scalar_t* __restrict__ Kmat,        // [BH, N, Dk]
    const int32_t*  __restrict__ topk_idx,    // [BH, N, K]
    const scalar_t* __restrict__ topk_val,    // [BH, N, K]   p_k (softmax-applied)
    const scalar_t* __restrict__ d_topk_val,  // [BH, N, K]   dp_k
    float*          __restrict__ dQ_fp32,     // [BH, N, Dk]
    float*          __restrict__ dK_fp32,     // [BH, N, Dk]
    const int BH, const int N, const int Dk,
    const float scale)
{
    const int bh = blockIdx.x;
    const int q  = blockIdx.y;
    if (bh >= BH || q >= N) return;
    const int tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    int*   sh_idx = (int*)   smem_raw;          // K
    float* sh_p   = (float*)(sh_idx + K_VAL);   // K
    float* sh_dp  = sh_p   + K_VAL;             // K
    float* sh_ds  = sh_dp  + K_VAL;             // K

    // Load p_k and dp_k for this (bh, q)
    if (tid < K_VAL) {
        sh_idx[tid] = topk_idx  [(bh * N + q) * K_VAL + tid];
        sh_p  [tid] = to_float<scalar_t>(topk_val  [(bh * N + q) * K_VAL + tid]);
        sh_dp [tid] = to_float<scalar_t>(d_topk_val[(bh * N + q) * K_VAL + tid]);
    }
    __syncthreads();

    // Compute dot = sum_j p_j * dp_j  (single-thread, K is small)
    if (tid == 0) {
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) dot += sh_p[k] * sh_dp[k];
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k)
            sh_ds[k] = sh_p[k] * (sh_dp[k] - dot);
    }
    __syncthreads();

    // dQ_q[m] = sum_k ds_k * K[I_k, m] * scale
    // dK_{I_k}[m] += ds_k * Q_q[m] * scale     (atomic; q's overlap on k indices)
    for (int m = tid; m < Dk; m += BLOCK_THREADS) {
        const float qm = to_float<scalar_t>(Q[(bh * N + q) * Dk + m]);
        float dq_acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_VAL; ++k) {
            const int   j   = sh_idx[k];
            const float ds  = sh_ds[k];
            const float km  = to_float<scalar_t>(Kmat[(bh * N + j) * Dk + m]);
            dq_acc += ds * km;
            atomicAdd(&dK_fp32[(bh * N + j) * Dk + m], ds * qm * scale);
        }
        dQ_fp32[(bh * N + q) * Dk + m] = dq_acc * scale;
    }
}


// =========================================================================
// LAUNCHERS / DISPATCH
// =========================================================================

// Macro to dispatch on K
#define DISPATCH_K(K_RUNTIME, K_NAME, ...) \
    do { \
        switch (K_RUNTIME) { \
            case 4:  { constexpr int K_NAME = 4;  __VA_ARGS__; } break; \
            case 8:  { constexpr int K_NAME = 8;  __VA_ARGS__; } break; \
            case 16: { constexpr int K_NAME = 16; __VA_ARGS__; } break; \
            case 32: { constexpr int K_NAME = 32; __VA_ARGS__; } break; \
            default: TORCH_CHECK(false, "Unsupported K=", K_RUNTIME, ". Supported: 4, 8, 16, 32."); \
        } \
    } while (0)


// ---------- Forward Phase 1 launcher ----------
template <typename scalar_t, int K_VAL>
void launch_phase1_fwd(
    const scalar_t* Q, const scalar_t* Kmat,
    int32_t* topk_idx, scalar_t* topk_val,
    int BH, int N, int Dk, float scale, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 64;
    dim3 grid(BH, N);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (size_t)Dk * sizeof(float)
                 + (size_t)BLOCK_THREADS * sizeof(float);
    flash_topk_attn_fwd_kernel<scalar_t, K_VAL, BLOCK_THREADS>
        <<<grid, block, shmem, stream>>>(Q, Kmat, topk_idx, topk_val, BH, N, Dk, scale);
}

// ---------- Forward Phase 2 launcher ----------
template <typename scalar_t, int K_VAL>
void launch_phase2_fwd(
    const scalar_t* V, const int32_t* topk_idx, const scalar_t* topk_val,
    const scalar_t* W, scalar_t* out,
    int BH, int N, int Dk, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 64;
    dim3 grid(BH, N);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (size_t)K_VAL * sizeof(int) + (size_t)K_VAL * sizeof(float);
    flash_conv_aggregate_fwd_kernel<scalar_t, K_VAL, BLOCK_THREADS>
        <<<grid, block, shmem, stream>>>(V, topk_idx, topk_val, W, out, BH, N, Dk);
}

// ---------- Backward Phase 2 launcher ----------
template <typename scalar_t, int K_VAL>
void launch_phase2_bwd(
    const scalar_t* grad_out, const scalar_t* V,
    const int32_t* topk_idx, const scalar_t* topk_val, const scalar_t* W,
    float* dV_fp32, float* dW_fp32, scalar_t* d_topk_val,
    int BH, int N, int Dk, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 64;
    dim3 grid(BH, N);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (size_t)K_VAL * sizeof(int)            // sh_idx
                 + (size_t)K_VAL * sizeof(float)          // sh_p
                 + (size_t)BLOCK_THREADS * K_VAL * sizeof(float); // sh_dp_part
    flash_conv_aggregate_bwd_kernel<scalar_t, K_VAL, BLOCK_THREADS>
        <<<grid, block, shmem, stream>>>(
            grad_out, V, topk_idx, topk_val, W,
            dV_fp32, dW_fp32, d_topk_val, BH, N, Dk);
}

// ---------- Backward Phase 1 launcher ----------
template <typename scalar_t, int K_VAL>
void launch_phase1_bwd(
    const scalar_t* Q, const scalar_t* Kmat,
    const int32_t* topk_idx, const scalar_t* topk_val, const scalar_t* d_topk_val,
    float* dQ_fp32, float* dK_fp32,
    int BH, int N, int Dk, float scale, cudaStream_t stream)
{
    constexpr int BLOCK_THREADS = 64;
    dim3 grid(BH, N);
    dim3 block(BLOCK_THREADS);
    size_t shmem = (size_t)K_VAL * sizeof(int)        // sh_idx
                 + (size_t)K_VAL * sizeof(float) * 3; // sh_p, sh_dp, sh_ds
    flash_topk_attn_bwd_kernel<scalar_t, K_VAL, BLOCK_THREADS>
        <<<grid, block, shmem, stream>>>(
            Q, Kmat, topk_idx, topk_val, d_topk_val,
            dQ_fp32, dK_fp32, BH, N, Dk, scale);
}


// =========================================================================
// PUBLIC ENTRY POINTS (called from C++ binding layer)
// =========================================================================

std::vector<torch::Tensor> flash_topk_attn_fwd(
    torch::Tensor Q, torch::Tensor K, int K_top)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda(), "Q and K must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 3 && K.dim() == 3, "Q, K must be [BH, N, Dk]");
    TORCH_CHECK(Q.sizes() == K.sizes(), "Q and K shape mismatch");
    Q = Q.contiguous();
    K = K.contiguous();

    const int BH = Q.size(0);
    const int N  = Q.size(1);
    const int Dk = Q.size(2);
    TORCH_CHECK(Dk <= FLASH_MAX_DK, "Dk too large; recompile with larger FLASH_MAX_DK");
    TORCH_CHECK(K_top <= N, "K_top (", K_top, ") must be <= sequence length N (", N, ")");
    TORCH_CHECK(K_top > 0, "K_top must be positive");

    auto idx_opts = torch::TensorOptions().dtype(torch::kInt32).device(Q.device());
    auto val_opts = torch::TensorOptions().dtype(Q.scalar_type()).device(Q.device());
    auto topk_idx = torch::empty({BH, N, K_top}, idx_opts);
    auto topk_val = torch::empty({BH, N, K_top}, val_opts);

    const float scale = 1.0f / std::sqrt((float)Dk);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const c10::cuda::CUDAGuard device_guard(Q.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, Q.scalar_type(),
        "flash_topk_attn_fwd", ([&] {
            DISPATCH_K(K_top, K_VAL, ({
                launch_phase1_fwd<scalar_t, K_VAL>(
                    Q.data_ptr<scalar_t>(),
                    K.data_ptr<scalar_t>(),
                    topk_idx.data_ptr<int32_t>(),
                    topk_val.data_ptr<scalar_t>(),
                    BH, N, Dk, scale, stream);
            }));
        }));
    return {topk_idx, topk_val};
}


torch::Tensor flash_conv_aggregate_fwd(
    torch::Tensor V, torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor W /*[Dk, K]*/)
{
    TORCH_CHECK(V.is_cuda() && topk_idx.is_cuda() && topk_val.is_cuda() && W.is_cuda(),
                "All inputs must be CUDA tensors");
    TORCH_CHECK(V.dim() == 3, "V must be [BH, N, Dk]");
    V = V.contiguous();
    topk_idx = topk_idx.contiguous();
    topk_val = topk_val.contiguous();
    W = W.contiguous();

    const int BH = V.size(0), N = V.size(1), Dk = V.size(2);
    const int K_top = topk_idx.size(2);
    TORCH_CHECK(W.dim() == 2 && W.size(0) == Dk && W.size(1) == K_top,
                "W must be [Dk, K]");

    auto out = torch::empty_like(V);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const c10::cuda::CUDAGuard device_guard(V.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, V.scalar_type(),
        "flash_conv_aggregate_fwd", ([&] {
            DISPATCH_K(K_top, K_VAL, ({
                launch_phase2_fwd<scalar_t, K_VAL>(
                    V.data_ptr<scalar_t>(),
                    topk_idx.data_ptr<int32_t>(),
                    topk_val.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    out.data_ptr<scalar_t>(),
                    BH, N, Dk, stream);
            }));
        }));
    return out;
}


std::vector<torch::Tensor> flash_conv_aggregate_bwd(
    torch::Tensor grad_out, torch::Tensor V,
    torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor W)
{
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be CUDA");
    grad_out = grad_out.contiguous();
    V        = V.contiguous();
    topk_idx = topk_idx.contiguous();
    topk_val = topk_val.contiguous();
    W        = W.contiguous();

    const int BH = V.size(0), N = V.size(1), Dk = V.size(2);
    const int K_top = topk_idx.size(2);

    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(V.device());
    auto dV_fp32 = torch::zeros({BH, N, Dk}, fp32_opts);
    auto dW_fp32 = torch::zeros({Dk, K_top}, fp32_opts);
    auto d_topk_val = torch::empty_like(topk_val);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const c10::cuda::CUDAGuard device_guard(V.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, V.scalar_type(),
        "flash_conv_aggregate_bwd", ([&] {
            DISPATCH_K(K_top, K_VAL, ({
                launch_phase2_bwd<scalar_t, K_VAL>(
                    grad_out.data_ptr<scalar_t>(),
                    V.data_ptr<scalar_t>(),
                    topk_idx.data_ptr<int32_t>(),
                    topk_val.data_ptr<scalar_t>(),
                    W.data_ptr<scalar_t>(),
                    dV_fp32.data_ptr<float>(),
                    dW_fp32.data_ptr<float>(),
                    d_topk_val.data_ptr<scalar_t>(),
                    BH, N, Dk, stream);
            }));
        }));

    auto dV = dV_fp32.to(V.scalar_type());
    auto dW = dW_fp32.to(W.scalar_type());
    return {dV, dW, d_topk_val};
}


std::vector<torch::Tensor> flash_topk_attn_bwd(
    torch::Tensor Q, torch::Tensor Kmat,
    torch::Tensor topk_idx, torch::Tensor topk_val,
    torch::Tensor d_topk_val)
{
    TORCH_CHECK(Q.is_cuda() && Kmat.is_cuda(), "Q, K must be CUDA");
    Q = Q.contiguous();
    Kmat = Kmat.contiguous();
    topk_idx   = topk_idx.contiguous();
    topk_val   = topk_val.contiguous();
    d_topk_val = d_topk_val.contiguous();

    const int BH = Q.size(0), N = Q.size(1), Dk = Q.size(2);
    const int K_top = topk_idx.size(2);
    const float scale = 1.0f / std::sqrt((float)Dk);

    auto fp32_opts = torch::TensorOptions().dtype(torch::kFloat32).device(Q.device());
    auto dQ_fp32 = torch::zeros({BH, N, Dk}, fp32_opts);
    auto dK_fp32 = torch::zeros({BH, N, Dk}, fp32_opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const c10::cuda::CUDAGuard device_guard(Q.device());

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, Q.scalar_type(),
        "flash_topk_attn_bwd", ([&] {
            DISPATCH_K(K_top, K_VAL, ({
                launch_phase1_bwd<scalar_t, K_VAL>(
                    Q.data_ptr<scalar_t>(),
                    Kmat.data_ptr<scalar_t>(),
                    topk_idx.data_ptr<int32_t>(),
                    topk_val.data_ptr<scalar_t>(),
                    d_topk_val.data_ptr<scalar_t>(),
                    dQ_fp32.data_ptr<float>(),
                    dK_fp32.data_ptr<float>(),
                    BH, N, Dk, scale, stream);
            }));
        }));

    auto dQ = dQ_fp32.to(Q.scalar_type());
    auto dK = dK_fp32.to(Q.scalar_type());
    return {dQ, dK};
}
