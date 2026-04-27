#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define NEG_INF (-1e30f)

template<typename T> __device__ __forceinline__ float to_f32(T x)  { return static_cast<float>(x); }
template<typename T> __device__ __forceinline__ T    from_f32(float x){ return static_cast<T>(x); }

template<int BLOCK>
__device__ __forceinline__ float blockReduceSum(float v) {
    constexpr int NW_ = BLOCK / 32;
    constexpr int NW  = (NW_ > 0 ? NW_ : 1);
    __shared__ float ws[NW];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if constexpr (NW_ <= 1) return v;
    if (lane == 0) ws[wid] = v;
    __syncthreads();
    v = (threadIdx.x < NW) ? ws[lane] : 0.f;
    if (wid == 0) {
        #pragma unroll
        for (int o = NW / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    }
    return v;
}

// =============================================================================
// FUSED FORWARD KERNEL (depthwise)
// =============================================================================
template<typename T, int D_K, int K_TOP, int BLOCK>
__global__ void __launch_bounds__(BLOCK)
fwd_kernel(const T* __restrict__ Q,  const T* __restrict__ Kt,
           const T* __restrict__ V,  const T* __restrict__ W,
           int* __restrict__ TKIDX, T* __restrict__ TKVAL,
           T* __restrict__ OUT, int N, float scale)
{
    static_assert(BLOCK >= D_K && BLOCK % 32 == 0, "");
    constexpr int NW = BLOCK / 32;
    int bh = blockIdx.y, t = blockIdx.x, tid = threadIdx.x;

    extern __shared__ char smem_raw[];
    float* scores = (float*)smem_raw;
    float* tk_val = scores + N;
    int*   tk_idx = (int*)(tk_val + K_TOP);

    __shared__ float q_sm[D_K];
    __shared__ float wmax[NW];
    __shared__ int   warg[NW];

    if (tid < D_K) q_sm[tid] = to_f32<T>(Q[(bh*N + t)*D_K + tid]);
    __syncthreads();

    for (int j = tid; j < N; j += BLOCK) {
        const T* k_row = Kt + (bh*N + j)*D_K;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < D_K; ++d) s += q_sm[d] * to_f32<T>(k_row[d]);
        scores[j] = s * scale;
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < K_TOP; ++k) {
        float lmax = NEG_INF; int larg = -1;
        for (int j = tid; j < N; j += BLOCK) {
            float s = scores[j];
            if (s > lmax) { lmax = s; larg = j; }
        }
        unsigned m = 0xFFFFFFFFu;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            float om = __shfl_xor_sync(m, lmax, o);
            int   oa = __shfl_xor_sync(m, larg, o);
            if (om > lmax) { lmax = om; larg = oa; }
        }
        int lane = tid & 31, wid = tid >> 5;
        if (lane == 0) { wmax[wid] = lmax; warg[wid] = larg; }
        __syncthreads();
        if (wid == 0) {
            float wm = (lane < NW) ? wmax[lane] : NEG_INF;
            int   wa = (lane < NW) ? warg[lane] : -1;
            #pragma unroll
            for (int o = NW / 2; o > 0; o >>= 1) {
                float om = __shfl_xor_sync(m, wm, o);
                int   oa = __shfl_xor_sync(m, wa, o);
                if (om > wm) { wm = om; wa = oa; }
            }
            if (lane == 0) { tk_val[k] = wm; tk_idx[k] = wa; scores[wa] = NEG_INF; }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mx = tk_val[0];
        #pragma unroll
        for (int k = 1; k < K_TOP; ++k) if (tk_val[k] > mx) mx = tk_val[k];
        float sum = 0.f;
        #pragma unroll
        for (int k = 0; k < K_TOP; ++k) { tk_val[k] = expf(tk_val[k] - mx); sum += tk_val[k]; }
        float inv = 1.f / sum;
        #pragma unroll
        for (int k = 0; k < K_TOP; ++k) tk_val[k] *= inv;
    }
    __syncthreads();

    if (tid < K_TOP) {
        int off = (bh*N + t)*K_TOP + tid;
        TKIDX[off] = tk_idx[tid];
        TKVAL[off] = from_f32<T>(tk_val[tid]);
    }

    if (tid < D_K) {
        int d = tid;
        float acc = 0.f;
        #pragma unroll
        for (int k = 0; k < K_TOP; ++k) {
            int   idx = tk_idx[k];
            float v   = to_f32<T>(V[(bh*N + idx)*D_K + d]);
            float a   = tk_val[k];
            float w   = to_f32<T>(W[d*K_TOP + k]);
            acc += v * a * w;
        }
        OUT[(bh*N + t)*D_K + d] = from_f32<T>(acc);
    }
}

// =============================================================================
// BACKWARD V/W KERNEL (depthwise, fused softmax-bwd)
// =============================================================================
template<typename T, int D_K, int K_TOP, int T_BLOCK>
__global__ void __launch_bounds__(D_K)
bwd_vw_kernel(const T* __restrict__ GO, const T* __restrict__ V, const T* __restrict__ W,
              const int* __restrict__ TKIDX, const T* __restrict__ TKVAL,
              float* __restrict__ GV, float* __restrict__ GW, float* __restrict__ GTV,
              int N)
{
    static_assert(D_K % 32 == 0, "");
    int bh = blockIdx.y, t_base = blockIdx.x * T_BLOCK, d = threadIdx.x;

    float gw_local[K_TOP];
    #pragma unroll
    for (int k = 0; k < K_TOP; ++k) gw_local[k] = 0.f;

    __shared__ float gpost[K_TOP];
    __shared__ float val_sm[K_TOP];

    #pragma unroll 1
    for (int to_ = 0; to_ < T_BLOCK; ++to_) {
        int t = t_base + to_;
        if (t >= N) break;

        float go = to_f32<T>(GO[(bh*N + t)*D_K + d]);
        if (d < K_TOP) val_sm[d] = to_f32<T>(TKVAL[(bh*N + t)*K_TOP + d]);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < K_TOP; ++k) {
            int   idx = TKIDX[(bh*N + t)*K_TOP + k];
            float a   = val_sm[k];
            float v   = to_f32<T>(V[(bh*N + idx)*D_K + d]);
            float w   = to_f32<T>(W[d*K_TOP + k]);

            float p = blockReduceSum<D_K>(go * v * w);
            if (d == 0) gpost[k] = p;

            atomicAdd(&GV[(bh*N + idx)*D_K + d], go * a * w);
            gw_local[k] += go * v * a;
        }
        __syncthreads();

        if (d == 0) {
            float dot = 0.f;
            #pragma unroll
            for (int k = 0; k < K_TOP; ++k) dot += val_sm[k] * gpost[k];
            #pragma unroll
            for (int k = 0; k < K_TOP; ++k)
                GTV[(bh*N + t)*K_TOP + k] = val_sm[k] * (gpost[k] - dot);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < K_TOP; ++k) atomicAdd(&GW[d*K_TOP + k], gw_local[k]);
}

// =============================================================================
// BACKWARD Q/K KERNEL (sparse attention bwd)
// =============================================================================
template<typename T, int D_K, int K_TOP>
__global__ void __launch_bounds__(D_K)
bwd_qk_kernel(const T* __restrict__ Q, const T* __restrict__ Kt,
              const int* __restrict__ TKIDX, const float* __restrict__ GTV_RAW,
              float* __restrict__ GQ, float* __restrict__ GK,
              int N, float scale)
{
    int bh = blockIdx.y, t = blockIdx.x, d = threadIdx.x;
    float q = to_f32<T>(Q[(bh*N + t)*D_K + d]);
    float gq_acc = 0.f;
    #pragma unroll 1
    for (int k = 0; k < K_TOP; ++k) {
        int   idx = TKIDX[(bh*N + t)*K_TOP + k];
        float gv  = GTV_RAW[(bh*N + t)*K_TOP + k] * scale;
        float kv  = to_f32<T>(Kt[(bh*N + idx)*D_K + d]);
        gq_acc += gv * kv;
        atomicAdd(&GK[(bh*N + idx)*D_K + d], gv * q);
    }
    GQ[(bh*N + t)*D_K + d] = gq_acc;
}

// =============================================================================
// HOST DISPATCH
// =============================================================================
#define DISPATCH_DK(D, ...)  do { switch (D) { \
    case 32:  { constexpr int kDK = 32;  __VA_ARGS__; break; } \
    case 64:  { constexpr int kDK = 64;  __VA_ARGS__; break; } \
    case 96:  { constexpr int kDK = 96;  __VA_ARGS__; break; } \
    case 128: { constexpr int kDK = 128; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported D_K=", D); } } while (0)

#define DISPATCH_KT(K, ...)  do { switch (K) { \
    case 4:  { constexpr int kKT = 4;  __VA_ARGS__; break; } \
    case 8:  { constexpr int kKT = 8;  __VA_ARGS__; break; } \
    case 16: { constexpr int kKT = 16; __VA_ARGS__; break; } \
    case 32: { constexpr int kKT = 32; __VA_ARGS__; break; } \
    default: TORCH_CHECK(false, "unsupported K_TOP=", K); } } while (0)

#define FWD_BLOCK 128
constexpr int BWD_T_BLOCK = 8;

std::vector<torch::Tensor> flash_dw_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                        torch::Tensor W, int64_t K_top, double scale)
{
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda() && W.is_cuda(), "all CUDA");
    Q = Q.contiguous(); K = K.contiguous(); V = V.contiguous(); W = W.contiguous();
    int B_NH = Q.size(0), N = Q.size(1), D_K = Q.size(2);
    TORCH_CHECK(W.dim() == 2 && W.size(0) == D_K && W.size(1) == K_top, "W must be (D_K, K_top)");

    const c10::cuda::CUDAGuard guard(Q.device());
    auto OUT   = torch::empty_like(Q);
    auto TKIDX = torch::empty({B_NH, N, (int)K_top},
                              torch::TensorOptions().device(Q.device()).dtype(torch::kInt32));
    auto TKVAL = torch::empty({B_NH, N, (int)K_top}, Q.options());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    DISPATCH_DK(D_K, DISPATCH_KT((int)K_top,
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, Q.scalar_type(), "fwd", [&] {
            dim3 grid(N, B_NH), block(FWD_BLOCK);
            size_t smem = (N + kKT) * sizeof(float) + kKT * sizeof(int);
            fwd_kernel<scalar_t, kDK, kKT, FWD_BLOCK><<<grid, block, smem, stream>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                V.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
                TKIDX.data_ptr<int>(),  TKVAL.data_ptr<scalar_t>(),
                OUT.data_ptr<scalar_t>(), N, (float)scale);
        });
    ));
    return {OUT, TKIDX, TKVAL};
}

std::vector<torch::Tensor> flash_dw_bwd_vw(torch::Tensor GO, torch::Tensor V, torch::Tensor W,
                                           torch::Tensor TKIDX, torch::Tensor TKVAL)
{
    GO = GO.contiguous(); V = V.contiguous(); W = W.contiguous();
    TKIDX = TKIDX.contiguous(); TKVAL = TKVAL.contiguous();
    int B_NH = V.size(0), N = V.size(1), D_K = V.size(2);
    int K_top = TKIDX.size(2);

    const c10::cuda::CUDAGuard guard(V.device());
    auto fopts = V.options().dtype(torch::kFloat);
    auto GV  = torch::zeros_like(V, fopts);
    auto GW  = torch::zeros_like(W, fopts);
    auto GTV = torch::empty({B_NH, N, K_top}, fopts);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    DISPATCH_DK(D_K, DISPATCH_KT(K_top,
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, V.scalar_type(), "bwd_vw", [&] {
            int t_blocks = (N + BWD_T_BLOCK - 1) / BWD_T_BLOCK;
            dim3 grid(t_blocks, B_NH), block(kDK);
            bwd_vw_kernel<scalar_t, kDK, kKT, BWD_T_BLOCK><<<grid, block, 0, stream>>>(
                GO.data_ptr<scalar_t>(), V.data_ptr<scalar_t>(), W.data_ptr<scalar_t>(),
                TKIDX.data_ptr<int>(),   TKVAL.data_ptr<scalar_t>(),
                GV.data_ptr<float>(),    GW.data_ptr<float>(), GTV.data_ptr<float>(), N);
        });
    ));
    return {GV, GW, GTV};
}

std::vector<torch::Tensor> flash_dw_bwd_qk(torch::Tensor Q, torch::Tensor K,
                                           torch::Tensor TKIDX, torch::Tensor GTV_RAW,
                                           double scale)
{
    Q = Q.contiguous(); K = K.contiguous();
    TKIDX = TKIDX.contiguous(); GTV_RAW = GTV_RAW.contiguous();
    int B_NH = Q.size(0), N = Q.size(1), D_K = Q.size(2);
    int K_top = TKIDX.size(2);

    const c10::cuda::CUDAGuard guard(Q.device());
    auto fopts = Q.options().dtype(torch::kFloat);
    auto GQ = torch::empty_like(Q, fopts);
    auto GK = torch::zeros_like(K, fopts);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    DISPATCH_DK(D_K, DISPATCH_KT(K_top,
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, Q.scalar_type(), "bwd_qk", [&] {
            dim3 grid(N, B_NH), block(kDK);
            bwd_qk_kernel<scalar_t, kDK, kKT><<<grid, block, 0, stream>>>(
                Q.data_ptr<scalar_t>(), K.data_ptr<scalar_t>(),
                TKIDX.data_ptr<int>(),  GTV_RAW.data_ptr<float>(),
                GQ.data_ptr<float>(),   GK.data_ptr<float>(), N, (float)scale);
        });
    ));
    return {GQ, GK};
}

// ============================== PYBIND ======================================
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("flash_dw_fwd",    &flash_dw_fwd,    "Flash ConvNN depthwise forward");
    m.def("flash_dw_bwd_vw", &flash_dw_bwd_vw, "Flash ConvNN depthwise V/W backward");
    m.def("flash_dw_bwd_qk", &flash_dw_bwd_qk, "Flash ConvNN depthwise Q/K backward");
}