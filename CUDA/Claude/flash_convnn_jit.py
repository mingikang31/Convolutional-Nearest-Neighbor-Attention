"""
flash_convnn_jit.py — single-file Flash CUDA ConvNN-Attention.
Run:  python flash_convnn_jit.py
First run JIT-compiles via torch.utils.cpp_extension.load_inline (~30s).
Subsequent runs load from cache (~/.cache/torch_extensions/).
Depthwise only.  Supports D_K in {32,64,96,128}, K_TOP in {4,8,16,32}.
"""

import os, time, math
import torch, torch.nn as nn
from torch.utils.cpp_extension import load_inline

assert torch.cuda.is_available(), "CUDA required."

# ============================================================================
# CUDA / C++ source
# ============================================================================
CPP_DECL = r"""
#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> flash_dw_fwd(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                        torch::Tensor W, int64_t K_top, double scale);
std::vector<torch::Tensor> flash_dw_bwd_vw(torch::Tensor GO, torch::Tensor V, torch::Tensor W,
                                           torch::Tensor TKIDX, torch::Tensor TKVAL);
std::vector<torch::Tensor> flash_dw_bwd_qk(torch::Tensor Q, torch::Tensor K,
                                           torch::Tensor TKIDX, torch::Tensor GTV_RAW,
                                           double scale);
"""

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

#define NEG_INF (-1e30f)

template<typename T> __device__ __forceinline__ float to_f32(T x)  { return static_cast<float>(x); }
template<typename T> __device__ __forceinline__ T    from_f32(float x){ return static_cast<T>(x); }

// Block reduction (BLOCK must be multiple of 32, > 0)
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
    return v;  // valid in thread 0
}

// =============================================================================
// FUSED FORWARD KERNEL  (depthwise)
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

    // Phase 1 — scores[j] = Q[t]·K[j] * scale
    for (int j = tid; j < N; j += BLOCK) {
        const T* k_row = Kt + (bh*N + j)*D_K;
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < D_K; ++d) s += q_sm[d] * to_f32<T>(k_row[d]);
        scores[j] = s * scale;
    }
    __syncthreads();

    // Phase 2 — top-K via K_TOP rounds of block-wide arg-max
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

    // Phase 3 — softmax over K_TOP survivors
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

    // Phase 4 — persist topk_idx / topk_val for backward
    if (tid < K_TOP) {
        int off = (bh*N + t)*K_TOP + tid;
        TKIDX[off] = tk_idx[tid];
        TKVAL[off] = from_f32<T>(tk_val[tid]);
    }

    // Phase 5 — gather V + depthwise conv
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
// BACKWARD V/W KERNEL  (depthwise, with fused softmax-bwd)
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

            // grad_topk_val_post[k] = <go, v⊙w>  (block reduce across D_K)
            float p = blockReduceSum<D_K>(go * v * w);
            if (d == 0) gpost[k] = p;

            // grad_V[bh,idx,d] += go * a * w  (atomic, scattered)
            atomicAdd(&GV[(bh*N + idx)*D_K + d], go * a * w);

            // grad_W[d,k] register accumulation across T_BLOCK queries
            gw_local[k] += go * v * a;
        }
        __syncthreads();

        // Softmax backward over K_TOP entries:  dx[k] = y[k] * (dy[k] - <y,dy>)
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

    // Single atomic_add per (d,k) per block — T_BLOCK× contention drop on GW.
    #pragma unroll
    for (int k = 0; k < K_TOP; ++k) atomicAdd(&GW[d*K_TOP + k], gw_local[k]);
}

// =============================================================================
// BACKWARD Q/K KERNEL  (sparse attention bwd — only K of N per row are non-zero)
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
    TORCH_CHECK(W.dim() == 2 && W.size(0) == D_K && W.size(1) == K_top,
                "W must be (D_K, K_top)");

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
"""

print("Compiling CUDA extension (first run only, ~30s)...")
_t0 = time.time()
EXT = load_inline(
    name="flash_convnn_jit_v1",
    cpp_sources=CPP_DECL,
    cuda_sources=CUDA_SRC,
    functions=["flash_dw_fwd", "flash_dw_bwd_vw", "flash_dw_bwd_qk"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++17"],
    extra_cflags=["-O3", "-std=c++17"],
    verbose=False,
)
print(f"Compiled in {time.time() - _t0:.1f}s.")


# ============================================================================
# Python autograd Function + Module
# ============================================================================
class _FlashConvNNFn(torch.autograd.Function):
    """Forward/backward through Q,K,V,W → out  for depthwise ConvNN-attention."""
    @staticmethod
    def forward(ctx, Q, K, V, W, scale):
        # Q,K,V: (B*NH, N, D_K)   W: (D_K, K_TOP)
        K_top = W.size(1)
        out, tkidx, tkval = EXT.flash_dw_fwd(Q, K, V, W, K_top, scale)
        ctx.save_for_backward(Q, K, V, W, tkidx, tkval)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        Q, K, V, W, tkidx, tkval = ctx.saved_tensors
        # grad_V, grad_W (in W-layout), grad_topk_val_raw (post softmax-bwd)
        gV, gW, gTV_raw = EXT.flash_dw_bwd_vw(grad_out.contiguous(), V, W, tkidx, tkval)
        # grad_Q, grad_K from sparse-attention bwd
        gQ, gK = EXT.flash_dw_bwd_qk(Q, K, tkidx, gTV_raw, ctx.scale)
        return (gQ.to(Q.dtype), gK.to(K.dtype), gV.to(V.dtype), gW.to(W.dtype), None)


class FlashCUDAMultiHeadConvNNAttention(nn.Module):
    """Drop-in for FastMultiHeadConvNNAttention. Depthwise only.
    Mathematically equivalent up to atomic-add summation order."""
    def __init__(self, d_hidden, num_heads, attention_dropout, K, seq_length=197):
        super().__init__()
        assert d_hidden % num_heads == 0
        self.d_hidden, self.num_heads = d_hidden, num_heads
        self.d_k = d_hidden // num_heads
        self.K = K
        self.seq_length = seq_length
        assert self.d_k in (32, 64, 96, 128), f"D_K={self.d_k} not in supported set"
        assert K in (4, 8, 16, 32),           f"K={K} not in supported set"

        self.W_q = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_k = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_v = nn.Linear(d_hidden, d_hidden, bias=False)
        self.W_o = nn.Linear(d_hidden, d_hidden, bias=False)
        self.dropout = nn.Dropout(attention_dropout)
        # depthwise weight, stored as (d_k, K) — the conv kernel sees it directly.
        self.conv_weight = nn.Parameter(torch.ones(self.d_k, self.K))

    def _split_head(self, x):
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.d_k).transpose(1, 2)  # (B, NH, N, d_k)

    def forward(self, x):
        B, N, _ = x.shape
        q = self._split_head(self.W_q(x))
        k = self._split_head(self.W_k(x))
        v = self._split_head(self.W_v(x))
        q = q.reshape(B * self.num_heads, N, self.d_k).contiguous()
        k = k.reshape(B * self.num_heads, N, self.d_k).contiguous()
        v = v.reshape(B * self.num_heads, N, self.d_k).contiguous()

        scale = 1.0 / math.sqrt(self.d_k)
        out = _FlashConvNNFn.apply(q, k, v, self.conv_weight, scale)

        out = out.view(B, self.num_heads, N, self.d_k).transpose(1, 2).contiguous().view(B, N, self.d_hidden)
        return self.W_o(self.dropout(out))


# ============================================================================
# Reference (pure PyTorch) for correctness
# ============================================================================
def _ref_convnn(Q, K, V, W, scale):
    """Pure-PyTorch reference: same math as the CUDA forward."""
    K_top = W.size(1)
    attn = (Q @ K.transpose(-2, -1)) * scale            # (B*NH, N, N)
    tv, ti = torch.topk(attn, K_top, dim=-1)            # (B*NH, N, K)
    tv = torch.softmax(tv, dim=-1)
    # gather V → (B*NH, N, K, D)
    B_NH, N, D = V.shape
    Vg = torch.gather(V.unsqueeze(1).expand(-1, N, -1, -1), 2,
                      ti.unsqueeze(-1).expand(-1, -1, -1, D))
    # out[b,t,d] = sum_k tv[b,t,k] * Vg[b,t,k,d] * W[d,k]
    return torch.einsum('btkd,btk,dk->btd', Vg, tv, W)


# ============================================================================
# Self-test
# ============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda")
    B_NH, N, D_K, K_top = 16, 197, 64, 8
    print(f"\nTesting:  B*NH={B_NH}  N={N}  D_K={D_K}  K_TOP={K_top}\n")

    Q = torch.randn(B_NH, N, D_K, device=dev, dtype=torch.float32, requires_grad=True)
    Kt = torch.randn(B_NH, N, D_K, device=dev, dtype=torch.float32, requires_grad=True)
    V = torch.randn(B_NH, N, D_K, device=dev, dtype=torch.float32, requires_grad=True)
    W = torch.randn(D_K, K_top, device=dev, dtype=torch.float32, requires_grad=True)
    scale = 1.0 / math.sqrt(D_K)

    # Ref clones for independent autograd graphs
    Qr, Kr, Vr, Wr = (t.detach().clone().requires_grad_(True) for t in (Q, Kt, V, W))

    # Forward
    out_flash = _FlashConvNNFn.apply(Q, Kt, V, W, scale)
    out_ref   = _ref_convnn(Qr, Kr, Vr, Wr, scale)
    print(f"forward  max|Δ|         = {(out_flash - out_ref).abs().max().item():.3e}")

    # Backward
    g = torch.randn_like(out_flash)
    out_flash.backward(g)
    out_ref.backward(g)
    print(f"grad_Q   max|Δ|         = {(Q.grad  - Qr.grad).abs().max().item():.3e}")
    print(f"grad_K   max|Δ|         = {(Kt.grad - Kr.grad).abs().max().item():.3e}")
    print(f"grad_V   max|Δ|         = {(V.grad  - Vr.grad).abs().max().item():.3e}")
    print(f"grad_W   max|Δ|         = {(W.grad  - Wr.grad).abs().max().item():.3e}")

    # Benchmark
    print("\nBenchmark (50 iters, fp32):")
    def bench(fn, name):
        for _ in range(5): fn()  # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(50): fn()
        torch.cuda.synchronize()
        print(f"  {name:18s}  {(time.time() - t0) * 1000 / 50:6.2f} ms/iter")

    Q.grad = None; Kt.grad = None; V.grad = None; W.grad = None
    Qr.grad = None; Kr.grad = None; Vr.grad = None; Wr.grad = None

    bench(lambda: _FlashConvNNFn.apply(Q, Kt, V, W, scale).sum().backward(),
          "Flash-CUDA fwd+bwd")
    bench(lambda: _ref_convnn(Qr, Kr, Vr, Wr, scale).sum().backward(),
          "PyTorch ref fwd+bwd")

    # Memory
    torch.cuda.reset_peak_memory_stats()
    _FlashConvNNFn.apply(Q, Kt, V, W, scale).sum().backward()
    flash_mem = torch.cuda.max_memory_allocated() / 1024**2

    Q.grad = None; Kt.grad = None; V.grad = None; W.grad = None
    torch.cuda.reset_peak_memory_stats()
    _ref_convnn(Qr, Kr, Vr, Wr, scale).sum().backward()
    ref_mem = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\n  Flash-CUDA peak mem: {flash_mem:6.1f} MB")
    print(f"  PyTorch ref peak  mem: {ref_mem:6.1f} MB  ({ref_mem/flash_mem:.2f}× more)")