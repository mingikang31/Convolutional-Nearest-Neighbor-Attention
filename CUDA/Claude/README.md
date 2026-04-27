# FlashConvNN-Attention (CUDA)

A FlashAttention-style fused CUDA implementation of the depthwise variant of
ConvNN-Attention from your honors-thesis work. The key win over your current
Triton implementation is **never materializing the `[B*NH, N, N]` attention
matrix**: Q@Kᵀ is streamed in tiles, the per-query top-K is maintained online
in registers, and softmax is applied to only the K survivors.

## Files

```
flash_convnn/
├── csrc/
│   ├── flash_convnn_attention_kernel.cu   # 4 CUDA kernels (fwd/bwd × phase 1/2)
│   └── flash_convnn_attention.cpp         # pybind11 module
├── flash_convnn_attention.py              # autograd.Function + nn.Module
├── setup.py                               # build script (A100 + H100)
├── test_flash.py                          # correctness + benchmark
└── README.md
```

## What got fused

Your current Triton path does:

```
1. q,k,v projections                                         (PyTorch)
2. scores = Q @ K^T / sqrt(d_k)         -> [B*NH, N, N]      (PyTorch matmul)
3. topk_v, topk_i = torch.topk(scores)  -> [B*NH, N, K]      (PyTorch op)
4. p = softmax(topk_v)                                       (PyTorch op)
5. fused gather V + p * w + sum_K       -> output            (Triton)
```

The Flash path collapses steps 2–4 into one CUDA kernel that never writes the
`N²` scores tensor. Step 5 is rewritten in CUDA with the same fusion as your
Triton kernel, but with explicit register tiling of the depthwise weight row
`w[m, :]`.

| | current (Triton) | flash (CUDA) |
|---|---|---|
| forward kernel launches | ~5–7 | 2 |
| backward kernel launches | ~5–7 | 2 |
| `[BH, N, N]` materialized | yes | **no** |
| top-K reads N² | yes | **no** (online) |
| dtype path | fp16/bf16 with fp32 accum | same |
| AMP-safe | yes (`custom_fwd`/`custom_bwd`) | same |

## Math implemented (depthwise)

```
y[b, q, m] = Σ_{k=0..K-1} softmax(s_q)_k · V[b, I_q[k], m] · w[m, k]
s_q[j]     = (Q_q · K_j) / √d_k
I_q        = argkmax_K(s_q)
```

This matches the depthwise instantiation in §3 of your thesis draft and is
mathematically equivalent (within fp16/bf16 rounding) to
`MultiHeadConvNNAttention(convolution_type='depthwise')` and to
`FastMultiHeadConvNNAttention(convolution_type='depthwise')`.

The standard / cross-channel mixing variant (`convolution_type='standard'`)
has a different memory profile (`[d_k, d_k, K]` weight, full mat-vec per K
step) and is **not** in this kernel — keep using the Triton version for that.

## Build

```bash
# from inside flash_convnn/
pip install -e .
```

The build defaults to `sm_80` (A100) and `sm_90` (H100). For other targets
(e.g. 4090 = `sm_89`, V100 = `sm_70`), set `TORCH_CUDA_ARCH_LIST` before
install:

```bash
TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" pip install -e .
```

## Use

Drop-in replacement for `FastMultiHeadConvNNAttention(convolution_type='depthwise')`:

```python
from flash_convnn_attention import FlashMultiHeadConvNNAttention

attn = FlashMultiHeadConvNNAttention(
    d_hidden=768, num_heads=12, attention_dropout=0.1,
    K=8, seq_length=197,
).cuda()
y = attn(x)   # x: [B, 197, 768]
```

Weights in `attn.conv_weight` use the same `[d_k, 1, K]` layout as
`nn.Conv1d`, so any checkpoint trained with `MultiHeadConvNNAttention` or
`FastMultiHeadConvNNAttention` (depthwise) loads directly.

## Test

```bash
python test_flash.py
```

Runs:
- per-phase numerical correctness (Q,K,V,W with autograd) at fp32 and fp16
  for K ∈ {4, 8, 16}
- module-level equivalence vs. the Triton `FastMultiHeadConvNNAttention`
- microbenchmarks at ViT-Base settings (`N=197`) and longer sequences
  (`N=1024`, `N=4096`) where the flash advantage is much larger

## What to expect

At `N=197` (ViT-Base), the absolute speedup over your Triton kernel will be
modest — probably **1.3–2×** on forward and similar on backward — because
the `[B*NH, N, N]` matrix at `N=197` is only ~150 KB per head and fits in
L2 cache anyway. The win is mostly from kernel-launch reduction and the
fused softmax-over-top-K.

At `N=1024` and beyond, the flash advantage compounds quickly. Memory
savings should track `O(B·NH·(N² − N·K))` floats — for `N=4096, K=8`,
that's roughly **500× less attention-matrix memory** per head.

## Known limitations / future work

1. **Top-K merge is single-threaded inside each block** (thread 0 does the
   insert into the K-element register array). For large K this becomes the
   bottleneck. A bitonic top-K across the warp would scale better — happy
   to write that next if K=32+ ends up dominating profile traces.

2. **dK uses atomicAdd** into an fp32 buffer because multiple queries can
   land on the same key. For a single attention layer at `N=197, K=8`,
   atomic contention is low (each key is hit ≈8 times on average), but at
   high `N·K/N = K`, contention is bounded. If profile shows it as a
   bottleneck, the standard fix is to invert the loop: parallelize over
   `(b, j)` and gather contributions from queries that selected `j`. This
   needs a transpose-style scatter/sort that I haven't written here.

3. **Per-head dim cap**: `FLASH_MAX_DK = 128` (for register and shared
   memory budgeting). ViT-Base's `d_k = 64` and ViT-Large's `d_k = 64` are
   both fine. If you ever go beyond 128, bump the constant and recompile.

4. **K is templated at compile time** — supported values are 4, 8, 16, 32.
   Other K values raise an error. To add a new K (e.g. K=64), add a case to
   `DISPATCH_K` in the .cu file.

5. **No causal mask** — full bidirectional attention only. Easy to add: an
   extra `bool causal` flag in phase-1 fwd/bwd; mask out keys with
   `j_global > q_idx` before insertion.

## Profiling tips

```bash
# Roofline / per-kernel breakdown
ncu --set full --target-processes all -o flash_profile python test_flash.py

# Quick timeline
nsys profile -o flash_timeline python test_flash.py
```

The two kernels of interest are
`flash_topk_attn_fwd_kernel` and `flash_conv_aggregate_fwd_kernel`. If the
first dominates, the win is the `BLOCK_THREADS` × `CHUNK` choice; if the
second dominates, look at the `dW` atomicAdd contention and the strided
channel loop.
