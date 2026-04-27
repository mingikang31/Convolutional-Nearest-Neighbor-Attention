"""
test_flash.py

Correctness + microbenchmark test for FlashMultiHeadConvNNAttention.

Compares against:
  1. A pure PyTorch reference (the math).
  2. FastMultiHeadConvNNAttention (your current Triton implementation).

Run after `pip install -e .` from the project root, with:
    python test_flash.py

Notes on tolerances:
  - fp32: rtol=1e-4, atol=1e-4
  - fp16/bf16: rtol=1e-2, atol=1e-2 (top-K can have ties at low precision,
                so the indices can disagree even when outputs are within tol;
                we test against the reference which uses the same top-K).
"""
import math
import time
import torch
import torch.nn as nn

from flash_convnn_attention import (
    FlashMultiHeadConvNNAttention,
    FlashConvNNFunction,
    reference_flash_forward,
)

# Import your existing Triton implementation for head-to-head benchmark.
# Adjust the import path to wherever you keep FastConvNNAttention.py.
try:
    from FastConvNNAttention import FastMultiHeadConvNNAttention
    HAVE_FAST = True
except ImportError:
    print("[warn] Could not import FastConvNNAttention; skipping head-to-head benchmark")
    HAVE_FAST = False


# ----------------------- correctness ---------------------------------

def test_phase_correctness(BH=4, N=197, d_k=64, K=8, dtype=torch.float32):
    print(f"\n[correctness:phase] BH={BH} N={N} d_k={d_k} K={K} dtype={dtype}")
    torch.manual_seed(0)
    Q = torch.randn(BH, N, d_k, device="cuda", dtype=dtype, requires_grad=True)
    K_ = torch.randn(BH, N, d_k, device="cuda", dtype=dtype, requires_grad=True)
    V = torch.randn(BH, N, d_k, device="cuda", dtype=dtype, requires_grad=True)
    W = torch.randn(d_k, K, device="cuda", dtype=dtype, requires_grad=True)

    # Reference (PyTorch math)
    Q_ref = Q.detach().clone().requires_grad_()
    K_ref = K_.detach().clone().requires_grad_()
    V_ref = V.detach().clone().requires_grad_()
    W_ref = W.detach().clone().requires_grad_()
    out_ref, _, _ = reference_flash_forward(Q_ref, K_ref, V_ref, W_ref)
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Flash CUDA
    out = FlashConvNNFunction.apply(Q, K_, V, W)
    loss = out.sum()
    loss.backward()

    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)

    def chk(name, a, b):
        d = (a - b).abs().max().item()
        ok = torch.allclose(a, b, rtol=rtol, atol=atol)
        print(f"  {name:6s}: max_abs_diff={d:.2e} {'OK' if ok else 'MISMATCH'}")
        return ok

    ok_out = chk("out",  out, out_ref)
    ok_dQ  = chk("dQ",   Q.grad,  Q_ref.grad)
    ok_dK  = chk("dK",   K_.grad, K_ref.grad)
    ok_dV  = chk("dV",   V.grad,  V_ref.grad)
    ok_dW  = chk("dW",   W.grad,  W_ref.grad)

    return all([ok_out, ok_dQ, ok_dK, ok_dV, ok_dW])


def test_module_equivalence(B=2, N=197, d_h=768, NH=12, K=8, dtype=torch.float32):
    """Load the same weights into FlashMHA and FastMHA, confirm forward matches."""
    if not HAVE_FAST:
        print("\n[correctness:module] skipped (no FastMultiHeadConvNNAttention)")
        return True
    print(f"\n[correctness:module] B={B} N={N} d_h={d_h} NH={NH} K={K} dtype={dtype}")
    torch.manual_seed(0)

    flash = FlashMultiHeadConvNNAttention(
        d_hidden=d_h, num_heads=NH, attention_dropout=0.0,
        K=K, seq_length=N, convolution_type="depthwise",
    ).cuda().to(dtype)

    fast = FastMultiHeadConvNNAttention(
        d_hidden=d_h, num_heads=NH, attention_dropout=0.0,
        K=K, seq_length=N, convolution_type="depthwise",
    ).cuda().to(dtype)

    # Copy weights one-to-one
    fast.W_q.weight.data.copy_(flash.W_q.weight.data)
    fast.W_k.weight.data.copy_(flash.W_k.weight.data)
    fast.W_v.weight.data.copy_(flash.W_v.weight.data)
    fast.W_o.weight.data.copy_(flash.W_o.weight.data)
    fast.conv_weight.data.copy_(flash.conv_weight.data)

    x = torch.randn(B, N, d_h, device="cuda", dtype=dtype)

    flash.eval(); fast.eval()
    with torch.no_grad():
        y_flash = flash(x)
        y_fast = fast(x)

    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    diff = (y_flash - y_fast).abs().max().item()
    ok = torch.allclose(y_flash, y_fast, rtol=rtol, atol=atol)
    print(f"  flash vs fast forward: max_abs_diff={diff:.2e} {'OK' if ok else 'MISMATCH'}")
    return ok


# ----------------------- benchmark -----------------------------------

def bench(fn, n_warmup=20, n_iter=100):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1e3  # ms


def benchmark(B=8, N=197, d_h=768, NH=12, K=8, dtype=torch.float16):
    print(f"\n[bench] B={B} N={N} d_h={d_h} NH={NH} K={K} dtype={dtype}")
    torch.manual_seed(0)

    flash = FlashMultiHeadConvNNAttention(
        d_hidden=d_h, num_heads=NH, attention_dropout=0.0,
        K=K, seq_length=N,
    ).cuda().to(dtype)

    x = torch.randn(B, N, d_h, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(B, N, d_h, device="cuda", dtype=dtype)

    # forward
    def fwd_flash():
        with torch.no_grad():
            flash(x)
    t_fwd_flash = bench(fwd_flash)

    # forward + backward
    def fwd_bwd_flash():
        x.grad = None
        out = flash(x)
        out.backward(g)
    t_fb_flash = bench(fwd_bwd_flash)

    print(f"  flash  forward      : {t_fwd_flash:.3f} ms")
    print(f"  flash  fwd+bwd      : {t_fb_flash:.3f} ms")

    if HAVE_FAST:
        fast = FastMultiHeadConvNNAttention(
            d_hidden=d_h, num_heads=NH, attention_dropout=0.0,
            K=K, seq_length=N, convolution_type="depthwise",
        ).cuda().to(dtype)

        def fwd_fast():
            with torch.no_grad():
                fast(x)
        t_fwd_fast = bench(fwd_fast)

        def fwd_bwd_fast():
            x.grad = None
            out = fast(x)
            out.backward(g)
        t_fb_fast = bench(fwd_bwd_fast)

        print(f"  triton forward      : {t_fwd_fast:.3f} ms"
              f"  -> {t_fwd_fast / t_fwd_flash:.2f}x speedup for flash")
        print(f"  triton fwd+bwd      : {t_fb_fast:.3f} ms"
              f"  -> {t_fb_fast  / t_fb_flash :.2f}x speedup for flash")

    # Memory: peak allocated for one fwd+bwd
    torch.cuda.reset_peak_memory_stats()
    fwd_bwd_flash()
    mem_flash = torch.cuda.max_memory_allocated() / 1024**2
    print(f"  flash  peak memory  : {mem_flash:.1f} MiB")
    if HAVE_FAST:
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd_fast()
        mem_fast = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  triton peak memory  : {mem_fast:.1f} MiB"
              f"  -> {mem_fast / mem_flash:.2f}x reduction for flash")


# ----------------------- main ----------------------------------------

if __name__ == "__main__":
    assert torch.cuda.is_available(), "Need a CUDA GPU"
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Phase-level correctness
    all_ok = True
    for K in (4, 8, 16):
        for dtype in (torch.float32, torch.float16):
            ok = test_phase_correctness(BH=4, N=197, d_k=64, K=K, dtype=dtype)
            all_ok = all_ok and ok

    # Module-level equivalence with the existing Triton implementation
    test_module_equivalence(dtype=torch.float32)
    test_module_equivalence(dtype=torch.float16)

    # Microbenchmarks at ViT-Base settings
    benchmark(B=8,  N=197, K=8,  dtype=torch.float16)
    benchmark(B=32, N=197, K=8,  dtype=torch.float16)
    benchmark(B=8,  N=197, K=16, dtype=torch.float16)

    # Larger seq length to show where flash wins most
    benchmark(B=4,  N=1024, K=8, dtype=torch.float16)
    benchmark(B=4,  N=4096, K=8, dtype=torch.float16)

    print("\nDone.", "All correctness checks passed." if all_ok else "Some checks FAILED.")
