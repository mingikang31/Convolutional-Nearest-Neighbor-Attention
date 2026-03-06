"""
Benchmark: Original PyTorch vs Tiled CUDA vs Triton Hybrid

Tests correctness (forward + backward) then benchmarks speed and memory.
"""
import torch
import torch.nn as nn
import numpy as np
import csv


# =============================================================================
# ORIGINAL PYTORCH IMPLEMENTATION
# =============================================================================
class OriginalPrimeConv(nn.Module):
    """Original _prime() + conv() as a standalone module for fair comparison."""
    def __init__(self, d_k, K):
        super().__init__()
        self.K = K
        self.d_k = d_k
        self.conv = nn.Conv1d(d_k, d_k, kernel_size=K, stride=K,
                              padding=0, groups=d_k, bias=False)
        self.conv.weight.data.fill_(1.0)

    def forward(self, v, attn):
        """
        v:    (B_NH, d_k, SL)
        attn: (B_NH, SL, SL)
        """
        b, c, t = v.shape
        K = self.K
        topk_values, topk_indices = torch.topk(attn, k=K, dim=2, largest=True)
        topk_values = torch.softmax(topk_values, dim=-1)
        topk_indices_exp = topk_indices.unsqueeze(1).expand(b, c, t, K)
        topk_values_exp = topk_values.unsqueeze(1).expand(b, c, t, K)
        v_expanded = v.unsqueeze(-1).expand(b, c, t, K).contiguous()
        prime = torch.gather(v_expanded, dim=2, index=topk_indices_exp)
        prime = topk_values_exp * prime
        prime = prime.view(b, c, -1)
        return self.conv(prime)


# =============================================================================
# TIMING UTILITY
# =============================================================================
def benchmark_fn(fn, warmup=20, repeats=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    timings = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))

    t = np.array(timings)
    return {'mean_ms': t.mean(), 'std_ms': t.std(), 'min_ms': t.min(), 'median_ms': np.median(t)}


def fmt(t):
    return f"{t['mean_ms']:8.3f} +/- {t['std_ms']:.3f}"


# =============================================================================
# CORRECTNESS CHECKS
# =============================================================================
def check_forward(original, module, v, attn, label):
    with torch.no_grad():
        out_ref = original(v, attn)
        out_test = module(v, attn)
    diff = (out_ref - out_test).abs().max().item()
    status = "PASS" if diff < 1e-3 else "FAIL"
    print(f"    {label} forward:  max_diff={diff:.2e} [{status}]")
    return diff < 1e-3


def check_backward(original, module, v, attn, label):
    """Compare gradients flowing through both modules."""
    # Clone inputs so each module gets fresh gradients
    v1 = v.clone().detach().requires_grad_(True)
    a1 = attn.clone().detach().requires_grad_(True)
    v2 = v.clone().detach().requires_grad_(True)
    a2 = attn.clone().detach().requires_grad_(True)

    out1 = original(v1, a1)
    out1.sum().backward()

    out2 = module(v2, a2)
    out2.sum().backward()

    results = {}
    for name, g1, g2 in [("grad_v", v1.grad, v2.grad),
                          ("grad_attn", a1.grad, a2.grad),
                          ("grad_conv_w",
                           original.conv.weight.grad.squeeze(1),
                           module.conv_weight.grad)]:
        diff = (g1 - g2).abs().max().item()
        results[name] = diff

    all_pass = all(d < 1e-2 for d in results.values())
    status = "PASS" if all_pass else "FAIL"

    details = ", ".join(f"{k}={v:.2e}" for k, v in results.items())
    print(f"    {label} backward: {details} [{status}]")
    return all_pass


def sync_conv_weights(original, module):
    """Copy conv weights from original Conv1d (d_k, 1, K) to fused (d_k, K)."""
    with torch.no_grad():
        module.conv_weight.data.copy_(original.conv.weight.data.squeeze(1))


# =============================================================================
# MAIN BENCHMARK
# =============================================================================
def run_all(device='cuda'):
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print()

    # Import fused modules
    from convnn_fused_tiled import FusedPrimeConvTiled
    from convnn_triton import FusedPrimeConvTriton

    # ==========================
    # CORRECTNESS
    # ==========================
    print("=" * 80)
    print("CORRECTNESS CHECKS")
    print("=" * 80)

    test_configs = [
        {'B_NH': 6, 'd_k': 64, 'SL': 50,  'K': 4,  'label': 'small'},
        {'B_NH': 48, 'd_k': 64, 'SL': 197, 'K': 8,  'label': 'ViT-B'},
        {'B_NH': 48, 'd_k': 64, 'SL': 197, 'K': 32, 'label': 'ViT-B K=32'},
        {'B_NH': 64, 'd_k': 64, 'SL': 197, 'K': 64, 'label': 'ViT-L K=64'},
    ]

    all_correct = True
    for cfg in test_configs:
        torch.manual_seed(42)
        B_NH, d_k, SL, K = cfg['B_NH'], cfg['d_k'], cfg['SL'], cfg['K']
        print(f"\n  Config: {cfg['label']} (B_NH={B_NH}, d_k={d_k}, SL={SL}, K={K})")

        v = torch.randn(B_NH, d_k, SL, device=device)
        attn = torch.randn(B_NH, SL, SL, device=device)

        original = OriginalPrimeConv(d_k, K).to(device)
        tiled = FusedPrimeConvTiled(d_k, K).to(device)
        triton_mod = FusedPrimeConvTriton(d_k, K).to(device)

        sync_conv_weights(original, tiled)
        sync_conv_weights(original, triton_mod)

        p1 = check_forward(original, tiled, v, attn, "Tiled CUDA")
        p2 = check_forward(original, triton_mod, v, attn, "Triton    ")

        # Reset grads for backward check
        original.zero_grad()
        tiled.zero_grad()
        triton_mod.zero_grad()

        p3 = check_backward(original, tiled, v, attn, "Tiled CUDA")
        p4 = check_backward(original, triton_mod, v, attn, "Triton    ")

        if not all([p1, p2, p3, p4]):
            all_correct = False

    print(f"\n  Overall correctness: {'ALL PASSED' if all_correct else 'SOME FAILED'}")

    # ==========================
    # SPEED BENCHMARKS
    # ==========================
    print("\n")
    print("=" * 80)
    print("SPEED BENCHMARKS")
    print("=" * 80)

    bench_configs = [
        {'B_NH': 12,  'd_k': 64, 'SL': 50,   'K': 8,  'label': 'Small'},
        {'B_NH': 48,  'd_k': 64, 'SL': 197,  'K': 8,  'label': 'ViT-B B=4'},
        {'B_NH': 192, 'd_k': 64, 'SL': 197,  'K': 8,  'label': 'ViT-B B=16'},
        {'B_NH': 768, 'd_k': 64, 'SL': 197,  'K': 8,  'label': 'ViT-B B=64'},
        {'B_NH': 192, 'd_k': 64, 'SL': 197,  'K': 4,  'label': 'ViT-B K=4'},
        {'B_NH': 192, 'd_k': 64, 'SL': 197,  'K': 16, 'label': 'ViT-B K=16'},
        {'B_NH': 192, 'd_k': 64, 'SL': 197,  'K': 32, 'label': 'ViT-B K=32'},
        {'B_NH': 192, 'd_k': 64, 'SL': 197,  'K': 64, 'label': 'ViT-B K=64'},
        {'B_NH': 64,  'd_k': 64, 'SL': 197,  'K': 8,  'label': 'ViT-L B=4'},
        {'B_NH': 48,  'd_k': 64, 'SL': 576,  'K': 8,  'label': 'ViT-B SL=576'},
        {'B_NH': 48,  'd_k': 64, 'SL': 1024, 'K': 8,  'label': 'ViT-B SL=1024'},
    ]

    # Header
    print(f"\n{'':>22s}  |  {'--- Forward (ms) ---':^60s}  |  {'--- Forward+Backward (ms) ---':^60s}")
    print(f"{'Config':>22s}  |  {'Original':>18s}  {'Tiled CUDA':>18s}  {'Triton':>18s}  "
          f"|  {'Original':>18s}  {'Tiled CUDA':>18s}  {'Triton':>18s}")
    print("-" * 170)

    csv_rows = []

    for cfg in bench_configs:
        torch.manual_seed(0)
        B_NH, d_k, SL, K = cfg['B_NH'], cfg['d_k'], cfg['SL'], cfg['K']

        v = torch.randn(B_NH, d_k, SL, device=device)
        attn = torch.randn(B_NH, SL, SL, device=device)

        original = OriginalPrimeConv(d_k, K).to(device)
        tiled = FusedPrimeConvTiled(d_k, K).to(device)
        triton_mod = FusedPrimeConvTriton(d_k, K).to(device)

        sync_conv_weights(original, tiled)
        sync_conv_weights(original, triton_mod)

        try:
            # Forward-only
            t_fwd_orig   = benchmark_fn(lambda: original(v, attn))
            t_fwd_tiled  = benchmark_fn(lambda: tiled(v, attn))
            t_fwd_triton = benchmark_fn(lambda: triton_mod(v, attn))

            # Forward + backward
            def fwd_bwd(mod):
                mod.zero_grad(set_to_none=True)
                out = mod(v.requires_grad_(True), attn.requires_grad_(True))
                out.sum().backward()

            t_bwd_orig   = benchmark_fn(lambda: fwd_bwd(original))
            t_bwd_tiled  = benchmark_fn(lambda: fwd_bwd(tiled))
            t_bwd_triton = benchmark_fn(lambda: fwd_bwd(triton_mod))

            print(f"{cfg['label']:>22s}  |  {fmt(t_fwd_orig):>18s}  {fmt(t_fwd_tiled):>18s}  "
                  f"{fmt(t_fwd_triton):>18s}  |  {fmt(t_bwd_orig):>18s}  {fmt(t_bwd_tiled):>18s}  "
                  f"{fmt(t_bwd_triton):>18s}")

            csv_rows.append({
                'config': cfg['label'], 'B_NH': B_NH, 'd_k': d_k, 'SL': SL, 'K': K,
                'fwd_orig': t_fwd_orig['mean_ms'], 'fwd_tiled': t_fwd_tiled['mean_ms'],
                'fwd_triton': t_fwd_triton['mean_ms'],
                'fwdbwd_orig': t_bwd_orig['mean_ms'], 'fwdbwd_tiled': t_bwd_tiled['mean_ms'],
                'fwdbwd_triton': t_bwd_triton['mean_ms'],
            })

        except RuntimeError as e:
            print(f"{cfg['label']:>22s}  |  SKIPPED: {e}")

        torch.cuda.empty_cache()

    # ==========================
    # SPEEDUP SUMMARY
    # ==========================
    print("\n")
    print("=" * 80)
    print("SPEEDUP SUMMARY (vs Original PyTorch)")
    print("=" * 80)
    print(f"\n{'Config':>22s}  |  {'Fwd Tiled':>10s}  {'Fwd Triton':>11s}  |  "
          f"{'F+B Tiled':>10s}  {'F+B Triton':>11s}")
    print("-" * 80)

    for r in csv_rows:
        fwd_t = r['fwd_orig'] / r['fwd_tiled'] if r['fwd_tiled'] > 0 else 0
        fwd_tr = r['fwd_orig'] / r['fwd_triton'] if r['fwd_triton'] > 0 else 0
        bwd_t = r['fwdbwd_orig'] / r['fwdbwd_tiled'] if r['fwdbwd_tiled'] > 0 else 0
        bwd_tr = r['fwdbwd_orig'] / r['fwdbwd_triton'] if r['fwdbwd_triton'] > 0 else 0
        print(f"{r['config']:>22s}  |  {fwd_t:>9.2f}x  {fwd_tr:>10.2f}x  |  "
              f"{bwd_t:>9.2f}x  {bwd_tr:>10.2f}x")

    # ==========================
    # MEMORY COMPARISON
    # ==========================
    print("\n")
    print("=" * 80)
    print("PEAK MEMORY COMPARISON (ViT-B, B_NH=192, K=8)")
    print("=" * 80)

    mem_cfg = {'B_NH': 192, 'd_k': 64, 'SL': 197, 'K': 8}
    modules = {
        'Original': OriginalPrimeConv(mem_cfg['d_k'], mem_cfg['K']),
        'Tiled CUDA': FusedPrimeConvTiled(mem_cfg['d_k'], mem_cfg['K']),
        'Triton': FusedPrimeConvTriton(mem_cfg['d_k'], mem_cfg['K']),
    }

    for name, mod in modules.items():
        mod = mod.to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        v = torch.randn(mem_cfg['B_NH'], mem_cfg['d_k'], mem_cfg['SL'],
                        device=device, requires_grad=True)
        attn = torch.randn(mem_cfg['B_NH'], mem_cfg['SL'], mem_cfg['SL'],
                           device=device, requires_grad=True)

        mod.zero_grad(set_to_none=True)
        out = mod(v, attn)
        out.sum().backward()
        torch.cuda.synchronize()

        peak = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"  {name:>12s}: {peak:8.1f} MB")

        del mod, v, attn, out
        torch.cuda.empty_cache()

    # ==========================
    # SAVE CSV
    # ==========================
    if csv_rows:
        with open('benchmark_results_all.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nResults saved to benchmark_results_all.csv")


if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA required"
    run_all('cuda')
