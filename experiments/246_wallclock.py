"""Experiment 246 — WALL-CLOCK: does the FFN FLOP saving translate to real latency?
Times the FFN sublayer (fp16, real model dims) on GPU: dense vs kept-neuron (gathered smaller GEMM)
+ the rank-r correction path. Honest about (a) GEMM efficiency at smaller sizes and (b) the per-token
gather/scatter overhead that erodes savings under dynamic routing.
Run on an IDLE GPU. python3 experiments/246_wallclock.py
"""
from __future__ import annotations
import os
import torch

DIMS = {"Qwen-1.5B": (1536, 8960), "Qwen-7B": (3584, 18944)}   # (d, m) SwiGLU
N = int(os.environ.get("NTOK", "4096"))                         # tokens in a batch
RFEAT = 512; H = 512; R = 128                                   # correction: PCA dim, MLP hidden, rank
ITERS = 50; WARM = 15
dev = "cuda"
dt = torch.float16


def silu(z):
    return z * torch.sigmoid(z)


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    ts = []
    for _ in range(ITERS):
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]                                     # median ms


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"  N={N} tokens, fp16, GPU={torch.cuda.get_device_name(0)}\n", flush=True)
    print(f"  {'model':12} {'keep':5} {'dense':>8} {'kept':>8} {'corr':>7} {'kept+corr':>10} "
          f"{'speedup':>8} {'FFN%':>6} {'gather':>8}", flush=True)
    for name, (d, m) in DIMS.items():
        x = torch.randn(N, d, device=dev, dtype=dt)
        Wg = torch.randn(d, m, device=dev, dtype=dt) / (d ** 0.5)
        Wu = torch.randn(d, m, device=dev, dtype=dt) / (d ** 0.5)
        Wd = torch.randn(m, d, device=dev, dtype=dt) / (m ** 0.5)

        def dense():
            a = silu(x @ Wg) * (x @ Wu)
            return a @ Wd
        dense_ms = timed(dense)

        # correction path: z=(x@P)[N,rfeat] -> MLP(rfeat->H->R) -> @ Br.T [R,d]
        P = torch.randn(d, RFEAT, device=dev, dtype=dt) / (d ** 0.5)
        W1 = torch.randn(RFEAT, H, device=dev, dtype=dt) / (RFEAT ** 0.5)
        W2 = torch.randn(H, R, device=dev, dtype=dt) / (H ** 0.5)
        Br = torch.randn(d, R, device=dev, dtype=dt) / (d ** 0.5)

        def corr():
            z = x @ P
            h = torch.nn.functional.gelu(z @ W1) @ W2
            return h @ Br.T
        corr_ms = timed(corr)

        for f in (0.50, 0.25):
            mk = int(round(f * m))
            idx = torch.randperm(m, device=dev)[:mk]
            Wgk = Wg[:, idx].contiguous(); Wuk = Wu[:, idx].contiguous(); Wdk = Wd[idx, :].contiguous()

            def kept():                                        # STATIC selection: precomputed gathered weights
                a = silu(x @ Wgk) * (x @ Wuk)
                return a @ Wdk
            kept_ms = timed(kept)

            # per-token dynamic gather overhead (honest): gather mk weight-rows per token is infeasible;
            # we time a representative scatter of the kept-mask over the full activation (what masking costs).
            full_a = silu(x @ Wg) * (x @ Wu)
            mask = torch.zeros(N, m, device=dev, dtype=dt); mask[:, idx] = 1.0

            def dyn_mask():                                    # compute full FFN then mask (no FLOP saving)
                a = silu(x @ Wg) * (x @ Wu)
                return (a * mask) @ Wd
            gather_ms = timed(dyn_mask)

            tot = kept_ms + corr_ms
            ffn_pct = 100.0 * (3 * f * d * m + (d * RFEAT + RFEAT * H + H * R + R * d)) / (3 * d * m)
            print(f"  {name:12} {int(f*100):4}% {dense_ms:7.3f}m {kept_ms:7.3f}m {corr_ms:6.3f}m "
                  f"{tot:9.3f}m {dense_ms/tot:7.2f}x {ffn_pct:5.1f}% {gather_ms:7.3f}m", flush=True)
        del x, Wg, Wu, Wd, P, W1, W2, Br; torch.cuda.empty_cache()
    print("\nREAD: 'kept' = static-selection smaller GEMM (real width reduction). 'speedup'=dense/(kept+corr).", flush=True)
    print("'gather'=full-FFN-then-mask (per-token dynamic, NO FLOP saving) — shows masking gives no latency win;", flush=True)
    print("real speedup needs STATIC/shared selection (batchable gather). Honest deployment caveat.", flush=True)


if __name__ == "__main__":
    main()
