"""Benchmark EML-Koopman across CPU, GPU, and GPU-compiled backends.

Measures training throughput, inference latency, and peak memory for:
  - cpu_taylor:    CPU with Horner-form Taylor series
  - gpu_taylor:    GPU with Horner-form Taylor series
  - gpu_native:    GPU with torch.exp/log (native CUDA math)
  - gpu_compiled:  GPU with torch.compile-fused Triton kernel

Outputs JSON results to results/benchmarks/backend_comparison.json
"""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from koopman_eml.koopman_model import KoopmanEML
from koopman_eml.training import train_koopman_eml

ROOT = Path(__file__).resolve().parents[1]


def generate_lorenz_data(
    n_traj: int = 20, n_steps: int = 5000, dt: float = 0.01, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.integrate import solve_ivp

    def lorenz(t, s, sigma=10.0, rho=28.0, beta=8 / 3):
        x, y, z = s
        return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

    rng = np.random.default_rng(seed)
    all_states = []
    for _ in range(n_traj):
        ic = rng.standard_normal(3) + np.array([1.0, 1.0, 25.0])
        sol = solve_ivp(lorenz, (0, n_steps * dt), ic,
                        t_eval=np.linspace(0, n_steps * dt, n_steps + 1),
                        rtol=1e-10, atol=1e-12)
        all_states.append(sol.y.T)

    states = np.concatenate(all_states, axis=0)
    mean, std = states.mean(0), states.std(0)
    std[std < 1e-8] = 1.0
    normed = (states - mean) / std
    return normed[:-1], normed[1:]


def make_model(device: str, backend: str, n_obs: int = 16, depth: int = 2) -> KoopmanEML:
    model = KoopmanEML(state_dim=3, n_observables=n_obs, tree_depth=depth)
    model.backend = backend
    return model.to(device)


def benchmark_training(
    X_k: np.ndarray, X_k1: np.ndarray,
    device: str, backend: str,
    n_epochs: int = 300, batch_size: int = 2048, n_runs: int = 3,
) -> dict:
    """Time the full training loop."""
    results = []
    for run in range(n_runs):
        torch.manual_seed(42 + run)
        model = make_model(device, backend)

        Xk = torch.tensor(X_k[:20000], dtype=torch.float32)
        Xk1 = torch.tensor(X_k1[:20000], dtype=torch.float32)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        history = train_koopman_eml(
            model, Xk, Xk1,
            n_epochs=n_epochs, lr=3e-3, batch_size=batch_size,
            device=device, verbose=False,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        peak_mem_mb = 0.0
        if device == "cuda":
            peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        n_pairs = len(Xk)
        pairs_per_sec = (n_pairs * n_epochs) / elapsed
        per_epoch = elapsed / n_epochs

        results.append({
            "run": run,
            "train_time_s": round(elapsed, 3),
            "per_epoch_ms": round(per_epoch * 1000, 2),
            "pairs_per_sec": round(pairs_per_sec),
            "peak_mem_mb": round(peak_mem_mb, 1),
            "final_loss": round(history["total"][-1], 6),
        })

        del model, history
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    return {
        "train_time_s": round(np.mean([r["train_time_s"] for r in results]), 3),
        "train_time_std": round(np.std([r["train_time_s"] for r in results]), 3),
        "per_epoch_ms": round(np.mean([r["per_epoch_ms"] for r in results]), 2),
        "pairs_per_sec": round(np.mean([r["pairs_per_sec"] for r in results])),
        "peak_mem_mb": round(np.mean([r["peak_mem_mb"] for r in results]), 1),
        "final_loss": round(np.mean([r["final_loss"] for r in results]), 6),
        "runs": results,
    }


def benchmark_inference(
    X_k: np.ndarray,
    device: str, backend: str,
    batch_sizes: list[int] = [1, 64, 256, 1024, 4096],
    warmup: int = 100, timed: int = 1000,
) -> dict:
    """Measure inference latency and throughput at various batch sizes."""
    torch.manual_seed(42)
    model = make_model(device, backend)
    model.eval()

    # Pre-train minimally so the model has sensible weights
    Xk = torch.tensor(X_k[:4096], dtype=torch.float32)
    Xk1_dummy = torch.tensor(X_k[1:4097], dtype=torch.float32)
    train_koopman_eml(model, Xk, Xk1_dummy, n_epochs=50, device=device, verbose=False)

    results = {}
    for B in batch_sizes:
        x = torch.randn(B, 3, dtype=torch.float32, device=device)

        with torch.no_grad():
            for _ in range(warmup):
                _ = model.lift(x, tau=0.01)
                _ = model.predict(_)
                _ = model.reconstruct(_)

        if device == "cuda":
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(timed):
                if device == "cuda":
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    g = model.lift(x, tau=0.01)
                    g_pred = model.predict(g)
                    x_recon = model.reconstruct(g_pred)
                    end.record()
                    torch.cuda.synchronize()
                    latencies.append(start.elapsed_time(end))
                else:
                    t0 = time.perf_counter()
                    g = model.lift(x, tau=0.01)
                    g_pred = model.predict(g)
                    x_recon = model.reconstruct(g_pred)
                    latencies.append((time.perf_counter() - t0) * 1000)

        lat_arr = np.array(latencies)
        results[f"B={B}"] = {
            "latency_ms_mean": round(float(lat_arr.mean()), 4),
            "latency_ms_std": round(float(lat_arr.std()), 4),
            "latency_ms_p50": round(float(np.percentile(lat_arr, 50)), 4),
            "latency_ms_p99": round(float(np.percentile(lat_arr, 99)), 4),
            "throughput_batches_per_sec": round(1000.0 / lat_arr.mean(), 1),
        }

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return results


def benchmark_rollout(
    X_k: np.ndarray,
    device: str, backend: str,
    horizon: int = 500, n_runs: int = 10,
) -> dict:
    """Time a multi-step prediction rollout."""
    torch.manual_seed(42)
    model = make_model(device, backend)

    Xk = torch.tensor(X_k[:4096], dtype=torch.float32)
    Xk1_dummy = torch.tensor(X_k[1:4097], dtype=torch.float32)
    train_koopman_eml(model, Xk, Xk1_dummy, n_epochs=50, device=device, verbose=False)
    model.eval()

    x0 = torch.tensor(X_k[0:1], dtype=torch.float32, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            g = model.lift(x0, tau=0.01)
            for _ in range(horizon):
                g = model.predict(g)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            g = model.lift(x0, tau=0.01)
            for step in range(horizon):
                g = model.predict(g)
            x_final = model.reconstruct(g)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "horizon": horizon,
        "rollout_ms_mean": round(np.mean(times) * 1000, 3),
        "rollout_ms_std": round(np.std(times) * 1000, 3),
    }


def main():
    print("=" * 70)
    print("EML-Koopman Backend Benchmark")
    print("=" * 70)

    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "N/A"

    meta = {
        "pytorch_version": torch.__version__,
        "cuda_available": has_cuda,
        "gpu_name": gpu_name,
        "driver_version": "",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if has_cuda:
        meta["driver_version"] = str(torch.version.cuda)
        meta["gpu_memory_mb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e6, 0
        )

    print(f"PyTorch: {meta['pytorch_version']}")
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {meta.get('driver_version', 'N/A')}")
    print()

    configs = [
        ("cpu_taylor",    "cpu",  "taylor"),
        ("gpu_taylor",    "cuda", "taylor"),
        ("gpu_native",    "cuda", "native"),
        ("gpu_compiled",  "cuda", "compiled"),
    ]

    if not has_cuda:
        configs = [c for c in configs if c[1] == "cpu"]
        print("WARNING: No CUDA device, running CPU-only benchmarks.\n")

    print("Generating Lorenz data...")
    X_k, X_k1 = generate_lorenz_data()
    print(f"  {X_k.shape[0]} training pairs, {X_k.shape[1]}D state\n")

    all_results = {"metadata": meta, "benchmarks": {}}

    for name, device, backend in configs:
        print(f"\n{'-' * 60}")
        print(f"  {name}  (device={device}, backend={backend})")
        print(f"{'-' * 60}")

        # --- Training ---
        print("  Training (300 epochs, 3 runs)...")
        try:
            train_res = benchmark_training(
                X_k, X_k1, device, backend, n_epochs=300, n_runs=3,
            )
            print(f"    Time: {train_res['train_time_s']:.1f} ± {train_res['train_time_std']:.1f}s")
            print(f"    Throughput: {train_res['pairs_per_sec']:,} pairs/s")
            print(f"    Peak mem: {train_res['peak_mem_mb']:.0f} MB")
        except Exception as e:
            print(f"    ERROR: {e}")
            train_res = {"error": str(e)}

        # --- Inference ---
        print("  Inference latency (1000 timed passes)...")
        try:
            infer_res = benchmark_inference(X_k, device, backend)
            for bs_key, bs_data in infer_res.items():
                print(f"    {bs_key}: {bs_data['latency_ms_mean']:.3f} ms "
                      f"({bs_data['throughput_batches_per_sec']:.0f} batch/s)")
        except Exception as e:
            print(f"    ERROR: {e}")
            infer_res = {"error": str(e)}

        # --- Rollout ---
        print("  Rollout (500 steps, 10 runs)...")
        try:
            rollout_res = benchmark_rollout(X_k, device, backend)
            print(f"    Rollout: {rollout_res['rollout_ms_mean']:.2f} ± "
                  f"{rollout_res['rollout_ms_std']:.2f} ms")
        except Exception as e:
            print(f"    ERROR: {e}")
            rollout_res = {"error": str(e)}

        all_results["benchmarks"][name] = {
            "device": device,
            "backend": backend,
            "training": train_res,
            "inference": infer_res,
            "rollout": rollout_res,
        }

    # --- Summary table ---
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<18} {'Train(s)':<12} {'Pairs/s':<12} "
          f"{'Infer B=1(ms)':<16} {'Infer B=4096(ms)':<18} {'Mem(MB)':<10}")
    print("-" * 86)
    for name, data in all_results["benchmarks"].items():
        t = data["training"]
        i = data["inference"]
        train_s = f"{t['train_time_s']:.1f}" if "train_time_s" in t else "ERR"
        pairs = f"{t['pairs_per_sec']:,}" if "pairs_per_sec" in t else "ERR"
        mem = f"{t['peak_mem_mb']:.0f}" if "peak_mem_mb" in t else "ERR"
        lat1 = f"{i.get('B=1', {}).get('latency_ms_mean', 'ERR')}"
        lat4096 = f"{i.get('B=4096', {}).get('latency_ms_mean', 'ERR')}"
        print(f"{name:<18} {train_s:<12} {pairs:<12} {lat1:<16} {lat4096:<18} {mem:<10}")

    # Save results
    out_dir = ROOT / "results" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backend_comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
