"""
GPU Inference Optimization Benchmark
=====================================
Compares three inference modes for the BC policy:
  1. Baseline    — standard PyTorch float32
  2. FP16        — half-precision (float16)
  3. torch.compile — compiled graph (PyTorch 2.x)
  4. ONNX Runtime — exported and run outside PyTorch

Run this on Kaggle (free T4 GPU) or locally on CPU (slower but works):
    python benchmarks/inference_benchmark.py --checkpoint checkpoints/bc_policy.pt

Results are saved to benchmarks/results.md and benchmarks/results.json
so they can be committed to GitHub and shown in the README.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import BCPolicy

BATCH_SIZES = [1, 32, 256]
WARMUP_RUNS = 20
TIMED_RUNS = 200


def load_model(ckpt_path: str, device: str) -> BCPolicy:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BCPolicy()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def benchmark_pytorch(model, batch, device, label="baseline"):
    """Benchmark standard PyTorch inference."""
    batch = batch.to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(batch)
    if device == "cuda":
        torch.cuda.synchronize()
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(TIMED_RUNS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(batch)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)
    return {
        "mode": label,
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_per_sec": int(batch.shape[0] / (np.mean(latencies) / 1000)),
    }


def benchmark_fp16(model, batch, device):
    """FP16 half-precision inference."""
    if device == "cpu":
        return {"mode": "fp16", "note": "FP16 not supported on CPU — skipped"}
    model_fp16 = model.half()
    batch_fp16 = batch.to(device).half()
    result = benchmark_pytorch(model_fp16, batch_fp16, device, label="fp16")
    model.float()  # restore
    return result


def benchmark_compiled(model, batch, device):
    """torch.compile — requires PyTorch 2.0+."""
    try:
        compiled = torch.compile(model, mode="reduce-overhead")
        result = benchmark_pytorch(compiled, batch, device, label="torch.compile")
        return result
    except Exception as e:
        return {"mode": "torch.compile", "note": f"skipped: {e}"}


def benchmark_onnx(model, batch, device, export_path="benchmarks/policy.onnx"):
    """Export to ONNX and benchmark with ONNX Runtime."""
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        return {"mode": "onnx", "note": "onnx/onnxruntime not installed — pip install onnx onnxruntime"}

    # Export — model and dummy must both be on CPU for ONNX export
    model_cpu = model.cpu()
    dummy = torch.randn(1, 8)  # CPU tensor
    torch.onnx.export(
        model_cpu, dummy, export_path,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch_size"}, "action": {0: "batch_size"}},
        opset_version=17,
    )
    model.to(device)  # move back to GPU after export
    print(f"  ONNX model exported → {export_path}  ({os.path.getsize(export_path)/1024:.1f} KB)")

    # Benchmark
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(export_path, providers=providers)
    input_name = sess.get_inputs()[0].name
    batch_np = batch.numpy()

    latencies = []
    for _ in range(WARMUP_RUNS):
        sess.run(None, {input_name: batch_np})
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        sess.run(None, {input_name: batch_np})
        latencies.append((time.perf_counter() - t0) * 1000)

    return {
        "mode": "onnx_runtime",
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "throughput_per_sec": int(batch.shape[0] / (np.mean(latencies) / 1000)),
        "model_size_kb": os.path.getsize(export_path) // 1024,
    }


def save_results(all_results: dict, output_dir: str = "benchmarks"):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # Markdown table (for README)
    md_path = os.path.join(output_dir, "results.md")
    with open(md_path, "w") as f:
        f.write("# Inference Benchmark Results\n\n")
        f.write(f"Device: {all_results['device']}  \n")
        f.write(f"Model: BC Policy (obs_dim=8, action_dim=2)  \n\n")
        for batch_size, runs in all_results["benchmarks"].items():
            f.write(f"## Batch size = {batch_size}\n\n")
            f.write("| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |\n")
            f.write("|---|---|---|---|\n")
            for r in runs:
                if "note" in r:
                    f.write(f"| {r['mode']} | — | — | {r.get('note','')} |\n")
                else:
                    f.write(f"| {r['mode']} | {r['p50_ms']:.3f} | {r['p95_ms']:.3f} | {r['throughput_per_sec']:,} |\n")
            f.write("\n")
    print(f"Markdown table → {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Inference optimization benchmark")
    parser.add_argument("--checkpoint", default="checkpoints/bc_policy.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    model = load_model(args.checkpoint, device)

    all_results = {"device": device, "benchmarks": {}}

    for batch_size in BATCH_SIZES:
        print(f"\n── Batch size {batch_size} ──────────────────────")
        batch = torch.randn(batch_size, 8)
        runs = []

        print("  [1/4] Baseline float32...")
        runs.append(benchmark_pytorch(model, batch.clone(), device, label="baseline_fp32"))

        print("  [2/4] FP16...")
        runs.append(benchmark_fp16(model, batch.clone(), device))

        print("  [3/4] torch.compile...")
        runs.append(benchmark_compiled(model, batch.clone(), device))

        print("  [4/4] ONNX Runtime...")
        runs.append(benchmark_onnx(model, batch.clone().cpu(), device))

        all_results["benchmarks"][str(batch_size)] = runs
        for r in runs:
            if "p50_ms" in r:
                print(f"    {r['mode']:<20} p50={r['p50_ms']:.3f}ms  throughput={r['throughput_per_sec']:,}/s")
            else:
                print(f"    {r['mode']:<20} {r.get('note','')}")

    save_results(all_results)
    print("\nDone. Commit benchmarks/results.md and benchmarks/results.json to GitHub.")


if __name__ == "__main__":
    main()
