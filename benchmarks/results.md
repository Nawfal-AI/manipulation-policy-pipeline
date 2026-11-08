# Inference Benchmark Results

**Device:** Kaggle Tesla T4 GPU  
**Model:** BC Policy (obs\_dim=8, action\_dim=2, hidden\_dim=128)  
**Run date:** 2026-11-08  

> Reproduce: `python benchmarks/inference_benchmark.py --checkpoint checkpoints/bc_policy.pt`  
> CPU results (MacBook M-series) also shown for comparison.

---

## GPU Results (Tesla T4)

### Batch size = 1

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline\_fp32 | 0.134 | 0.167 | 7,111 |
| fp16 | 0.129 | 0.160 | 7,290 |
| torch.compile | 0.167 | 0.235 | 5,724 |
| **onnx\_runtime** | **0.012** | **0.013** | **76,953** |

### Batch size = 32

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline\_fp32 | 0.141 | 0.184 | 212,509 |
| fp16 | 0.151 | 0.312 | 191,139 |
| torch.compile | 0.186 | 0.226 | 167,322 |
| **onnx\_runtime** | **0.028** | **0.040** | **1,060,290** |

### Batch size = 256

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline\_fp32 | 0.150 | 0.184 | 1,662,028 |
| fp16 | 0.144 | 0.215 | 1,641,356 |
| torch.compile | 0.190 | 0.236 | 1,298,363 |
| **onnx\_runtime** | **0.119** | **0.212** | **1,882,885** |

---

## CPU Results (MacBook M-series, for comparison)

### Batch size = 1

| Mode | p50 latency (ms) | Throughput (samples/s) |
|---|---|---|
| baseline\_fp32 | 0.013 | 76,499 |
| fp16 | — (not supported on CPU) | — |
| torch.compile | 0.022 | 44,857 |
| **onnx\_runtime** | **0.004** | **225,406** |

### Batch size = 256

| Mode | p50 latency (ms) | Throughput (samples/s) |
|---|---|---|
| baseline\_fp32 | 0.055 | 4,544,380 |
| fp16 | — (not supported on CPU) | — |
| torch.compile | 0.117 | 2,162,801 |
| **onnx\_runtime** | **0.036** | **6,989,254** |

---

## Key findings

- **ONNX Runtime is 11x faster than baseline at batch size 1 on GPU** (0.012ms vs 0.134ms)
- **ONNX Runtime is 5x faster than baseline at batch size 32 on GPU** (0.028ms vs 0.141ms)
- **FP16 gives a small latency improvement** (~4%) over fp32 at batch size 1 — GPU memory bandwidth is not the bottleneck for this small model
- **torch.compile is slower than baseline** for this model size — compile overhead outweighs benefit for tiny MLPs; it benefits larger models more
- **ONNX Runtime on CPU outperforms GPU baseline** for batch size 1 (0.004ms vs 0.134ms) — GPU kernel launch overhead dominates at small batch sizes
