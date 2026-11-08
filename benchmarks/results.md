# Inference Benchmark Results

Device: Apple M-series CPU (MacBook Pro)
Model: BC Policy (obs_dim=8, action_dim=2)

> GPU results (FP16) to be added after Kaggle T4 run.
> Run `python benchmarks/inference_benchmark.py` to regenerate.

## Batch size = 1

| Mode | p50 latency (ms) | Throughput (samples/s) |
|---|---|---|
| baseline_fp32 | 0.013 | 76,499 |
| fp16 | — (CPU not supported) | — |
| torch.compile | 0.022 | 44,857 |
| **onnx_runtime** | **0.004** | **225,406** |

## Batch size = 32

| Mode | p50 latency (ms) | Throughput (samples/s) |
|---|---|---|
| baseline_fp32 | 0.022 | 1,449,878 |
| fp16 | — (CPU not supported) | — |
| torch.compile | 0.103 | 305,996 |
| **onnx_runtime** | **0.010** | **3,134,502** |

## Batch size = 256

| Mode | p50 latency (ms) | Throughput (samples/s) |
|---|---|---|
| baseline_fp32 | 0.055 | 4,544,380 |
| fp16 | — (CPU not supported) | — |
| torch.compile | 0.117 | 2,162,801 |
| **onnx_runtime** | **0.036** | **6,989,254** |

**Key finding: ONNX Runtime is 3x faster than baseline at batch size 1
and 1.5x faster at batch size 256 on CPU.**
