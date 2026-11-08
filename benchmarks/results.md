# Inference Benchmark Results

Device: Kaggle T4 GPU  
Model: BC Policy (obs_dim=8, action_dim=2)  

> Run `python benchmarks/inference_benchmark.py --checkpoint checkpoints/bc_policy.pt` to regenerate.

## Batch size = 1

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline_fp32 | — | — | *(run benchmark to populate)* |
| fp16 | — | — | |
| torch.compile | — | — | |
| onnx_runtime | — | — | |

## Batch size = 32

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline_fp32 | — | — | |
| fp16 | — | — | |
| torch.compile | — | — | |
| onnx_runtime | — | — | |

## Batch size = 256

| Mode | p50 latency (ms) | p95 latency (ms) | Throughput (samples/s) |
|---|---|---|---|
| baseline_fp32 | — | — | |
| fp16 | — | — | |
| torch.compile | — | — | |
| onnx_runtime | — | — | |
