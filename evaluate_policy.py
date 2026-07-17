"""
Evaluate a trained policy: task success rate, inference latency, and a
robustness stress test (observation noise), mirroring the JD's own KPI list
(task success rate, robustness, latency).
"""
import argparse
import json
import time

import numpy as np
import torch

from envs.push_env import PushEnv, PushEnvConfig
from models import BCPolicy, DiffusionPolicy


def load_policy(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = BCPolicy() if ckpt["policy_type"] == "bc" else DiffusionPolicy()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt["obs_mean"].to(device), ckpt["obs_std"].to(device)


def rollout(env, model, obs_mean, obs_std, device, obs_noise_std: float = 0.0):
    obs = env.reset()
    done = False
    latencies = []
    while not done:
        if obs_noise_std > 0:
            obs = obs + np.random.normal(0, obs_noise_std, size=obs.shape).astype(np.float32)
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        obs_t = (obs_t - obs_mean) / obs_std
        start = time.perf_counter()
        action = model.act(obs_t).cpu().numpy()[0]
        latencies.append(time.perf_counter() - start)
        obs, _, done, info = env.step(action)
    return info["success"], latencies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--output", type=str, default="results/eval_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, obs_mean, obs_std = load_policy(args.checkpoint, device)
    cfg = PushEnvConfig()
    env = PushEnv(cfg)

    results = {}
    for label, noise in [("clean", 0.0), ("noisy_obs_low", 0.02), ("noisy_obs_high", 0.06)]:
        successes, all_latencies = 0, []
        for ep in range(args.num_episodes):
            env.seed(10_000 + ep)
            success, latencies = rollout(env, model, obs_mean, obs_std, device, obs_noise_std=noise)
            successes += success
            all_latencies.extend(latencies)
        all_latencies = np.array(all_latencies) * 1000  # ms
        results[label] = {
            "success_rate": successes / args.num_episodes,
            "latency_p50_ms": float(np.percentile(all_latencies, 50)),
            "latency_p95_ms": float(np.percentile(all_latencies, 95)),
        }
        print(f"[{label}] success_rate={results[label]['success_rate']:.2%}  "
              f"p50={results[label]['latency_p50_ms']:.2f}ms  p95={results[label]['latency_p95_ms']:.2f}ms")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
