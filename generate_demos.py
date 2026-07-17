"""Generate a demonstration dataset from the scripted expert, for imitation learning."""
import argparse
import json
import os

import numpy as np

from envs.push_env import PushEnv, PushEnvConfig, ScriptedExpert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=500)
    parser.add_argument("--noise_std", type=float, default=0.015, help="Action noise for demo diversity")
    parser.add_argument("--output", type=str, default="data/demos.npz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PushEnvConfig(seed=args.seed)
    env = PushEnv(cfg)
    expert = ScriptedExpert(cfg, noise_std=args.noise_std)

    all_obs, all_actions, all_ep_ids = [], [], []
    successes = 0

    for ep in range(args.num_episodes):
        env.seed(args.seed + ep)
        obs = env.reset()
        done = False
        while not done:
            action = expert.act(env)
            all_obs.append(obs)
            all_actions.append(action)
            all_ep_ids.append(ep)
            obs, _, done, info = env.step(action)
        successes += info["success"]
        # Only keep successful demonstrations — standard practice for BC data
        # collection; failed rollouts would teach the policy to imitate failure.
        if not info["success"]:
            keep_from = len(all_ep_ids) - 1
            while keep_from >= 0 and all_ep_ids[keep_from] == ep:
                all_obs.pop()
                all_actions.pop()
                all_ep_ids.pop()
                keep_from -= 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        obs=np.array(all_obs, dtype=np.float32),
        actions=np.array(all_actions, dtype=np.float32),
        episode_ids=np.array(all_ep_ids, dtype=np.int64),
    )

    stats = {
        "num_episodes_attempted": int(args.num_episodes),
        "num_episodes_kept": int(successes),
        "num_transitions": int(len(all_obs)),
        "expert_success_rate": float(successes / args.num_episodes),
    }
    print(json.dumps(stats, indent=2))
    with open(os.path.join(os.path.dirname(args.output), "demo_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
