"""Train a Behavior Cloning or Diffusion policy on the generated demonstration dataset."""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from models import BCPolicy, DiffusionPolicy


def load_data(path: str):
    d = np.load(path)
    obs = torch.tensor(d["obs"], dtype=torch.float32)
    actions = torch.tensor(d["actions"], dtype=torch.float32)
    # Normalize observations (helps both BC and diffusion training stability)
    obs_mean, obs_std = obs.mean(0, keepdim=True), obs.std(0, keepdim=True) + 1e-6
    obs_norm = (obs - obs_mean) / obs_std
    return obs_norm, actions, obs_mean, obs_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["bc", "diffusion"], default="bc")
    parser.add_argument("--data", type=str, default="data/demos.npz")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, actions, obs_mean, obs_std = load_data(args.data)
    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = (BCPolicy() if args.policy == "bc" else DiffusionPolicy()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for obs_batch, action_batch in loader:
            obs_batch, action_batch = obs_batch.to(device), action_batch.to(device)
            optimizer.zero_grad()
            if args.policy == "bc":
                pred = model(obs_batch)
                loss = torch.nn.functional.mse_loss(pred, action_batch)
            else:
                loss = model.training_loss(obs_batch, action_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * obs_batch.shape[0]
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[{args.policy}] epoch {epoch+1}/{args.epochs}  loss={epoch_loss:.5f}")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"{args.policy}_policy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "policy_type": args.policy,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        },
        ckpt_path,
    )
    with open(os.path.join(args.output_dir, f"{args.policy}_train_history.json"), "w") as f:
        json.dump({"final_loss": history[-1], "loss_history": history}, f, indent=2)

    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
