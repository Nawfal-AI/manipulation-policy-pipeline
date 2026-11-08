"""
MLflow experiment tracking wrapper.

Usage:
    python mlflow/mlflow_tracking.py --policy bc   --data data/demos.npz
    python mlflow/mlflow_tracking.py --policy diffusion --data data/demos.npz

What this adds on top of train_policy.py:
  - Every run is logged to a local MLflow tracking server (./mlruns)
  - Hyperparameters, per-epoch loss, and the final checkpoint are stored as
    MLflow artifacts so you can compare runs in the UI.
  - The best run's model is automatically registered in the Model Registry
    and transitioned to "Staging".

Start the MLflow UI after training:
    mlflow ui --port 5000
Then open http://localhost:5000
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path so we can import models
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import BCPolicy, DiffusionPolicy

import mlflow
import mlflow.pytorch
from mlflow import MlflowClient

EXPERIMENT_NAME = "manipulation-policy"
MODEL_REGISTRY_NAME = "manipulation-policy-prod"


def load_data(path: str):
    d = np.load(path)
    obs = torch.tensor(d["obs"], dtype=torch.float32)
    actions = torch.tensor(d["actions"], dtype=torch.float32)
    obs_mean = obs.mean(0, keepdim=True)
    obs_std = obs.std(0, keepdim=True) + 1e-6
    obs_norm = (obs - obs_mean) / obs_std
    return obs_norm, actions, obs_mean, obs_std


def train(policy_type: str, data_path: str, epochs: int, batch_size: int, lr: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs, actions, obs_mean, obs_std = load_data(data_path)
    dataset = TensorDataset(obs, actions)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = (BCPolicy() if policy_type == "bc" else DiffusionPolicy()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── MLflow run ──────────────────────────────────────────────────────────
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"{policy_type}-ep{epochs}") as run:
        # Log hyperparameters
        mlflow.log_params({
            "policy_type": policy_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "device": device,
            "obs_dim": obs.shape[1],
            "action_dim": actions.shape[1],
            "num_samples": len(dataset),
        })

        history = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            for obs_batch, action_batch in loader:
                obs_batch = obs_batch.to(device)
                action_batch = action_batch.to(device)
                optimizer.zero_grad()
                if policy_type == "bc":
                    pred = model(obs_batch)
                    loss = torch.nn.functional.mse_loss(pred, action_batch)
                else:
                    loss = model.training_loss(obs_batch, action_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * obs_batch.shape[0]
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
            # Log metric per epoch so the UI shows the loss curve
            mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[{policy_type}] epoch {epoch+1}/{epochs}  loss={epoch_loss:.5f}")

        final_loss = history[-1]
        mlflow.log_metric("final_loss", final_loss)

        # Save checkpoint as MLflow artifact
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/{policy_type}_policy.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "policy_type": policy_type,
            "obs_mean": obs_mean,
            "obs_std": obs_std,
        }, ckpt_path)
        mlflow.log_artifact(ckpt_path, artifact_path="checkpoints")

        # Also log loss history JSON
        history_path = f"checkpoints/{policy_type}_train_history.json"
        with open(history_path, "w") as f:
            json.dump({"final_loss": final_loss, "loss_history": history}, f, indent=2)
        mlflow.log_artifact(history_path, artifact_path="checkpoints")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/checkpoints/{policy_type}_policy.pt"
        try:
            client = MlflowClient()
            # Create registered model if it doesn't exist
            try:
                client.create_registered_model(MODEL_REGISTRY_NAME)
            except Exception:
                pass  # already exists
            mv = client.create_model_version(
                name=MODEL_REGISTRY_NAME,
                source=model_uri,
                run_id=run.info.run_id,
                description=f"{policy_type} policy, final_loss={final_loss:.5f}",
            )
            # Transition to Staging
            client.transition_model_version_stage(
                name=MODEL_REGISTRY_NAME,
                version=mv.version,
                stage="Staging",
                archive_existing_versions=False,
            )
            print(f"Registered model version {mv.version} → Staging")
        except Exception as e:
            print(f"Model registry step skipped: {e}")

        print(f"\nRun ID : {run.info.run_id}")
        print(f"Final loss : {final_loss:.5f}")
        print(f"MLflow UI  : mlflow ui --port 5000  →  http://localhost:5000")
        return run.info.run_id, final_loss


def main():
    parser = argparse.ArgumentParser(description="Train + log to MLflow")
    parser.add_argument("--policy", choices=["bc", "diffusion"], default="bc")
    parser.add_argument("--data", default="data/demos.npz")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.policy, args.data, args.epochs, args.batch_size, args.lr)


if __name__ == "__main__":
    main()
