"""
Model Registry helper — promote the best Staging model to Production.

Usage:
    # List all registered versions
    python mlflow/model_registry.py list

    # Promote the latest Staging version to Production
    python mlflow/model_registry.py promote

    # Compare all Staging runs by final_loss and promote the best one
    python mlflow/model_registry.py best

Start the MLflow UI first:
    mlflow ui --port 5000
"""

import argparse
import sys

from mlflow import MlflowClient

MODEL_NAME = "manipulation-policy-prod"


def list_versions(client: MlflowClient):
    print(f"\nRegistered versions for '{MODEL_NAME}':")
    print(f"{'Version':<10} {'Stage':<15} {'Run ID':<36} {'Description'}")
    print("-" * 90)
    for mv in client.search_model_versions(f"name='{MODEL_NAME}'"):
        print(f"{mv.version:<10} {mv.current_stage:<15} {mv.run_id:<36} {mv.description or ''}")


def promote_latest_staging(client: MlflowClient):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging = [v for v in versions if v.current_stage == "Staging"]
    if not staging:
        print("No Staging versions found. Run mlflow_tracking.py first.")
        sys.exit(1)
    latest = max(staging, key=lambda v: int(v.version))
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=latest.version,
        stage="Production",
        archive_existing_versions=True,  # archive old Production versions
    )
    print(f"Promoted version {latest.version} → Production  (run_id={latest.run_id})")


def promote_best(client: MlflowClient):
    """Find the Staging version with the lowest final_loss and promote it."""
    from mlflow.tracking import MlflowClient as _C
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging = [v for v in versions if v.current_stage == "Staging"]
    if not staging:
        print("No Staging versions found.")
        sys.exit(1)

    best_version = None
    best_loss = float("inf")
    for mv in staging:
        try:
            run = client.get_run(mv.run_id)
            loss = run.data.metrics.get("final_loss", float("inf"))
            print(f"  version={mv.version}  final_loss={loss:.5f}")
            if loss < best_loss:
                best_loss = loss
                best_version = mv
        except Exception as e:
            print(f"  version={mv.version}  could not fetch metrics: {e}")

    if best_version:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=best_version.version,
            stage="Production",
            archive_existing_versions=True,
        )
        print(f"\nBest version {best_version.version} (loss={best_loss:.5f}) → Production")
    else:
        print("Could not determine best version.")


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Registry helper")
    parser.add_argument("command", choices=["list", "promote", "best"])
    args = parser.parse_args()
    client = MlflowClient()
    if args.command == "list":
        list_versions(client)
    elif args.command == "promote":
        promote_latest_staging(client)
    elif args.command == "best":
        promote_best(client)


if __name__ == "__main__":
    main()
