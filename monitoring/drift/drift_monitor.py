"""
Data drift monitor — compares a reference batch (training data) against a
production batch (new incoming observations) using Evidently AI.

This simulates what CERN would need: detecting when the sensor/detector data
distribution shifts, which would degrade model performance silently.

Usage:
    # Install: pip install evidently
    python monitoring/drift/drift_monitor.py \
        --reference data/demos.npz \
        --production data/demos.npz \
        --output monitoring/drift/drift_report.html

    Then open drift_report.html in your browser.

The script also exits with code 1 if drift is detected above threshold,
so it can be used as a CI/CD gate (see .github/workflows/ci.yml).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    from evidently.report import Report
    from evidently.metric_presets import DataDriftPreset
    from evidently.metrics import DatasetDriftMetric
except ImportError:
    print("ERROR: evidently not installed. Run: pip install evidently")
    sys.exit(1)

FEATURE_NAMES = [
    "ee_x", "ee_y",          # end-effector position
    "obj_x", "obj_y",         # object position
    "goal_x", "goal_y",       # goal position
    "delta_x", "delta_y",     # object-to-goal delta
]

DRIFT_THRESHOLD = 0.5   # fraction of features that may drift before alert


def load_obs(path: str, split: str = "first_half") -> pd.DataFrame:
    """Load observations from npz demo file, return as DataFrame."""
    d = np.load(path)
    obs = d["obs"]  # shape (N, 8)
    n = len(obs)
    if split == "first_half":
        obs = obs[: n // 2]
    elif split == "second_half":
        obs = obs[n // 2 :]
    return pd.DataFrame(obs, columns=FEATURE_NAMES)


def run_drift_check(reference_path: str, production_path: str, output_path: str,
                    threshold: float = DRIFT_THRESHOLD):
    reference_df = load_obs(reference_path, split="first_half")
    production_df = load_obs(production_path, split="second_half")

    print(f"Reference samples : {len(reference_df)}")
    print(f"Production samples: {len(production_df)}")

    report = Report(metrics=[
        DataDriftPreset(),
        DatasetDriftMetric(drift_share=threshold),
    ])
    report.run(reference_data=reference_df, current_data=production_df)
    report.save_html(output_path)
    print(f"\nDrift report saved → {output_path}")

    # Extract drift result for CI gate
    result = report.as_dict()
    dataset_drift = result["metrics"][1]["result"]
    num_drifted = dataset_drift.get("number_of_drifted_columns", 0)
    drift_detected = dataset_drift.get("dataset_drift", False)
    share = dataset_drift.get("share_of_drifted_columns", 0.0)

    print(f"\nDrift summary:")
    print(f"  Features drifted : {num_drifted}/{len(FEATURE_NAMES)}")
    print(f"  Share drifted    : {share:.1%}")
    print(f"  Drift detected   : {drift_detected}")
    print(f"  Threshold        : {threshold:.1%}")

    # Save JSON summary (useful for GitHub Actions step summary)
    summary = {
        "num_drifted": num_drifted,
        "total_features": len(FEATURE_NAMES),
        "share_drifted": share,
        "drift_detected": drift_detected,
        "threshold": threshold,
    }
    summary_path = output_path.replace(".html", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary JSON     : {summary_path}")

    if drift_detected:
        print("\n⚠️  DRIFT ALERT: distribution shift exceeds threshold. Retraining recommended.")
        sys.exit(1)   # non-zero exit → CI step fails → triggers alert
    else:
        print("\n✅  No significant drift detected.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Data drift monitor")
    parser.add_argument("--reference", default="data/demos.npz",
                        help="Path to reference (training) data (.npz)")
    parser.add_argument("--production", default="data/demos.npz",
                        help="Path to production batch data (.npz)")
    parser.add_argument("--output", default="monitoring/drift/drift_report.html",
                        help="Output path for the HTML report")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD,
                        help="Max share of features allowed to drift (0–1)")
    args = parser.parse_args()
    run_drift_check(args.reference, args.production, args.output, args.threshold)


if __name__ == "__main__":
    main()
