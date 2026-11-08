"""
Data drift monitor using Evidently AI 0.4.x
"""
import argparse
import json
import sys
import os

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "ee_x", "ee_y",
    "obj_x", "obj_y",
    "goal_x", "goal_y",
    "delta_x", "delta_y",
]

DRIFT_THRESHOLD = 0.5


def load_obs(path: str, split: str = "first_half") -> pd.DataFrame:
    d = np.load(path)
    obs = d["obs"]
    n = len(obs)
    if split == "first_half":
        obs = obs[: n // 2]
    else:
        obs = obs[n // 2 :]
    return pd.DataFrame(obs, columns=FEATURE_NAMES)


def run_drift_check(reference_path, production_path, output_path, threshold=DRIFT_THRESHOLD):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    reference_df = load_obs(reference_path, split="first_half")
    production_df = load_obs(production_path, split="second_half")

    print(f"Reference samples : {len(reference_df)}")
    print(f"Production samples: {len(production_df)}")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=production_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"\nDrift report saved → {output_path}")

    result_dict = report.as_dict()
    drift_result = result_dict["metrics"][0]["result"]

    num_drifted = drift_result.get("number_of_drifted_columns", 0)
    share = drift_result.get("share_of_drifted_columns", 0.0)
    drift_detected = share >= threshold

    print(f"\nDrift summary:")
    print(f"  Features drifted : {num_drifted}/{len(FEATURE_NAMES)}")
    print(f"  Share drifted    : {share:.1%}")
    print(f"  Drift detected   : {drift_detected}")
    print(f"  Threshold        : {threshold:.1%}")

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
        print("\n⚠️  DRIFT ALERT: distribution shift exceeds threshold.")
        sys.exit(1)
    else:
        print("\n✅  No significant drift detected.")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default="data/demos.npz")
    parser.add_argument("--production", default="data/demos.npz")
    parser.add_argument("--output", default="monitoring/drift/drift_report.html")
    parser.add_argument("--threshold", type=float, default=DRIFT_THRESHOLD)
    args = parser.parse_args()
    run_drift_check(args.reference, args.production, args.output, args.threshold)


if __name__ == "__main__":
    main()
