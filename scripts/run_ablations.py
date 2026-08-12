"""Runs feature family ablation experiments."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from tabulate import tabulate

from src.evaluation.ablation import FeatureAblationRunner


def main():
    print("=" * 60)
    print("PHASE 1 FEATURE FAMILY ABLATION EXPERIMENTS")
    print("=" * 60)

    data_dir = os.path.join("data", "processed")
    train_df = pd.read_csv(os.path.join(data_dir, "train_features.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_features.csv"))

    runner = FeatureAblationRunner(random_state=42)
    ablation_df, _ = runner.run_ablations(train_df, test_df)

    out_csv = os.path.join(data_dir, "ablation_results.csv")
    ablation_df.to_csv(out_csv, index=False)

    print("\n[Ablation Results Summary - Held-Out Test Set]")
    print(tabulate(
        ablation_df[[
            "configuration", "feature_count", "lr_accuracy", "lr_f1", "lr_roc_auc", "lr_fpr",
            "rf_accuracy", "rf_f1", "rf_roc_auc", "rf_fpr"
        ]],
        headers=[
            "Config", "Feats", "LR Acc", "LR F1", "LR AUC", "LR FPR",
            "RF Acc", "RF F1", "RF AUC", "RF FPR"
        ],
        tablefmt="github",
        floatfmt=".4f"
    ))

    print(f"\nAblation table exported to {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
