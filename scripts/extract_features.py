"""Extracts all feature families across train, validation, and test datasets."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np

from src.features.pipeline import EssayFeaturePipeline


def main():
    print("=" * 60)
    print("PHASE 1 FEATURE EXTRACTION PIPELINE")
    print("=" * 60)

    data_dir = os.path.join("data", "processed")
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")
    test_path = os.path.join(data_dir, "test.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing train dataset at {train_path}. Run scripts/prepare_data.py first.")

    train_df = pd.read_json(train_path, lines=True)
    val_df = pd.read_json(val_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)

    print(f"Loaded datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    pipeline = EssayFeaturePipeline()

    print("\n[1/4] Extracting base features on Train split to fit Human Distribution...")
    # First pass on Train to get base features
    train_base_rows = []
    base_cols = pipeline.get_base_feature_names()
    for _, row in train_df.iterrows():
        base_dict = pipeline.extract_base_features_single(row["text"])
        base_dict["binary_label"] = row["binary_label"]
        train_base_rows.append(base_dict)
    train_base_df = pd.DataFrame(train_base_rows)

    # Fit Human Distribution strictly on training human essays (leakage-free!)
    pipeline.fit_human_distribution(train_base_df)
    print("      [PASSED] Empirical Human Baseline fitted on Train human samples.")

    print("\n[2/4] Transforming Full Feature Matrices (Surface, Discourse, Distributional)...")
    train_features_df = pipeline.transform_dataset(train_df)
    val_features_df = pipeline.transform_dataset(val_df)
    test_features_df = pipeline.transform_dataset(test_df)

    all_feature_cols = pipeline.get_all_feature_names()
    print(f"      - Total Features Extracted per essay: {len(all_feature_cols)}")
    print(f"      - Surface Features: {len([c for c in all_feature_cols if c.startswith('surface_')])}")
    print(f"      - Discourse Features: {len([c for c in all_feature_cols if c.startswith('discourse_')])}")
    print(f"      - Distributional Features: {len([c for c in all_feature_cols if c.startswith('dist_')])}")

    # Verify integrity (no NaNs or Infs)
    for name, df in [("Train", train_features_df), ("Val", val_features_df), ("Test", test_features_df)]:
        nan_count = df[all_feature_cols].isna().sum().sum()
        inf_count = np.isinf(df[all_feature_cols].to_numpy()).sum()
        assert nan_count == 0, f"FATAL: Found {nan_count} NaNs in {name} feature matrix!"
        assert inf_count == 0, f"FATAL: Found {inf_count} Infs in {name} feature matrix!"
        print(f"      [PASSED] {name} Matrix Verification: Zero NaNs, Zero Infs.")

    # Save feature datasets
    train_feat_path = os.path.join(data_dir, "train_features.csv")
    val_feat_path = os.path.join(data_dir, "val_features.csv")
    test_feat_path = os.path.join(data_dir, "test_features.csv")
    meta_path = os.path.join(data_dir, "feature_metadata.json")

    train_features_df.to_csv(train_feat_path, index=False)
    val_features_df.to_csv(val_feat_path, index=False)
    test_features_df.to_csv(test_feat_path, index=False)

    metadata_list = [m.__dict__ for m in pipeline.get_all_metadata()]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"\n[3/4] Feature tables saved to {data_dir}/")
    print(f"[4/4] Feature metadata registry saved to {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
