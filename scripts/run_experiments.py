"""Trains and evaluates primary interpretable baseline models."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import pandas as pd
import numpy as np

from src.models.baselines import InterpretableBaselineModels
from src.models.evaluator import ModelEvaluator


def main():
    print("=" * 60)
    print("PHASE 1 BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 60)

    data_dir = os.path.join("data", "processed")
    train_feat_path = os.path.join(data_dir, "train_features.csv")
    val_feat_path = os.path.join(data_dir, "val_features.csv")
    test_feat_path = os.path.join(data_dir, "test_features.csv")

    train_df = pd.read_csv(train_feat_path)
    val_df = pd.read_csv(val_feat_path)
    test_df = pd.read_csv(test_feat_path)

    # All numerical feature columns
    feature_cols = [c for c in train_df.columns if c.startswith(("surface_", "discourse_", "dist_"))]
    print(f"Total candidate features for baseline modeling: {len(feature_cols)}")

    y_train = train_df["binary_label"].to_numpy(dtype=int)
    y_val = val_df["binary_label"].to_numpy(dtype=int)
    y_test = test_df["binary_label"].to_numpy(dtype=int)

    # Train baselines
    model = InterpretableBaselineModels(random_state=42)
    model.fit(train_df, y_train, feature_cols)
    print("[1/4] Fitted Logistic Regression and Random Forest on Train split.")

    evaluator = ModelEvaluator()

    # Validation Evaluation
    lr_val_preds, lr_val_probs = model.predict_lr(val_df)
    rf_val_preds, rf_val_probs = model.predict_rf(val_df)
    lr_val_metrics = evaluator.evaluate(y_val, lr_val_preds, lr_val_probs)
    rf_val_metrics = evaluator.evaluate(y_val, rf_val_preds, rf_val_probs)

    # Test Evaluation (Strictly held-out)
    lr_test_preds, lr_test_probs = model.predict_lr(test_df)
    rf_test_preds, rf_test_probs = model.predict_rf(test_df)
    lr_test_metrics = evaluator.evaluate(y_test, lr_test_preds, lr_test_probs)
    rf_test_metrics = evaluator.evaluate(y_test, rf_test_preds, rf_test_probs)

    print("\n[2/4] Model Evaluation Summary (Held-Out Test Set):")
    print(f"      --- Logistic Regression ---")
    print(f"      Accuracy:  {lr_test_metrics['accuracy']:.4f}")
    print(f"      Precision: {lr_test_metrics['precision']:.4f}")
    print(f"      Recall:    {lr_test_metrics['recall']:.4f}")
    print(f"      F1 Score:  {lr_test_metrics['f1']:.4f}")
    print(f"      ROC-AUC:   {lr_test_metrics['roc_auc']:.4f}")
    print(f"      FPR:       {lr_test_metrics['false_positive_rate']:.4f}")
    print(f"      FNR:       {lr_test_metrics['false_negative_rate']:.4f}")

    print(f"\n      --- Random Forest ---")
    print(f"      Accuracy:  {rf_test_metrics['accuracy']:.4f}")
    print(f"      Precision: {rf_test_metrics['precision']:.4f}")
    print(f"      Recall:    {rf_test_metrics['recall']:.4f}")
    print(f"      F1 Score:  {rf_test_metrics['f1']:.4f}")
    print(f"      ROC-AUC:   {rf_test_metrics['roc_auc']:.4f}")
    print(f"      FPR:       {rf_test_metrics['false_positive_rate']:.4f}")
    print(f"      FNR:       {rf_test_metrics['false_negative_rate']:.4f}")

    # Feature Importance Inspection
    lr_weights_df = model.get_logistic_regression_weights()
    rf_imp_df = model.get_random_forest_importance(val_df, y_val)

    print("\n[3/4] Top 10 Features Driving Predictions (Logistic Regression):")
    for idx, row in lr_weights_df.head(10).iterrows():
        print(f"      {idx+1:2d}. {row['feature']:<38} (Coef: {row['coefficient']:+.4f} | {row['direction']})")

    # Save results
    results_payload = {
        "validation_metrics": {
            "logistic_regression": lr_val_metrics,
            "random_forest": rf_val_metrics,
        },
        "test_metrics": {
            "logistic_regression": lr_test_metrics,
            "random_forest": rf_test_metrics,
        },
    }

    out_metrics_path = os.path.join(data_dir, "baseline_experiment_results.json")
    out_lr_weights_path = os.path.join(data_dir, "logistic_regression_weights.csv")
    out_rf_imp_path = os.path.join(data_dir, "random_forest_importance.csv")

    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    lr_weights_df.to_csv(out_lr_weights_path, index=False)
    rf_imp_df.to_csv(out_rf_imp_path, index=False)

    print(f"\n[4/4] Saved results to {data_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
