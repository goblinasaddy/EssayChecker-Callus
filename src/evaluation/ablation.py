"""Ablation experiment runner for feature families."""
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from src.models.baselines import InterpretableBaselineModels
from src.models.evaluator import ModelEvaluator


class FeatureAblationRunner:
    """
    Evaluates isolated and combined feature families:
    - Model A: Surface / Stylometric only
    - Model B: Discourse / Narrative only
    - Model C: Distributional / StoryScope only
    - Model D: Surface + Discourse
    - Model E: Full Suite (Surface + Discourse + Distributional)
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.evaluator = ModelEvaluator()

    def run_ablations(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, InterpretableBaselineModels]]:
        """
        Runs ablations across all 5 configurations on both Logistic Regression and Random Forest.
        """
        surface_cols = [c for c in train_df.columns if c.startswith("surface_")]
        discourse_cols = [c for c in train_df.columns if c.startswith("discourse_")]
        dist_cols = [c for c in train_df.columns if c.startswith("dist_")]

        configurations = {
            "Model A (Surface Only)": surface_cols,
            "Model B (Discourse Only)": discourse_cols,
            "Model C (Distributional Only)": dist_cols,
            "Model D (Surface + Discourse)": surface_cols + discourse_cols,
            "Model E (Full: Surface + Discourse + Dist)": surface_cols + discourse_cols + dist_cols,
        }

        results = []
        fitted_models = {}

        y_train = train_df["binary_label"].to_numpy(dtype=int)
        y_test = test_df["binary_label"].to_numpy(dtype=int)

        for name, cols in configurations.items():
            if not cols:
                continue

            model = InterpretableBaselineModels(random_state=self.random_state)
            model.fit(train_df, y_train, cols)
            fitted_models[name] = model

            # Logistic Regression Test Metrics
            lr_preds, lr_probs = model.predict_lr(test_df)
            lr_metrics = self.evaluator.evaluate(y_test, lr_preds, lr_probs)

            # Random Forest Test Metrics
            rf_preds, rf_probs = model.predict_rf(test_df)
            rf_metrics = self.evaluator.evaluate(y_test, rf_preds, rf_probs)

            results.append({
                "configuration": name,
                "feature_count": len(cols),
                "lr_accuracy": lr_metrics["accuracy"],
                "lr_f1": lr_metrics["f1"],
                "lr_roc_auc": lr_metrics["roc_auc"],
                "lr_fpr": lr_metrics["false_positive_rate"],
                "lr_fnr": lr_metrics["false_negative_rate"],
                "rf_accuracy": rf_metrics["accuracy"],
                "rf_f1": rf_metrics["f1"],
                "rf_roc_auc": rf_metrics["roc_auc"],
                "rf_fpr": rf_metrics["false_positive_rate"],
                "rf_fnr": rf_metrics["false_negative_rate"],
            })

        ablation_df = pd.DataFrame(results)
        return ablation_df, fitted_models
