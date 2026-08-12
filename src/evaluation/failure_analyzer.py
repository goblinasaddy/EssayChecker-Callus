"""Failure and error analysis for admissions essay AI detection."""
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.models.baselines import InterpretableBaselineModels


class FailureAnalyzer:
    """
    Identifies and analyzes candidate failure cases (False Positives and False Negatives).
    Diagnoses which specific feature anomalies drove the erroneous prediction.
    """

    def __init__(self, baseline_model: InterpretableBaselineModels, train_df: pd.DataFrame):
        self.model = baseline_model
        self.feature_names = baseline_model.feature_names
        
        # Calculate human training feature distributions for anomaly diagnosis (mean and std)
        human_train = train_df[train_df["binary_label"] == 0]
        self.human_means = human_train[self.feature_names].mean()
        self.human_stds = human_train[self.feature_names].std().replace(0, 1.0)

    def find_failure_cases(self, test_df: pd.DataFrame, top_k: int = 5) -> Dict[str, List[Dict]]:
        """
        Identifies top False Positives (Human misclassified as AI with highest AI probability)
        and top False Negatives (AI misclassified as Human with lowest AI probability).
        """
        preds, probs = self.model.predict_lr(test_df)
        
        df = test_df.copy()
        df["pred"] = preds
        df["ai_prob"] = probs

        # 1. False Positives (Ground Truth = Human (0), Prediction = AI (1))
        fps = df[(df["binary_label"] == 0) & (df["pred"] == 1)].sort_values(by="ai_prob", ascending=False)
        
        # 2. False Negatives (Ground Truth = AI/Polished (1), Prediction = Human (0))
        fns = df[(df["binary_label"] == 1) & (df["pred"] == 0)].sort_values(by="ai_prob", ascending=True)

        fp_cases = []
        for _, row in fps.head(top_k).iterrows():
            anomalies = self._diagnose_sample_anomalies(row)
            fp_cases.append({
                "essay_id": row.get("essay_id", "unknown"),
                "ground_truth": "Human",
                "predicted": "AI",
                "ai_probability": float(row["ai_prob"]),
                "topic": row.get("topic_category", "unspecified"),
                "word_count": int(row.get("word_count", 0)),
                "text_snippet": row["text"][:300] + "...",
                "top_ai_skewed_features": anomalies,
            })

        fn_cases = []
        for _, row in fns.head(top_k).iterrows():
            anomalies = self._diagnose_sample_anomalies(row)
            fn_cases.append({
                "essay_id": row.get("essay_id", "unknown"),
                "ground_truth": f"AI ({row.get('model_family', 'unknown')})",
                "predicted": "Human",
                "ai_probability": float(row["ai_prob"]),
                "topic": row.get("topic_category", "unspecified"),
                "word_count": int(row.get("word_count", 0)),
                "text_snippet": row["text"][:300] + "...",
                "top_human_skewed_features": anomalies,
            })

        return {
            "false_positives": fp_cases,
            "false_negatives": fn_cases,
        }

    def _diagnose_sample_anomalies(self, sample_row: pd.Series, top_n: int = 4) -> List[Dict]:
        """Calculates Z-scores of sample features relative to human training distribution."""
        z_scores = []
        weights_df = self.model.get_logistic_regression_weights().set_index("feature")

        for feat in self.feature_names:
            val = float(sample_row[feat])
            mu = float(self.human_means[feat])
            sigma = float(self.human_stds[feat])
            z = (val - mu) / sigma
            
            coef = float(weights_df.loc[feat, "coefficient"]) if feat in weights_df.index else 0.0
            # Impact towards AI prediction: z * coef
            impact = z * coef
            z_scores.append({
                "feature": feat,
                "value": round(val, 4),
                "human_mean": round(mu, 4),
                "z_score": round(z, 2),
                "model_impact": round(impact, 3),
            })

        # Sort by highest model impact
        z_scores.sort(key=lambda x: abs(x["model_impact"]), reverse=True)
        return z_scores[:top_n]
