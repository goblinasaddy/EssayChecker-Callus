"""Robustness, subgroup slicing, and ESL sensitivity analysis."""
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from src.models.baselines import InterpretableBaselineModels
from src.models.evaluator import ModelEvaluator


class RobustnessAnalyzer:
    """
    Evaluates detector performance across subgroups:
    - Essay length buckets
    - Topic categories
    - AI generation models
    - Synthetic AI-polished essays vs Pure AI vs Natural Human
    - ESL / Non-native English sensitivity audit
    """

    def __init__(self, evaluator: ModelEvaluator = None):
        self.evaluator = evaluator or ModelEvaluator()

    def evaluate_length_robustness(
        self, model: InterpretableBaselineModels, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates metrics across essay length slices."""
        df = test_df.copy()
        
        bins = [0, 350, 550, 10000]
        labels = ["Short (<350w)", "Medium (350-550w)", "Long (>550w)"]
        df["length_bucket"] = pd.cut(df["word_count"], bins=bins, labels=labels, right=False)

        results = []
        for bucket in labels:
            sub = df[df["length_bucket"] == bucket]
            if len(sub) == 0:
                continue

            y_true = sub["binary_label"].to_numpy(dtype=int)
            lr_preds, lr_probs = model.predict_lr(sub)
            rf_preds, rf_probs = model.predict_rf(sub)

            lr_m = self.evaluator.evaluate(y_true, lr_preds, lr_probs)
            rf_m = self.evaluator.evaluate(y_true, rf_preds, rf_probs)

            results.append({
                "length_slice": bucket,
                "sample_count": len(sub),
                "human_count": int(sum(y_true == 0)),
                "ai_count": int(sum(y_true == 1)),
                "lr_accuracy": lr_m["accuracy"],
                "lr_f1": lr_m["f1"],
                "lr_fpr": lr_m["false_positive_rate"],
                "lr_fnr": lr_m["false_negative_rate"],
                "rf_accuracy": rf_m["accuracy"],
                "rf_f1": rf_m["f1"],
                "rf_fpr": rf_m["false_positive_rate"],
                "rf_fnr": rf_m["false_negative_rate"],
            })

        return pd.DataFrame(results)

    def evaluate_topic_robustness(
        self, model: InterpretableBaselineModels, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates metrics across essay topic domains."""
        results = []
        topics = test_df["topic_category"].unique()

        for topic in topics:
            sub = test_df[test_df["topic_category"] == topic]
            if len(sub) == 0:
                continue

            y_true = sub["binary_label"].to_numpy(dtype=int)
            lr_preds, lr_probs = model.predict_lr(sub)
            rf_preds, rf_probs = model.predict_rf(sub)

            lr_m = self.evaluator.evaluate(y_true, lr_preds, lr_probs)
            rf_m = self.evaluator.evaluate(y_true, rf_preds, rf_probs)

            results.append({
                "topic": topic,
                "sample_count": len(sub),
                "human_count": int(sum(y_true == 0)),
                "ai_count": int(sum(y_true == 1)),
                "lr_accuracy": lr_m["accuracy"],
                "lr_f1": lr_m["f1"],
                "lr_fpr": lr_m["false_positive_rate"],
                "lr_fnr": lr_m["false_negative_rate"],
                "rf_accuracy": rf_m["accuracy"],
                "rf_f1": rf_m["f1"],
                "rf_fpr": rf_m["false_positive_rate"],
                "rf_fnr": rf_m["false_negative_rate"],
            })

        if not results:
            return pd.DataFrame(columns=["topic", "sample_count", "human_count", "ai_count", "lr_accuracy", "lr_f1", "lr_fpr", "lr_fnr", "rf_accuracy", "rf_f1", "rf_fpr", "rf_fnr"])
        return pd.DataFrame(results)

    def evaluate_ai_model_robustness(
        self, model: InterpretableBaselineModels, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates detection recall across individual AI generation models."""
        ai_sub = test_df[test_df["binary_label"] == 1].copy()
        results = []

        for m_family in ai_sub["model_family"].dropna().unique():
            sub = ai_sub[ai_sub["model_family"] == m_family]
            if len(sub) == 0:
                continue

            y_true = np.ones(len(sub), dtype=int)
            lr_preds, lr_probs = model.predict_lr(sub)
            rf_preds, rf_probs = model.predict_rf(sub)

            lr_recall = float(np.mean(lr_preds == 1))
            rf_recall = float(np.mean(rf_preds == 1))
            lr_mean_prob = float(np.mean(lr_probs))
            rf_mean_prob = float(np.mean(rf_probs))

            results.append({
                "model_family": m_family,
                "sample_count": len(sub),
                "lr_detection_recall": lr_recall,
                "lr_mean_ai_probability": lr_mean_prob,
                "rf_detection_recall": rf_recall,
                "rf_mean_ai_probability": rf_mean_prob,
            })

        return pd.DataFrame(results)

    def evaluate_polished_vs_pure_ai(
        self, model: InterpretableBaselineModels, test_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates performance on synthetic AI-polished text vs pure AI vs natural human."""
        results = []
        labels = test_df["label"].unique()

        for lbl in labels:
            sub = test_df[test_df["label"] == lbl]
            if len(sub) == 0:
                continue

            lr_preds, lr_probs = model.predict_lr(sub)
            rf_preds, rf_probs = model.predict_rf(sub)

            results.append({
                "label_category": lbl,
                "sample_count": len(sub),
                "lr_predicted_as_ai_ratio": float(np.mean(lr_preds == 1)),
                "lr_mean_ai_confidence": float(np.mean(lr_probs)),
                "rf_predicted_as_ai_ratio": float(np.mean(rf_preds == 1)),
                "rf_mean_ai_confidence": float(np.mean(rf_probs)),
            })

        return pd.DataFrame(results)

    def evaluate_esl_sensitivity(
        self, model: InterpretableBaselineModels, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Audits false positive rates on verified ESL/non-native subsets if ethical metadata exists.
        Strict rule: Never infers ESL status from text alone.
        """
        if "esl_metadata" not in test_df.columns:
            return {
                "status": "NOT_EVALUATED",
                "reason": "Dataset does not contain verified ESL/non-native-English demographic metadata.",
            }

        human_sub = test_df[test_df["binary_label"] == 0]
        esl_human = human_sub[human_sub["esl_metadata"] == "verified_esl"]
        native_human = human_sub[human_sub["esl_metadata"] == "verified_native"]

        if len(esl_human) == 0 or len(native_human) == 0:
            return {
                "status": "INSUFFICIENT_METADATA",
                "verified_esl_count": len(esl_human),
                "verified_native_count": len(native_human),
                "reason": "Insufficient verified ESL admissions samples for statistically valid disparity estimation.",
            }

        lr_preds_esl, _ = model.predict_lr(esl_human)
        lr_preds_native, _ = model.predict_lr(native_human)

        fpr_esl = float(np.mean(lr_preds_esl == 1))
        fpr_native = float(np.mean(lr_preds_native == 1))

        return {
            "status": "EVALUATED",
            "verified_esl_count": len(esl_human),
            "verified_native_count": len(native_human),
            "lr_fpr_esl": fpr_esl,
            "lr_fpr_native": fpr_native,
            "fpr_disparity_ratio": fpr_esl / max(1e-5, fpr_native),
        }
