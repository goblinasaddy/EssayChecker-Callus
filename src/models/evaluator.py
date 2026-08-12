"""Comprehensive and honest evaluation metrics engine for AI detection."""
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)


class ModelEvaluator:
    """
    Computes rigorous classification metrics, calibration measures, and error breakdowns.
    """

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Computes Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR, and Confusion Matrix.
        """
        y_true = np.array(y_true, dtype=int)
        y_pred = np.array(y_pred, dtype=int)

        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        fpr = float(fp / max(1, fp + tn))  # False Positive Rate (Human predicted as AI)
        fnr = float(fn / max(1, fn + tp))  # False Negative Rate (AI predicted as Human)

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "confusion_matrix": cm.tolist(),
        }

        if y_prob is not None:
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = 0.5
            brier = float(brier_score_loss(y_true, y_prob))
            metrics["roc_auc"] = auc
            metrics["brier_score"] = brier
        else:
            metrics["roc_auc"] = None
            metrics["brier_score"] = None

        return metrics
