"""Unit tests for baseline models, evaluation metrics, and ablation runners."""
import pytest
import numpy as np
import pandas as pd
from src.models.baselines import InterpretableBaselineModels
from src.models.evaluator import ModelEvaluator
from src.evaluation.ablation import FeatureAblationRunner


def test_baseline_models_fit_and_predict():
    np.random.seed(42)
    n_samples = 40
    n_feats = 5
    feature_names = [f"surface_feat_{i}" for i in range(n_feats)]
    
    X_mat = np.random.randn(n_samples, n_feats)
    # Simple linear decision boundary
    y = (X_mat[:, 0] + X_mat[:, 1] * 0.5 > 0).astype(int)

    df = pd.DataFrame(X_mat, columns=feature_names)
    df["binary_label"] = y

    model = InterpretableBaselineModels(random_state=42)
    model.fit(df, y, feature_names)

    assert model.is_fitted

    preds_lr, probs_lr = model.predict_lr(df)
    preds_rf, probs_rf = model.predict_rf(df)

    assert len(preds_lr) == n_samples
    assert len(probs_lr) == n_samples
    assert np.all((probs_lr >= 0.0) & (probs_lr <= 1.0))
    assert np.all((probs_rf >= 0.0) & (probs_rf <= 1.0))

    weights_df = model.get_logistic_regression_weights()
    assert len(weights_df) == n_feats
    assert "coefficient" in weights_df.columns
    assert "odds_ratio" in weights_df.columns

    rf_imp = model.get_random_forest_importance()
    assert len(rf_imp) == n_feats
    assert "gini_importance" in rf_imp.columns


def test_evaluator_metrics_computation():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])  # 1 FP
    y_prob = np.array([0.1, 0.7, 0.8, 0.9])

    metrics = ModelEvaluator.evaluate(y_true, y_pred, y_prob)

    assert metrics["accuracy"] == 0.75
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.0
    assert metrics["roc_auc"] is not None


def test_evaluator_edge_cases():
    # All zeros
    y_true = np.array([0, 0, 0])
    y_pred = np.array([0, 0, 0])
    metrics = ModelEvaluator.evaluate(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
