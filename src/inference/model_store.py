"""Serialization and schema verification for production detector artifacts."""
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.pipeline import EssayFeaturePipeline


class ProductionModelArtifact:
    """
    Stores and validates all parameters, reference statistics, and schemas for inference.
    """

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DEFAULT_ARTIFACT_PATH = os.path.join(REPO_ROOT, "data", "models", "detector_artifact_v2.json")

    def __init__(
        self,
        model_version: str = "2.0.0",
        feature_names: Optional[List[str]] = None,
        coefficients: Optional[Dict[str, float]] = None,
        intercept: float = 0.0,
        scaler_mean: Optional[Dict[str, float]] = None,
        scaler_scale: Optional[Dict[str, float]] = None,
        human_reference_stats: Optional[Dict[str, Dict[str, float]]] = None,
        distributional_baseline: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model_version = model_version
        self.feature_names = feature_names or []
        self.coefficients = coefficients or {}
        self.intercept = intercept
        self.scaler_mean = scaler_mean or {}
        self.scaler_scale = scaler_scale or {}
        self.human_reference_stats = human_reference_stats or {}
        self.distributional_baseline = distributional_baseline or {}
        
        # Documented threshold boundaries based on training validation distributions:
        # P < 0.35 -> Likely Human
        # 0.35 <= P < 0.65 -> Uncertain / Borderline
        # 0.65 <= P < 0.85 -> Likely AI-Assisted / Polished
        # P >= 0.85 -> Likely AI-Generated
        self.thresholds = thresholds or {
            "human_max": 0.35,
            "uncertain_max": 0.65,
            "ai_assisted_max": 0.85,
        }

    @classmethod
    def train_and_export(
        cls,
        train_features_path: str = os.path.join("data", "processed", "train_features.csv"),
        output_path: str = DEFAULT_ARTIFACT_PATH,
        random_state: int = 42,
    ) -> "ProductionModelArtifact":
        """
        Trains the production calibrated Logistic Regression model strictly on the Train split,
        computes human reference statistics, and exports the serialized bundle.
        """
        train_df = pd.read_csv(train_features_path)
        feature_cols = [c for c in train_df.columns if c.startswith(("surface_", "discourse_", "dist_"))]

        y_train = train_df["binary_label"].to_numpy(dtype=int)
        X_train = train_df[feature_cols].to_numpy(dtype=float)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Train calibrated Logistic Regression
        lr = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        lr.fit(X_scaled, y_train)

        coef_map = {feat: float(coef) for feat, coef in zip(feature_cols, lr.coef_[0])}
        mean_map = {feat: float(m) for feat, m in zip(feature_cols, scaler.mean_)}
        scale_map = {feat: float(s) for feat, s in zip(feature_cols, scaler.scale_)}

        # Compute empirical human reference statistics strictly from human training samples
        human_train = train_df[train_df["binary_label"] == 0]
        human_ref_stats = {}
        for feat in feature_cols:
            vals = human_train[feat].to_numpy(dtype=float)
            human_ref_stats[feat] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if np.std(vals) > 1e-6 else 1.0,
                "median": float(np.median(vals)),
                "p10": float(np.percentile(vals, 10)),
                "p90": float(np.percentile(vals, 90)),
            }

        # Distributional human centroid & covariance
        pipeline = EssayFeaturePipeline()
        pipeline.fit_human_distribution(train_df)
        dist_ext = pipeline.dist_extractor
        
        dist_baseline = {
            "is_fitted": dist_ext.is_fitted,
            "human_centroid": dist_ext.human_centroid.tolist() if dist_ext.human_centroid is not None else [],
            "feature_means": dist_ext.feature_means.tolist() if dist_ext.feature_means is not None else [],
            "feature_stds": dist_ext.feature_stds.tolist() if dist_ext.feature_stds is not None else [],
            "human_cov_inv": dist_ext.human_cov_inv.tolist() if dist_ext.human_cov_inv is not None else [],
        }

        artifact = cls(
            model_version="2.0.0",
            feature_names=feature_cols,
            coefficients=coef_map,
            intercept=float(lr.intercept_[0]),
            scaler_mean=mean_map,
            scaler_scale=scale_map,
            human_reference_stats=human_ref_stats,
            distributional_baseline=dist_baseline,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        artifact.save(output_path)
        return artifact

    def save(self, file_path: str):
        payload = {
            "model_version": self.model_version,
            "feature_names": self.feature_names,
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "human_reference_stats": self.human_reference_stats,
            "distributional_baseline": self.distributional_baseline,
            "thresholds": self.thresholds,
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, file_path: str = DEFAULT_ARTIFACT_PATH) -> "ProductionModelArtifact":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model artifact not found at {file_path}. Train and export artifact first.")
        with open(file_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(
            model_version=payload["model_version"],
            feature_names=payload["feature_names"],
            coefficients=payload["coefficients"],
            intercept=payload["intercept"],
            scaler_mean=payload["scaler_mean"],
            scaler_scale=payload["scaler_scale"],
            human_reference_stats=payload["human_reference_stats"],
            distributional_baseline=payload["distributional_baseline"],
            thresholds=payload.get("thresholds"),
        )
