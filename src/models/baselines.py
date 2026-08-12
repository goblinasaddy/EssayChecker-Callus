"""Interpretable baseline classifiers for Phase 1 AI authorship analysis."""
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance


class InterpretableBaselineModels:
    """
    Trains and inspects primary interpretable machine learning baselines.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        self.lr_model = LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
        )
        
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: List[str]):
        """Fits both Logistic Regression and Random Forest on training data."""
        self.feature_names = list(feature_cols)
        X_mat = X_train[self.feature_names].to_numpy(dtype=float)
        
        # Scale features for Logistic Regression
        X_scaled = self.scaler.fit_transform(X_mat)
        
        self.lr_model.fit(X_scaled, y_train)
        self.rf_model.fit(X_mat, y_train)
        self.is_fitted = True

    def predict_lr(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (binary_predictions, ai_probabilities) for Logistic Regression."""
        X_mat = X[self.feature_names].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X_mat)
        preds = self.lr_model.predict(X_scaled)
        probs = self.lr_model.predict_proba(X_scaled)[:, 1]
        return preds, probs

    def predict_rf(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (binary_predictions, ai_probabilities) for Random Forest."""
        X_mat = X[self.feature_names].to_numpy(dtype=float)
        preds = self.rf_model.predict(X_mat)
        probs = self.rf_model.predict_proba(X_mat)[:, 1]
        return preds, probs

    def get_logistic_regression_weights(self) -> pd.DataFrame:
        """Extracts standardized weights and odds ratios for Logistic Regression."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        coefs = self.lr_model.coef_[0]
        odds_ratios = np.exp(coefs)
        
        df = pd.DataFrame({
            "feature": self.feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
            "odds_ratio": odds_ratios,
            "direction": ["Higher -> AI" if c > 0 else "Higher -> Human" for c in coefs],
        })
        return df.sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    def get_random_forest_importance(self, X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Extracts MDI and permutation importances for Random Forest."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet.")

        gini_imp = self.rf_model.feature_importances_
        df = pd.DataFrame({
            "feature": self.feature_names,
            "gini_importance": gini_imp,
        })

        if X_val is not None and y_val is not None:
            X_mat = X_val[self.feature_names].to_numpy(dtype=float)
            perm = permutation_importance(self.rf_model, X_mat, y_val, n_repeats=10, random_state=self.random_state)
            df["permutation_importance_mean"] = perm.importances_mean
            df["permutation_importance_std"] = perm.importances_std

        return df.sort_values(by="gini_importance", ascending=False).reset_index(drop=True)
