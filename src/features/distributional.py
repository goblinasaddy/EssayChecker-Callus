"""StoryScope-inspired distributional geometry and human subspace feature extractor."""
from typing import Dict, List, Optional
import numpy as np
from sklearn.covariance import EmpiricalCovariance

from src.features.base import BaseFeatureExtractor, FeatureMetadata
from src.segmentation.segmenter import HierarchicalSegmenter, EssaySegmentation


class DistributionalFeatureExtractor(BaseFeatureExtractor):
    """
    Computes distributional distance metrics relative to an empirical human training baseline.
    Treated as a research hypothesis to test whether structural manifold distances
    provide independent signal beyond standard linear/tree feature combinations.
    """

    def __init__(self, segmenter: Optional[HierarchicalSegmenter] = None):
        self.segmenter = segmenter or HierarchicalSegmenter()
        self.human_centroid: Optional[np.ndarray] = None
        self.human_cov_inv: Optional[np.ndarray] = None
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit_human_baseline(self, human_feature_matrix: np.ndarray, regularization: float = 1e-3):
        """
        Fits the human reference distribution strictly on training human essays.
        Uses Ledoit-Wolf / ridge-regularized covariance to guarantee invertibility.
        """
        if len(human_feature_matrix) < 5:
            # Fallback if baseline size is tiny
            self.human_centroid = np.mean(human_feature_matrix, axis=0)
            self.human_cov_inv = np.eye(human_feature_matrix.shape[1])
            self.feature_means = self.human_centroid
            self.feature_stds = np.ones(human_feature_matrix.shape[1])
            self.is_fitted = True
            return

        self.human_centroid = np.mean(human_feature_matrix, axis=0)
        self.feature_means = self.human_centroid
        self.feature_stds = np.std(human_feature_matrix, axis=0)
        self.feature_stds[self.feature_stds < 1e-6] = 1.0

        # Regularized Empirical Covariance on standardized features
        std_matrix = (human_feature_matrix - self.human_centroid) / self.feature_stds
        cov_estimator = EmpiricalCovariance().fit(std_matrix)
        cov = cov_estimator.covariance_ + regularization * np.eye(std_matrix.shape[1])
        self.human_cov_inv = np.linalg.pinv(cov)
        self.is_fitted = True

    def extract_distributional_metrics(self, base_feature_vector: np.ndarray, paragraph_feature_vectors: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
        """Computes distances relative to the fitted human manifold."""
        if not self.is_fitted or self.human_centroid is None:
            return {
                "dist_human_euclidean": 0.0,
                "dist_human_mahalanobis": 0.0,
                "dist_human_cosine": 1.0,
                "dist_structural_dispersion": 0.0,
            }

        # Standardize vector
        std_vec = (base_feature_vector - self.feature_means) / self.feature_stds
        
        # 1. Standardized Euclidean distance
        euclidean_dist = float(np.linalg.norm(std_vec))

        # 2. Mahalanobis distance
        mahal_sq = float(std_vec.T @ self.human_cov_inv @ std_vec)
        mahalanobis_dist = float(np.sqrt(max(0.0, mahal_sq)))

        # 3. Cosine similarity with human centroid
        norm_v = np.linalg.norm(base_feature_vector)
        norm_c = np.linalg.norm(self.human_centroid)
        if norm_v > 1e-6 and norm_c > 1e-6:
            cosine_sim = float(np.dot(base_feature_vector, self.human_centroid) / (norm_v * norm_c))
        else:
            cosine_sim = 1.0

        # 4. Structural Dispersion across essay paragraphs
        if paragraph_feature_vectors and len(paragraph_feature_vectors) > 1:
            p_mat = np.array(paragraph_feature_vectors)
            p_disp = float(np.mean(np.var(p_mat, axis=0)))
        else:
            p_disp = 0.0

        return {
            "dist_human_euclidean": euclidean_dist,
            "dist_human_mahalanobis": mahalanobis_dist,
            "dist_human_cosine": cosine_sim,
            "dist_structural_dispersion": p_disp,
        }

    def extract_features(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> Dict[str, float]:
        # Standalone call defaults to 0.0 if not called through pipeline with base vector
        return {m.name: 0.0 for m in self.get_metadata()}

    def get_metadata(self) -> List[FeatureMetadata]:
        return [
            FeatureMetadata("dist_human_euclidean", "distributional", "Standardized Euclidean distance from human training centroid", "[0.0, 50.0]", "Deviation from average human admissions profile", "Sensitive to dimension scaling"),
            FeatureMetadata("dist_human_mahalanobis", "distributional", "Mahalanobis distance from human empirical distribution", "[0.0, 50.0]", "Covariance-weighted distance from human subspace", "Requires invertible covariance estimation"),
            FeatureMetadata("dist_human_cosine", "distributional", "Cosine similarity to human centroid vector", "[-1.0, 1.0]", "Directional alignment with human feature archetype", "Ignores absolute magnitude"),
            FeatureMetadata("dist_structural_dispersion", "distributional", "Mean feature variance across essay paragraphs", "[0.0, 10.0]", "Structural modulation across essay body", "Requires multiple paragraphs"),
        ]
