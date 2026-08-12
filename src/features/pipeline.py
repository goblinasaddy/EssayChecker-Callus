"""Unified feature extraction pipeline combining Surface, Discourse, and Distributional extractors."""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from src.features.base import BaseFeatureExtractor, FeatureMetadata
from src.features.surface import SurfaceFeatureExtractor
from src.features.discourse import DiscourseFeatureExtractor
from src.features.distributional import DistributionalFeatureExtractor
from src.segmentation.segmenter import HierarchicalSegmenter, EssaySegmentation


class EssayFeaturePipeline:
    """
    Orchestrates multi-family feature extraction across text samples.
    """

    def __init__(self):
        self.segmenter = HierarchicalSegmenter()
        self.surface_extractor = SurfaceFeatureExtractor(self.segmenter)
        self.discourse_extractor = DiscourseFeatureExtractor(self.segmenter)
        self.dist_extractor = DistributionalFeatureExtractor(self.segmenter)

    def extract_base_features_single(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> Dict[str, float]:
        """Extracts surface and discourse features for a single essay."""
        if segmentation is None:
            segmentation = self.segmenter.segment(text)

        feats = {}
        feats.update(self.surface_extractor.extract_features(text, segmentation))
        feats.update(self.discourse_extractor.extract_features(text, segmentation))
        return feats

    def fit_human_distribution(self, train_df: pd.DataFrame):
        """Fits the distributional human baseline strictly on training human rows."""
        human_train = train_df[train_df["binary_label"] == 0]
        if len(human_train) == 0:
            return

        base_feat_cols = self.get_base_feature_names()
        human_mat = human_train[base_feat_cols].to_numpy(dtype=float)
        self.dist_extractor.fit_human_baseline(human_mat)

    def transform_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a DataFrame containing 'text' into a complete feature DataFrame.
        """
        rows = []
        base_feature_names = self.get_base_feature_names()

        for idx, row in df.iterrows():
            text = row["text"]
            seg = self.segmenter.segment(text)
            
            # Base features
            base_dict = self.extract_base_features_single(text, seg)
            
            # Paragraph feature vectors for structural dispersion
            p_vectors = []
            for p in seg.paragraphs:
                p_dict = self.extract_base_features_single(p.text)
                p_vec = np.array([p_dict.get(k, 0.0) for k in base_feature_names], dtype=float)
                p_vectors.append(p_vec)

            # Distributional metrics if fitted
            base_vec = np.array([base_dict.get(k, 0.0) for k in base_feature_names], dtype=float)
            dist_dict = self.dist_extractor.extract_distributional_metrics(base_vec, p_vectors)
            
            combined = {}
            # Retain sample metadata & raw text
            for col in ["essay_id", "group_id", "label", "binary_label", "topic_category", "model_family", "word_count", "esl_metadata", "text"]:
                if col in row:
                    combined[col] = row[col]

            combined.update(base_dict)
            combined.update(dist_dict)
            rows.append(combined)

        feature_df = pd.DataFrame(rows)
        return feature_df

    def get_base_feature_names(self) -> List[str]:
        names = [m.name for m in self.surface_extractor.get_metadata()]
        names.extend([m.name for m in self.discourse_extractor.get_metadata()])
        return names

    def get_all_feature_names(self) -> List[str]:
        names = self.get_base_feature_names()
        names.extend([m.name for m in self.dist_extractor.get_metadata()])
        return names

    def get_all_metadata(self) -> List[FeatureMetadata]:
        all_meta = []
        all_meta.extend(self.surface_extractor.get_metadata())
        all_meta.extend(self.discourse_extractor.get_metadata())
        all_meta.extend(self.dist_extractor.get_metadata())
        return all_meta
