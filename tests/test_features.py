"""Unit tests for surface, discourse, and distributional feature extractors."""
import pytest
import numpy as np
from src.features.surface import SurfaceFeatureExtractor
from src.features.discourse import DiscourseFeatureExtractor
from src.features.distributional import DistributionalFeatureExtractor
from src.features.pipeline import EssayFeaturePipeline


SAMPLE_ESSAY = """Every Saturday morning, our kitchen transformed into a dim sum workshop.
Flour dusted the floor tiles like a soft blanket of snow, while the rhythmic tempo of my grandmother's cleaver established the day's cadence.
My responsibility was clear: master the delicate art of folding siu mai dumplings.
At seven years old, my inexperienced hands frequently tore the thin dough wrappers, scattering seasoned pork across the surface.
Yet Nai Nai never reprimanded me; instead, she gently guided my fingers with enduring patience.

Through this cherished tradition, I absorbed the profound cultural essence of my heritage.
In our household, love was not expressed through overt verbal declarations, but through steaming bamboo baskets and comforting bowls of winter melon soup.
When our family relocated to Ohio, this culinary connection became my stabilizing foundation, allowing me to embrace my cultural identity with confidence.

As time progressed, this kitchen experience inspired me to establish the Cultural Heritage Exchange in high school.
Guiding my peers through Nai Nai's dumpling-folding technique illuminated the transformative power of culinary storytelling.
In college, I look forward to integrating sociology and cultural studies to preserve and celebrate diverse immigrant traditions."""


def test_surface_features_validity():
    extractor = SurfaceFeatureExtractor()
    features = extractor.extract_features(SAMPLE_ESSAY)

    assert isinstance(features, dict)
    assert len(features) >= 25
    
    for name, val in features.items():
        assert not np.isnan(val), f"Feature {name} returned NaN"
        assert not np.isinf(val), f"Feature {name} returned Inf"
        assert isinstance(val, (int, float))

    assert features["surface_ttr"] > 0.0 and features["surface_ttr"] <= 1.0
    assert features["surface_sent_len_mean"] > 5.0
    assert features["surface_char_entropy"] > 0.0


def test_discourse_features_validity():
    extractor = DiscourseFeatureExtractor()
    features = extractor.extract_features(SAMPLE_ESSAY)

    assert isinstance(features, dict)
    assert len(features) >= 10

    for name, val in features.items():
        assert not np.isnan(val), f"Feature {name} returned NaN"
        assert not np.isinf(val), f"Feature {name} returned Inf"
        assert isinstance(val, (int, float))

    assert features["discourse_reflection_density"] >= 0.0
    assert 0.0 <= features["discourse_agency_ratio"] <= 1.0


def test_distributional_features_fit_and_extract():
    extractor = DistributionalFeatureExtractor()
    
    # Synthetic human feature vectors
    np.random.seed(42)
    fake_human_matrix = np.random.randn(20, 10)
    extractor.fit_human_baseline(fake_human_matrix)

    assert extractor.is_fitted

    test_vec = np.random.randn(10)
    para_vecs = [np.random.randn(10) for _ in range(3)]
    dist_feats = extractor.extract_distributional_metrics(test_vec, para_vecs)

    assert dist_feats["dist_human_euclidean"] > 0.0
    assert dist_feats["dist_human_mahalanobis"] >= 0.0
    assert -1.0 <= dist_feats["dist_human_cosine"] <= 1.0
    assert dist_feats["dist_structural_dispersion"] >= 0.0


def test_unified_pipeline_single():
    pipeline = EssayFeaturePipeline()
    base_feats = pipeline.extract_base_features_single(SAMPLE_ESSAY)

    assert "surface_word_entropy" in base_feats
    assert "discourse_agency_action_density" in base_feats
    assert len(base_feats) == len(pipeline.get_base_feature_names())
