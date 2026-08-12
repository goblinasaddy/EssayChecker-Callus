"""Unit tests for Phase 2 production inference engine and calibrated detector."""
import pytest
import numpy as np
from src.inference.detector import AdmissionsAIDetector
from src.inference.model_store import ProductionModelArtifact


def test_detector_initialization_and_schema():
    detector = AdmissionsAIDetector()
    assert detector.artifact is not None
    assert detector.artifact.model_version == "2.0.0"
    assert len(detector.artifact.feature_names) == 48
    assert "surface_compression_ratio" in detector.artifact.feature_names
    assert "discourse_abstract_vocab_density" in detector.artifact.feature_names
    assert "dist_human_mahalanobis" in detector.artifact.feature_names


def test_detector_analyze_human_essay():
    detector = AdmissionsAIDetector()
    human_text = """Every Saturday morning, our kitchen transformed into a bustling dim sum factory.
Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day.
My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings.
At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop.
Nai Nai never scolded me; she merely pressed another circle of dough into my palm, her rough calloused thumbs guiding mine with silent patience.
Through those dumplings, I learned the quiet language of my heritage."""

    res = detector.analyze(human_text)

    assert res["status"] == "SUCCESS"
    assert res["model_version"] == "2.0.0"
    assert "assessment" in res
    assert res["assessment"]["verdict_code"] in ("LIKELY_HUMAN", "UNCERTAIN")
    assert 0.0 <= res["assessment"]["calibrated_ai_probability"] <= 1.0
    assert len(res["highlighted_spans"]) > 0
    assert "disclaimers" in res


def test_detector_analyze_ai_essay():
    detector = AdmissionsAIDetector()
    ai_text = """Growing up at the intersection of two distinct cultures, my identity was forged through the vibrant tapestry of traditions that adorned our family home.
This multifaceted environment served as a powerful catalyst for my personal evolution, fostering a deep appreciation for the profound interconnectedness of diverse human experiences.
Throughout my formative years, I often found myself navigating the complex dichotomy between preserving my cultural heritage and assimilating into my community.
Rather than viewing this as an obstacle, I embraced it as a quintessential opportunity for intellectual and emotional enrichment.
Ultimately, this transformative journey has illuminated the pivotal importance of cultural diplomacy in an increasingly globalized world."""

    res = detector.analyze(ai_text)

    assert res["status"] == "SUCCESS"
    assert res["assessment"]["verdict_code"] in ("LIKELY_AI_GENERATED", "LIKELY_AI_ASSISTED")
    assert res["assessment"]["calibrated_ai_probability"] >= 0.60
    assert res["metadata"]["flagged_sentence_count"] >= 1


def test_detector_empty_and_short_input_edge_cases():
    detector = AdmissionsAIDetector()

    # Empty string
    res_empty = detector.analyze("")
    assert res_empty["status"] == "ERROR"

    # Too short (< 20 words)
    res_short = detector.analyze("This essay is way too short to evaluate.")
    assert res_short["status"] == "INSUFFICIENT_LENGTH"
    assert res_short["word_count"] < 20
