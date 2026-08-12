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
    human_text = """Every Saturday morning, our kitchen transformed into a bustling dim sum factory. Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother’s cleaver set the tempo for the day. My job was simple yet unforgiving: pinch the pleats of the siu mai dumplings. At seven years old, my clumsy fingers tore through delicate wrappers, spilling seasoned pork across the countertop. Nai Nai never scolded me; she merely pressed another circle of dough into my palm, her rough calloused thumbs guiding mine with silent patience.

Through those dumplings, I learned the quiet language of my heritage. In a household where 'I love you' was rarely spoken aloud, affection was measured in steaming bamboo baskets and bowls of slow-simmered winter melon soup. When my family moved to suburban Ohio in fifth grade, that culinary dialect became my anchor. While classmates brought Lunchables, I unpacked containers of fragrant scallion pancakes, learning to embrace the curious glances rather than shrink from them.

As I grew older, this kitchen apprenticeship evolved into a broader curiosity about food anthropology and cultural preservation. In high school, I founded the Cultural Heritage Exchange, organizing community dinners where students from immigrant backgrounds shared their families' traditional dishes along with the stories behind them. Standing before twenty peers, teaching them Nai Nai’s precise three-fold pleating technique, I realized that food is more than sustenance—it is living history. At university, I hope to continue bridging cultural divides, combining sociology and culinary traditions to ensure immigrant narratives are preserved and celebrated."""

    res = detector.analyze(human_text)

    assert res["status"] == "SUCCESS"
    assert res["model_version"] == "2.0.0"
    assert "assessment" in res
    assert res["assessment"]["verdict_code"] == "LIKELY_HUMAN"
    assert res["assessment"]["calibrated_ai_probability"] < 0.20
    assert len(res["highlighted_spans"]) > 0
    assert "disclaimers" in res


def test_detector_analyze_ai_essay():
    detector = AdmissionsAIDetector()
    ai_text = """Growing up at the intersection of two distinct cultures, my identity was forged through the vibrant tapestry of traditions that adorned our family home. Every Sunday afternoon, our living room resonated with the harmonious melodies of traditional folk songs juxtaposed against the rhythmic cadence of contemporary American music. This multifaceted environment served as a powerful catalyst for my personal evolution, fostering a deep appreciation for the profound interconnectedness of diverse human experiences.

Throughout my formative years, I often found myself navigating the complex dichotomy between preserving my cultural heritage and assimilating into my suburban community. Rather than viewing this duality as an insurmountable obstacle, I embraced it as a quintessential opportunity for intellectual and emotional enrichment. In high school, I sought to bridge these disparate worlds by organizing multicultural symposiums that celebrated diversity and fostered dialogue among students from myriad backgrounds.

Ultimately, this transformative journey has illuminated the pivotal importance of cultural diplomacy in an increasingly globalized world. As I prepare to embark on the next chapter of my academic career, I am eager to leverage these multifaceted insights to champion inclusive discourse, fostering environments where divergent viewpoints are not merely tolerated, but enthusiastically embraced as catalysts for positive societal transformation."""

    res = detector.analyze(ai_text)

    assert res["status"] == "SUCCESS"
    assert res["assessment"]["verdict_code"] == "LIKELY_AI_GENERATED"
    assert res["assessment"]["calibrated_ai_probability"] >= 0.85
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


def test_detector_short_text_insufficient_evidence_safeguard():
    detector = AdmissionsAIDetector()

    # Mahatma Gandhi short text (36 words)
    text_short = "Mahatma Gandhi is father of India, he was one of the freedom Fighter of India, We love him. His face is printed in all paper currency of India Rupees. I am millions faod ajdn akdnd jwnfl"
    res = detector.analyze(text_short)

    assert res["status"] == "SUCCESS"
    assert res["assessment"]["verdict_code"] == "INSUFFICIENT_EVIDENCE"
    assert res["assessment"]["category"] == "Insufficient Evidence"
    assert res["assessment"]["is_indeterminate"] is True
    assert "at least 75 words" in res["assessment"]["summary"]
    # Ensure it did NOT falsely claim 100% AI
    assert res["assessment"]["calibrated_ai_probability"] == 0.50
