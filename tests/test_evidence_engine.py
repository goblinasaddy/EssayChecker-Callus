"""Unit tests for two-level evidence attribution engine and character offset alignment."""
import pytest
from src.inference.model_store import ProductionModelArtifact
from src.inference.evidence_engine import EvidenceEngine
from src.inference.detector import AdmissionsAIDetector
from src.segmentation.segmenter import HierarchicalSegmenter


def test_evidence_engine_exact_character_spans():
    artifact = ProductionModelArtifact.load()
    engine = EvidenceEngine(artifact)

    essay = (
        "This multifaceted environment served as a powerful catalyst for my personal evolution. "
        "Every Saturday morning, we assembled bicycle frames in the garage with my uncle."
    )

    spans = engine.analyze_sentences(essay)

    assert len(spans) == 2
    
    # Exact substring reconstruction
    for span in spans:
        slice_text = essay[span.start_char:span.end_char]
        assert slice_text == span.text, f"Offset mismatch: '{slice_text}' vs '{span.text}'"

    # Span 1 has abstract buzzwords ("multifaceted", "catalyst") -> AI skewed (HIGH/RED)
    span1 = spans[0]
    assert span1.overall_severity in ("high", "medium")
    assert any(e["feature_name"] == "discourse_abstract_vocab_density" for e in span1.evidence_items)


def test_evidence_engine_evidence_record_schema():
    artifact = ProductionModelArtifact.load()
    engine = EvidenceEngine(artifact)

    text = "Ultimately, this transformative experience taught me that resilience is essential."
    spans = engine.analyze_sentences(text)

    assert len(spans) == 1
    span = spans[0]
    assert len(span.evidence_items) >= 1

    item = span.evidence_items[0]
    required_keys = [
        "sentence_idx", "start_char", "end_char", "feature_name", "feature_display_name",
        "observed_value", "reference_mean", "reference_std", "deviation_z", "direction",
        "evidence_strength", "explanation"
    ]
    for key in required_keys:
        assert key in item, f"Missing required evidence key: {key}"


def test_evidence_engine_human_grounded_vs_neutral():
    artifact = ProductionModelArtifact.load()
    engine = EvidenceEngine(artifact)

    text_human = "Flour coated the linoleum tiles like fresh snow, and the steady rhythmic thumping of my grandmother's cleaver set the tempo."
    spans_human = engine.analyze_sentences(text_human)
    assert len(spans_human) == 1
    # Contains concrete sensory words ("flour", "linoleum", "cleaver")
    assert spans_human[0].overall_severity == "human_grounded"

    text_neutral = "The library was located on the second floor of the main administrative building."
    spans_neutral = engine.analyze_sentences(text_neutral)
    assert len(spans_neutral) == 1
    assert spans_neutral[0].overall_severity == "neutral"


def test_evidence_engine_global_and_local_alignment():
    detector = AdmissionsAIDetector()
    
    # Synthetic India Gate essay
    india_gate = (
        "India Gate, located in the heart of New Delhi, stands as an imposing monumental tribute "
        "to the brave soldiers who sacrificed their lives during the First World War. "
        "This majestic archway serves as a poignant reminder of valor, resilience, and historical significance. "
        "The architectural grandeur of the monument, coupled with the eternal flame of the Amar Jawan Jyoti, "
        "evokes a profound sense of patriotism and collective memory among visitors from around the world. "
        "Throughout the decades, this iconic landmark has continued to inspire generations, "
        "symbolizing the enduring spirit of national unity and cultural heritage in an ever-evolving global society."
    )

    res = detector.analyze(india_gate)
    assert res["status"] == "SUCCESS"
    assert res["assessment"]["verdict_code"] == "LIKELY_AI_GENERATED"
    assert res["assessment"]["calibrated_ai_probability"] >= 0.85
    
    # Check that local AI evidence is actually flagged (RED spans > 0)
    high_spans = [s for s in res["highlighted_spans"] if s["overall_severity"] == "high"]
    assert len(high_spans) >= 2

    # Check that top global decision drivers are returned with non-zero impact
    assert len(res["evidence"]["top_ai_evidence"]) >= 3
    assert all("impact_score" in d for d in res["evidence"]["top_ai_evidence"])


def test_evidence_engine_paragraph_fallback():
    artifact = ProductionModelArtifact.load()
    engine = EvidenceEngine(artifact)

    # Paragraph with high abstract density across short sentences
    para_text = (
        "Our journey was transformative. "
        "We embraced the dichotomy of our environment. "
        "This multifaceted experience fostered resilience."
    )
    spans = engine.analyze_sentences(para_text)
    assert len(spans) == 3
    # All sentences should exhibit AI-skewed markers (either sentence-level or paragraph-level)
    assert all(s.overall_severity in ("high", "medium") for s in spans)
