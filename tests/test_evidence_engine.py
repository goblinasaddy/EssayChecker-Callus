"""Unit tests for sentence-level evidence engine and character offset alignment."""
import pytest
from src.inference.model_store import ProductionModelArtifact
from src.inference.evidence_engine import EvidenceEngine
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

    # Span 1 has abstract buzzwords ("multifaceted", "catalyst") -> AI skewed
    span1 = spans[0]
    assert span1.overall_severity in ("high", "medium")
    assert any(e["feature_name"] == "discourse_abstract_vocab_density" for e in span1.evidence_items)

    # Span 2 has concrete action ("assembled") -> human grounded or neutral
    span2 = spans[1]
    assert span2.overall_severity in ("human_grounded", "neutral")


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
