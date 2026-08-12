"""Unit tests for hierarchical paragraph and sentence segmentation."""
import pytest
from src.segmentation.segmenter import HierarchicalSegmenter


def test_segmenter_character_span_fidelity():
    essay = (
        "The odometer on my father's car read 287,000 miles. "
        "We spent weekends fixing the transmission in the driveway.\n\n"
        "Tearing down that engine taught me how mechanical systems interact. "
        "It was my first laboratory; it challenged my patience."
    )

    segmenter = HierarchicalSegmenter()
    seg = segmenter.segment(essay)

    assert seg.total_paragraphs == 2
    assert seg.total_sentences == 4

    # Verify that raw_text slice at start_char:end_char exactly matches sentence.text
    for s in seg.sentences:
        reconstructed = essay[s.start_char:s.end_char]
        assert reconstructed == s.text, f"Mismatch: expected '{s.text}' got '{reconstructed}'"
        assert s.token_count > 0
        assert len(s.tokens) == s.token_count


def test_segmenter_empty_string_handling():
    segmenter = HierarchicalSegmenter()
    seg = segmenter.segment("")

    assert seg.total_paragraphs == 0
    assert seg.total_sentences == 0
    assert seg.total_tokens == 0
    assert seg.paragraphs == []
    assert seg.sentences == []


def test_sentence_lookup_at_char():
    text = "First short sentence. Second longer sentence here."
    segmenter = HierarchicalSegmenter()
    seg = segmenter.segment(text)

    s1 = seg.get_sentence_at_char(5)
    assert s1 is not None
    assert s1.sentence_idx == 0
    assert "First short sentence" in s1.text

    s2 = seg.get_sentence_at_char(30)
    assert s2 is not None
    assert s2.sentence_idx == 1
    assert "Second longer sentence" in s2.text
