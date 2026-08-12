"""Unit tests for text cleaning and normalization."""
import pytest
from src.preprocessing.cleaner import EssayCleaner
from src.preprocessing.normalizer import TextNormalizer


def test_cleaner_strips_ai_prefixes_and_suffixes():
    raw_ai_text = """Here is an admissions essay for the Common App:

Every Saturday morning, our kitchen transformed into a dim sum workshop.
The smell of ginger and scallions filled the air.

I hope this essay captures the prompt and helps your application!"""

    cleaner = EssayCleaner()
    cleaned, meta = cleaner.clean(raw_ai_text)

    assert "Here is an admissions essay" not in cleaned
    assert "I hope this essay captures" not in cleaned
    assert "Every Saturday morning" in cleaned
    assert meta["removed_prefixes"] >= 1
    assert meta["removed_suffixes"] >= 1


def test_cleaner_preserves_paragraphs_and_punctuation():
    text = "Paragraph 1 sentence.\n\nParagraph 2 sentence with semi-colon; and em-dash—yes!"
    cleaner = EssayCleaner()
    cleaned, meta = cleaner.clean(text)

    assert "\n\n" in cleaned
    assert ";" in cleaned
    assert "—" in cleaned
    assert meta["paragraph_count"] == 2


def test_text_normalizer_word_and_char_counts():
    raw = "  Hello   world!  This is a   test. \n\n Second paragraph here. "
    proc = TextNormalizer.process(raw)

    assert proc.word_count == 9
    assert "Hello world! This is a test." in proc.analysis_text
    assert proc.char_count > 0
