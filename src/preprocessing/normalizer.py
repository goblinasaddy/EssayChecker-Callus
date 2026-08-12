"""Text normalizer maintaining distinct Raw Text and Analysis Text representations."""
import re
from typing import NamedTuple


class ProcessedText(NamedTuple):
    raw_text: str
    cleaned_text: str
    analysis_text: str
    word_count: int
    char_count: int


class TextNormalizer:
    """Provides consistent text normalization for feature extraction."""

    @staticmethod
    def normalize_for_analysis(text: str) -> str:
        """
        Creates an analysis-ready string:
        - Replaces multiple consecutive spaces with a single space.
        - Preserves sentence-ending punctuation and casing.
        """
        if not text:
            return ""
        # Collapse multiple inline spaces while keeping paragraph line breaks
        lines = []
        for line in text.split("\n"):
            normalized_line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(normalized_line)
        return "\n".join(lines)

    @classmethod
    def process(cls, raw_text: str, cleaner=None) -> ProcessedText:
        if cleaner is not None:
            cleaned, _ = cleaner.clean(raw_text)
        else:
            cleaned = raw_text.strip()
        analysis = cls.normalize_for_analysis(cleaned)
        words = re.findall(r"\b\w+\b", analysis)
        return ProcessedText(
            raw_text=raw_text,
            cleaned_text=cleaned,
            analysis_text=analysis,
            word_count=len(words),
            char_count=len(analysis),
        )
