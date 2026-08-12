"""Text cleaning and artifact stripping for college admissions essays."""
import re
import unicodedata
from typing import Tuple


class EssayCleaner:
    """Cleans admissions essays while strictly preserving structural features."""

    # Regex patterns for common AI prompt leakage and formatting wrappers
    AI_PREFIX_PATTERNS = [
        r"^(?:Here(?:'s| is) (?:an|a|the) (?:college admissions |college |admissions )?essay(?::|;|\.\.\.|\.)?)\s*",
        r"^(?:Title:\s*[^\n]+\n+)",
        r"^(?:Prompt:\s*[^\n]+\n+)",
        r"^(?:Sure,?\s+(?:here is|I can provide)[^\n]+\n+)",
        r"^(?:As requested,?\s+[^\n]+\n+)",
    ]

    AI_SUFFIX_PATTERNS = [
        r"\n+(?:I hope this essay (?:meets your needs|helps you|captures the prompt)[^\n]*)$",
        r"\n+(?:Note:\s*[^\n]+)$",
        r"\n+(?:Word count:\s*\d+[^\n]*)$",
    ]

    def __init__(self, strip_markdown_headers: bool = True):
        self.strip_markdown_headers = strip_markdown_headers

    def clean(self, raw_text: str) -> Tuple[str, dict]:
        """
        Cleans raw essay text and returns (cleaned_text, audit_metadata).
        
        Preserves paragraph and sentence structures, punctuation, and case.
        """
        if not raw_text or not isinstance(raw_text, str):
            return "", {"is_empty": True, "removed_prefixes": 0, "removed_suffixes": 0}

        # 1. Normalize unicode (NFKC normalization to handle smart quotes, em-dashes uniformly)
        text = unicodedata.normalize("NFKC", raw_text)

        # 2. Standardize line endings to \n
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        removed_prefixes = 0
        removed_suffixes = 0

        # 3. Strip artificial generation prefixes
        for pattern in self.AI_PREFIX_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                text = text[match.end():]
                removed_prefixes += 1

        # 4. Strip artificial generation suffixes
        for pattern in self.AI_SUFFIX_PATTERNS:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                text = text[:match.start()]
                removed_suffixes += 1

        # 5. Optionally strip top-level markdown headers like # My Journey
        if self.strip_markdown_headers:
            text = re.sub(r"^#+\s+[^\n]+\n+", "", text)

        # 6. Normalize trailing/leading whitespace per line, but preserve paragraph breaks (\n\n)
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        cleaned_text = "\n\n".join(paragraphs)

        audit_meta = {
            "is_empty": len(cleaned_text.strip()) == 0,
            "raw_char_count": len(raw_text),
            "cleaned_char_count": len(cleaned_text),
            "paragraph_count": len(paragraphs),
            "removed_prefixes": removed_prefixes,
            "removed_suffixes": removed_suffixes,
        }

        return cleaned_text, audit_meta
