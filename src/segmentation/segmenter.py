"""Hierarchical segmentation engine for essays, paragraphs, and sentences."""
from dataclasses import dataclass, field
from typing import List, Optional
import re
import nltk


@dataclass
class SentenceSpan:
    sentence_idx: int
    paragraph_idx: int
    text: str
    start_char: int
    end_char: int
    token_count: int
    tokens: List[str] = field(default_factory=list)


@dataclass
class ParagraphSpan:
    paragraph_idx: int
    text: str
    start_char: int
    end_char: int
    sentence_count: int
    sentences: List[SentenceSpan] = field(default_factory=list)


@dataclass
class EssaySegmentation:
    raw_text: str
    total_paragraphs: int
    total_sentences: int
    total_tokens: int
    paragraphs: List[ParagraphSpan] = field(default_factory=list)
    sentences: List[SentenceSpan] = field(default_factory=list)

    def get_sentence_at_char(self, char_offset: int) -> Optional[SentenceSpan]:
        for s in self.sentences:
            if s.start_char <= char_offset < s.end_char:
                return s
        return None


class HierarchicalSegmenter:
    """
    Splits text into paragraphs and sentences while retaining exact character offsets.
    Guarantees that raw_text[s.start_char:s.end_char] exactly matches s.text.
    """

    def __init__(self):
        try:
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        except Exception:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            self.sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def segment(self, text: str) -> EssaySegmentation:
        if not text:
            return EssaySegmentation(
                raw_text="",
                total_paragraphs=0,
                total_sentences=0,
                total_tokens=0,
                paragraphs=[],
                sentences=[],
            )

        paragraphs: List[ParagraphSpan] = []
        all_sentences: List[SentenceSpan] = []
        global_sent_idx = 0
        total_tokens = 0

        # Find paragraph boundaries using regex to preserve character positions
        para_matches = list(re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text))

        for p_idx, p_match in enumerate(para_matches):
            p_start, p_end = p_match.span()
            p_text = p_match.group(0)

            # Sentence segmentation within the paragraph slice
            sent_spans_in_p = self._segment_sentences_with_spans(p_text, p_start, p_idx, global_sent_idx)
            
            p_sentences: List[SentenceSpan] = []
            for s in sent_spans_in_p:
                p_sentences.append(s)
                all_sentences.append(s)
                total_tokens += s.token_count
                global_sent_idx += 1

            p_span = ParagraphSpan(
                paragraph_idx=p_idx,
                text=p_text,
                start_char=p_start,
                end_char=p_end,
                sentence_count=len(p_sentences),
                sentences=p_sentences,
            )
            paragraphs.append(p_span)

        return EssaySegmentation(
            raw_text=text,
            total_paragraphs=len(paragraphs),
            total_sentences=len(all_sentences),
            total_tokens=total_tokens,
            paragraphs=paragraphs,
            sentences=all_sentences,
        )

    def _segment_sentences_with_spans(
        self, para_text: str, para_start_offset: int, p_idx: int, starting_sent_idx: int
    ) -> List[SentenceSpan]:
        """Extracts sentences and their exact character spans within a paragraph."""
        sentences: List[SentenceSpan] = []
        
        # Punkt sentence tokenizer span_tokenize
        current_idx = starting_sent_idx
        for s_start, s_end in self.sent_tokenizer.span_tokenize(para_text):
            s_text = para_text[s_start:s_end]
            tokens = re.findall(r"\b\w+(?:'\w+)?\b|[.,!?;:\"'—–-]", s_text)
            
            sent_span = SentenceSpan(
                sentence_idx=current_idx,
                paragraph_idx=p_idx,
                text=s_text,
                start_char=para_start_offset + s_start,
                end_char=para_start_offset + s_end,
                token_count=len(tokens),
                tokens=tokens,
            )
            sentences.append(sent_span)
            current_idx += 1

        return sentences
