"""Sentence and passage-level evidence generation engine for admissions essays."""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import numpy as np

from src.segmentation.segmenter import HierarchicalSegmenter, SentenceSpan, EssaySegmentation
from src.features.surface import SurfaceFeatureExtractor
from src.features.discourse import DiscourseFeatureExtractor
from src.inference.model_store import ProductionModelArtifact


@dataclass
class SentenceEvidence:
    sentence_idx: int
    paragraph_idx: int
    start_char: int
    end_char: int
    text: str
    feature_name: str
    feature_display_name: str
    observed_value: float
    reference_mean: float
    reference_std: float
    deviation_z: float
    direction: str  # "AI-skewed" or "Human-skewed"
    evidence_strength: str  # "Mild", "Moderate", "Strong"
    explanation: str


@dataclass
class HighlightedSpan:
    span_id: str
    sentence_idx: int
    start_char: int
    end_char: int
    text: str
    overall_severity: str  # "high", "medium", "low", "neutral"
    ai_evidence_count: int
    evidence_items: List[Dict[str, Any]]


class EvidenceEngine:
    """
    Localizes measurable AI and human stylistic evidence to exact sentence character spans.
    Compares local sentence measurements against the fixed human reference distribution.
    """

    def __init__(self, artifact: ProductionModelArtifact, segmenter: Optional[HierarchicalSegmenter] = None):
        self.artifact = artifact
        self.segmenter = segmenter or HierarchicalSegmenter()
        self.ref_stats = artifact.human_reference_stats

    def analyze_sentences(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> List[HighlightedSpan]:
        """
        Analyzes every sentence in the essay against human reference distributions
        and returns a list of HighlightedSpans with attached evidence.
        """
        if not text or not text.strip():
            return []

        if segmentation is None:
            segmentation = self.segmenter.segment(text)

        highlighted_spans: List[HighlightedSpan] = []

        for s in segmentation.sentences:
            s_text = s.text
            words = re.findall(r"\b\w+(?:'\w+)?\b", s_text.lower())
            token_count = max(1, len(words))

            evidence_items: List[SentenceEvidence] = []

            # 1. Abstract Vocabulary Density Check
            abstract_words = [w for w in words if w in DiscourseFeatureExtractor.ABSTRACT_VOCABULARY]
            abstract_density = (len(abstract_words) / token_count) * 100
            ref_abstract = self.ref_stats.get("discourse_abstract_vocab_density", {"mean": 1.5, "std": 1.2})
            z_abstract = (abstract_density - ref_abstract["mean"]) / max(0.5, ref_abstract["std"])

            if abstract_density >= 6.0 or (z_abstract >= 1.8 and len(abstract_words) >= 1):
                strength = "Strong" if z_abstract >= 3.0 or len(abstract_words) >= 2 else "Moderate"
                matched_str = ", ".join([f"'{w}'" for w in set(abstract_words)])
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_abstract_vocab_density",
                    feature_display_name="Abstract Buzzword Density",
                    observed_value=round(abstract_density, 2),
                    reference_mean=round(ref_abstract["mean"], 2),
                    reference_std=round(ref_abstract["std"], 2),
                    deviation_z=round(z_abstract, 2),
                    direction="AI-skewed",
                    evidence_strength=strength,
                    explanation=(
                        f"Sentence contains high abstract buzzword density ({abstract_density:.1f}% vs human reference {ref_abstract['mean']:.1f}%). "
                        f"Detected keywords: {matched_str}."
                    )
                ))

            # 2. Formulaic Moral / Takeaway Wrap Check
            s_lower = s_text.lower()
            matched_moral = []
            for pat in DiscourseFeatureExtractor.FORMULAIC_MORAL_PATTERNS:
                if re.search(pat, s_lower):
                    matched_moral.append(pat.replace(r"\b", "").replace(" (?:experience )?", " "))

            if matched_moral:
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_formulaic_moral_density",
                    feature_display_name="Formulaic Lesson Pattern",
                    observed_value=1.0,
                    reference_mean=0.1,
                    reference_std=0.3,
                    deviation_z=3.0,
                    direction="AI-skewed",
                    evidence_strength="Moderate",
                    explanation=f"Contains formulaic moralizing phrasing: '{matched_moral[0]}'."
                ))

            # 3. Personal Agency Check (First person without active verbs vs with active verbs)
            first_person = any(w in ("i", "my", "me") for w in words)
            action_verbs = [w for w in words if w in DiscourseFeatureExtractor.ACTION_VERBS]
            passive_verbs = [w for w in words if w in DiscourseFeatureExtractor.PASSIVE_STATE_VERBS]

            if first_person and len(passive_verbs) > 0 and len(action_verbs) == 0:
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_agency_passive_density",
                    feature_display_name="Passive Protagonist Framing",
                    observed_value=float(len(passive_verbs)),
                    reference_mean=0.3,
                    reference_std=0.5,
                    deviation_z=1.5,
                    direction="AI-skewed",
                    evidence_strength="Mild",
                    explanation="Personal reference framed via passive observation/feeling rather than concrete action."
                ))
            elif first_person and len(action_verbs) >= 1 and len(abstract_words) == 0:
                # Strong human grounding marker
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_agency_action_density",
                    feature_display_name="Concrete Personal Agency",
                    observed_value=float(len(action_verbs)),
                    reference_mean=1.2,
                    reference_std=0.8,
                    deviation_z=0.0,
                    direction="Human-skewed",
                    evidence_strength="Moderate",
                    explanation=f"Direct first-person action grounding: '{action_verbs[0]}'."
                ))

            # 4. Sentence Pacing & Extreme Length Check
            ref_len = self.ref_stats.get("surface_sent_len_mean", {"mean": 21.0, "std": 7.0})
            if token_count >= 40:
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="surface_sent_len_mean",
                    feature_display_name="Elongated Complex Clause",
                    observed_value=float(token_count),
                    reference_mean=round(ref_len["mean"], 1),
                    reference_std=round(ref_len["std"], 1),
                    deviation_z=round((token_count - ref_len["mean"]) / max(1.0, ref_len["std"]), 2),
                    direction="AI-skewed" if len(abstract_words) > 0 else "Human-skewed",
                    evidence_strength="Mild",
                    explanation=f"Sentence length ({token_count} words) is substantially longer than typical admissions pacing."
                ))

            # Determine overall span severity
            ai_items = [e for e in evidence_items if e.direction == "AI-skewed"]
            has_strong_ai = any(e.evidence_strength == "Strong" for e in ai_items)
            
            if has_strong_ai or len(ai_items) >= 2:
                severity = "high"
            elif len(ai_items) == 1:
                severity = "medium"
            elif any(e.direction == "Human-skewed" for e in evidence_items):
                severity = "human_grounded"
            else:
                severity = "neutral"

            highlighted_spans.append(HighlightedSpan(
                span_id=f"span_{s.sentence_idx}",
                sentence_idx=s.sentence_idx,
                start_char=s.start_char,
                end_char=s.end_char,
                text=s_text,
                overall_severity=severity,
                ai_evidence_count=len(ai_items),
                evidence_items=[asdict(e) for e in evidence_items],
            ))

        return highlighted_spans
