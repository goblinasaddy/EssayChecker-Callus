"""Sentence and passage-level evidence generation engine for admissions essays."""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import re
import zlib
import numpy as np

from src.segmentation.segmenter import HierarchicalSegmenter, SentenceSpan, ParagraphSpan, EssaySegmentation
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
    paragraph_idx: int
    start_char: int
    end_char: int
    text: str
    overall_severity: str  # "high", "medium", "human_grounded", "neutral"
    ai_evidence_score: float
    ai_evidence_count: int
    evidence_items: List[Dict[str, Any]]


class EvidenceEngine:
    """
    Two-Level Evidence Attribution Engine:
    Localizes measurable AI and human stylistic evidence to exact sentence and paragraph character spans.
    Compares local measurements against fixed empirical human reference distributions.
    """

    def __init__(self, artifact: ProductionModelArtifact, segmenter: Optional[HierarchicalSegmenter] = None):
        self.artifact = artifact
        self.segmenter = segmenter or HierarchicalSegmenter()
        self.ref_stats = artifact.human_reference_stats

    def analyze_sentences(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> List[HighlightedSpan]:
        """
        Analyzes every sentence (with paragraph-level context fallback) against human reference distributions
        and returns a list of HighlightedSpans with attached empirical evidence records.
        """
        if not text or not text.strip():
            return []

        if segmentation is None:
            segmentation = self.segmenter.segment(text)

        # 1. Pre-calculate paragraph-level metrics for fallback attribution
        para_metrics = {}
        for p in segmentation.paragraphs:
            p_words = re.findall(r"\b\w+(?:'\w+)?\b", p.text.lower())
            p_tokens = max(1, len(p_words))
            p_abstract = [w for w in p_words if w in DiscourseFeatureExtractor.ABSTRACT_VOCABULARY]
            p_abs_density = (len(p_abstract) / p_tokens) * 100
            
            # Local paragraph compression
            p_bytes = p.text.encode("utf-8")
            p_comp_ratio = len(zlib.compress(p_bytes)) / max(1, len(p_bytes)) if len(p_bytes) > 20 else 0.85

            para_metrics[p.paragraph_idx] = {
                "word_count": p_tokens,
                "abstract_words": p_abstract,
                "abstract_density": p_abs_density,
                "compression_ratio": p_comp_ratio,
            }

        highlighted_spans: List[HighlightedSpan] = []

        for s in segmentation.sentences:
            s_text = s.text
            words = re.findall(r"\b\w+(?:'\w+)?\b", s_text.lower())
            token_count = max(1, len(words))

            evidence_items: List[SentenceEvidence] = []
            local_ai_score = 0.0

            # --- A. Abstract Vocabulary Density Check ---
            abstract_words = [w for w in words if w in DiscourseFeatureExtractor.ABSTRACT_VOCABULARY]
            abstract_density = (len(abstract_words) / token_count) * 100
            ref_abstract = self.ref_stats.get("discourse_abstract_vocab_density", {"mean": 1.5, "std": 1.2})
            z_abstract = (abstract_density - ref_abstract["mean"]) / max(0.5, ref_abstract["std"])

            if len(abstract_words) >= 1:
                strength = "Strong" if (len(abstract_words) >= 2 or z_abstract >= 2.5) else "Moderate"
                matched_str = ", ".join([f"'{w}'" for w in sorted(set(abstract_words))])
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_abstract_vocab_density",
                    feature_display_name="Abstract Buzzword Concentration",
                    observed_value=round(abstract_density, 2),
                    reference_mean=round(ref_abstract["mean"], 2),
                    reference_std=round(ref_abstract["std"], 2),
                    deviation_z=round(max(0.0, z_abstract), 2),
                    direction="AI-skewed",
                    evidence_strength=strength,
                    explanation=(
                        f"Sentence contains {abstract_density:.1f}% abstract concept vocabulary vs human reference mean of {ref_abstract['mean']:.1f}%. "
                        f"Detected keyword(s): {matched_str}."
                    )
                ))
                local_ai_score += 1.5 if strength == "Strong" else 1.0

            # --- B. Formulaic / Expository Takeaway Patterns ---
            s_lower = s_text.lower()
            matched_moral = []
            for pat in DiscourseFeatureExtractor.FORMULAIC_MORAL_PATTERNS:
                if re.search(pat, s_lower):
                    clean_pat = pat.replace(r"\b", "").replace(r"(?:", "").replace(r")?", "").replace(r")", "")
                    matched_moral.append(clean_pat)

            if matched_moral:
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_formulaic_moral_density",
                    feature_display_name="Formulaic Rhetorical Framing",
                    observed_value=1.0,
                    reference_mean=0.1,
                    reference_std=0.3,
                    deviation_z=3.0,
                    direction="AI-skewed",
                    evidence_strength="Strong" if len(abstract_words) >= 1 else "Moderate",
                    explanation=f"Contains standardized expository/moral conclusion phrasing: '{matched_moral[0]}'."
                ))
                local_ai_score += 1.5

            # --- C. Personal Agency & Concrete Grounding vs Passive State ---
            first_person = any(w in ("i", "my", "me", "we", "our") for w in words)
            action_verbs = [w for w in words if w in DiscourseFeatureExtractor.ACTION_VERBS]
            passive_verbs = [w for w in words if w in DiscourseFeatureExtractor.PASSIVE_STATE_VERBS]
            concrete_words = [w for w in words if w in DiscourseFeatureExtractor.CONCRETE_SENSORY_WORDS]

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
                    deviation_z=1.4,
                    direction="AI-skewed",
                    evidence_strength="Mild",
                    explanation="Protagonist agency is framed via passive emotional states or observations rather than active decisions."
                ))
                local_ai_score += 0.6
            elif len(concrete_words) >= 1 and len(abstract_words) == 0 and len(matched_moral) == 0:
                # Genuine Positive Human Grounding
                matched_conc = ", ".join([f"'{w}'" for w in sorted(set(concrete_words))])
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_concrete_sensory_density",
                    feature_display_name="Concrete Situational Grounding",
                    observed_value=round((len(concrete_words) / token_count) * 100, 2),
                    reference_mean=2.8,
                    reference_std=1.5,
                    deviation_z=0.0,
                    direction="Human-skewed",
                    evidence_strength="Moderate",
                    explanation=f"Specific tactile, physical, or situational sensory grounding: {matched_conc}."
                ))

            # --- D. Paragraph Context Fallback ---
            p_info = para_metrics.get(s.paragraph_idx, {})
            if local_ai_score < 1.0 and p_info.get("abstract_density", 0.0) >= 3.5 and len(p_info.get("abstract_words", [])) >= 2:
                # Paragraph as a whole is heavily abstract
                p_abs_str = ", ".join([f"'{w}'" for w in sorted(set(p_info["abstract_words"]))[:3]])
                evidence_items.append(SentenceEvidence(
                    sentence_idx=s.sentence_idx,
                    paragraph_idx=s.paragraph_idx,
                    start_char=s.start_char,
                    end_char=s.end_char,
                    text=s_text,
                    feature_name="discourse_paragraph_abstraction",
                    feature_display_name="Paragraph Abstract Density",
                    observed_value=round(p_info["abstract_density"], 2),
                    reference_mean=1.5,
                    reference_std=1.2,
                    deviation_z=round((p_info["abstract_density"] - 1.5) / 1.2, 2),
                    direction="AI-skewed",
                    evidence_strength="Mild",
                    explanation=f"Surrounding paragraph #{s.paragraph_idx + 1} contains elevated abstract vocabulary density ({p_info['abstract_density']:.1f}%). Keywords: {p_abs_str}."
                ))
                local_ai_score += 0.8

            # --- E. Local Evidence Severity Classification ---
            ai_items = [e for e in evidence_items if e.direction == "AI-skewed"]
            has_strong_ai = any(e.evidence_strength == "Strong" for e in ai_items)
            human_items = [e for e in evidence_items if e.direction == "Human-skewed"]

            if has_strong_ai or local_ai_score >= 1.8:
                severity = "high"
            elif len(ai_items) >= 1 or local_ai_score >= 0.8:
                severity = "medium"
            elif len(human_items) >= 1 and len(ai_items) == 0:
                severity = "human_grounded"
            else:
                severity = "neutral"

            highlighted_spans.append(HighlightedSpan(
                span_id=f"span_{s.sentence_idx}",
                sentence_idx=s.sentence_idx,
                paragraph_idx=s.paragraph_idx,
                start_char=s.start_char,
                end_char=s.end_char,
                text=s_text,
                overall_severity=severity,
                ai_evidence_score=round(local_ai_score, 2),
                ai_evidence_count=len(ai_items),
                evidence_items=[asdict(e) for e in evidence_items],
            ))

        return highlighted_spans
