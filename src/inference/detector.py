"""Production AI Authorship Detector with multi-level evidence attribution."""
from typing import Dict, List, Any, Optional
import os
import numpy as np
import pandas as pd

from src.preprocessing.cleaner import EssayCleaner
from src.preprocessing.normalizer import TextNormalizer
from src.segmentation.segmenter import HierarchicalSegmenter, EssaySegmentation
from src.features.pipeline import EssayFeaturePipeline
from src.inference.model_store import ProductionModelArtifact
from src.inference.evidence_engine import EvidenceEngine, HighlightedSpan


class AdmissionsAIDetector:
    """
    Production AI Authorship Analysis Engine for College Admissions Essays.
    Combines calibrated tabular model predictions with fine-grained sentence-level evidence attribution.
    Never relies on an LLM for final classification or scoring.
    """

    def __init__(self, artifact_path: Optional[str] = None):
        if artifact_path and os.path.exists(artifact_path):
            self.artifact = ProductionModelArtifact.load(artifact_path)
        else:
            default_path = ProductionModelArtifact.DEFAULT_ARTIFACT_PATH
            if os.path.exists(default_path):
                self.artifact = ProductionModelArtifact.load(default_path)
            else:
                # Train and export artifact if not present
                self.artifact = ProductionModelArtifact.train_and_export(output_path=default_path)

        self.cleaner = EssayCleaner()
        self.normalizer = TextNormalizer()
        self.segmenter = HierarchicalSegmenter()
        self.pipeline = EssayFeaturePipeline()
        
        # Hydrate pipeline distributional extractor with fixed training human baseline
        dist_base = self.artifact.distributional_baseline
        if dist_base and dist_base.get("is_fitted"):
            self.pipeline.dist_extractor.is_fitted = True
            self.pipeline.dist_extractor.human_centroid = np.array(dist_base["human_centroid"])
            self.pipeline.dist_extractor.feature_means = np.array(dist_base["feature_means"])
            self.pipeline.dist_extractor.feature_stds = np.array(dist_base["feature_stds"])
            self.pipeline.dist_extractor.human_cov_inv = np.array(dist_base["human_cov_inv"])

        self.evidence_engine = EvidenceEngine(self.artifact, self.segmenter)

    def analyze(self, raw_text: str) -> Dict[str, Any]:
        """
        Executes full evidence-based authorship analysis on an admissions essay.
        """
        if not raw_text or not raw_text.strip():
            return self._empty_response("Input text is empty.")

        cleaned_text, clean_meta = self.cleaner.clean(raw_text)
        words = cleaned_text.split()
        if len(words) < 20:
            return self._short_text_response(raw_text, len(words))

        # Hierarchical segmentation on raw text to preserve character offsets exactly
        segmentation = self.segmenter.segment(raw_text)

        # 1. Feature Extraction
        base_dict = self.pipeline.extract_base_features_single(cleaned_text)
        base_feature_names = self.pipeline.get_base_feature_names()
        
        # Paragraph vectors for dispersion
        p_vectors = []
        for p in segmentation.paragraphs:
            p_dict = self.pipeline.extract_base_features_single(p.text)
            p_vec = np.array([p_dict.get(k, 0.0) for k in base_feature_names], dtype=float)
            p_vectors.append(p_vec)

        base_vec = np.array([base_dict.get(k, 0.0) for k in base_feature_names], dtype=float)
        dist_dict = self.pipeline.dist_extractor.extract_distributional_metrics(base_vec, p_vectors)

        all_features = {}
        all_features.update(base_dict)
        all_features.update(dist_dict)

        # 2. Model Scoring (Calibrated Logistic Regression Logit)
        feature_cols = self.artifact.feature_names
        x_raw = np.array([all_features.get(f, 0.0) for f in feature_cols], dtype=float)
        
        # Standardize using saved scaler parameters
        means = np.array([self.artifact.scaler_mean.get(f, 0.0) for f in feature_cols], dtype=float)
        scales = np.array([self.artifact.scaler_scale.get(f, 1.0) for f in feature_cols], dtype=float)
        scales[scales < 1e-6] = 1.0
        x_scaled = (x_raw - means) / scales

        coefs = np.array([self.artifact.coefficients.get(f, 0.0) for f in feature_cols], dtype=float)
        logit = float(np.dot(x_scaled, coefs) + self.artifact.intercept)
        ai_probability = float(1.0 / (1.0 + np.exp(-logit)))

        # 3. Categorical Assessment Mapping
        thresholds = self.artifact.thresholds
        if ai_probability < thresholds["human_max"]:
            category = "Likely Human"
            verdict_code = "LIKELY_HUMAN"
            verdict_color = "#10b981"  # Emerald Green
            confidence_level = "High" if ai_probability < 0.20 else "Moderate"
            summary_desc = (
                "The essay displays authentic student writing rhythm, idiosyncratic sentence burstiness, "
                "active personal agency, and grounded situational descriptions consistent with human admissions essays."
            )
        elif ai_probability < thresholds["uncertain_max"]:
            category = "Uncertain / Mixed Evidence"
            verdict_code = "UNCERTAIN"
            verdict_color = "#f59e0b"  # Amber
            confidence_level = "Low"
            summary_desc = (
                "The essay contains a balance of human and machine-like signals. It may represent heavily edited student writing, "
                "an articulate human essay with elevated diction, or lightly assisted composition. Definite authorship cannot be reliably asserted."
            )
        elif ai_probability < thresholds["ai_assisted_max"]:
            category = "Likely AI-Assisted / Polished"
            verdict_code = "LIKELY_AI_ASSISTED"
            verdict_color = "#8b5cf6"  # Purple
            confidence_level = "Moderate"
            summary_desc = (
                "The essay exhibits structural markers consistent with human ideas subsequently revised or polished by an LLM, "
                "such as smoothed sentence delta variance and elevated abstract vocabulary density."
            )
        else:
            category = "Likely AI-Generated"
            verdict_code = "LIKELY_AI_GENERATED"
            verdict_color = "#ef4444"  # Coral Red
            confidence_level = "High" if ai_probability > 0.92 else "Moderate"
            summary_desc = (
                "Multiple independent feature families strongly diverge from the human reference distribution, including high Mahalanobis "
                "structural distance, prominent abstract concept buzzwords, formulaic moral conclusions, and uniform token compressibility."
            )

        # 4. Feature Contributions Breakdown
        contributions = []
        for feat, x_val, s_val, c_val in zip(feature_cols, x_raw, x_scaled, coefs):
            impact = float(s_val * c_val)
            ref_stat = self.artifact.human_reference_stats.get(feat, {"mean": 0.0, "std": 1.0})
            contributions.append({
                "feature": feat,
                "observed_value": round(float(x_val), 4),
                "reference_mean": round(float(ref_stat["mean"]), 4),
                "impact_score": round(impact, 4),
                "direction": "AI-skewed" if impact > 0 else "Human-skewed",
            })

        contributions.sort(key=lambda x: abs(x["impact_score"]), reverse=True)
        top_ai_evidence = [c for c in contributions if c["direction"] == "AI-skewed"][:5]
        top_human_evidence = [c for c in contributions if c["direction"] == "Human-skewed"][:5]

        # 5. Sentence-Level Evidence Highlighting
        highlighted_spans = self.evidence_engine.analyze_sentences(raw_text, segmentation)
        flagged_count = sum(1 for s in highlighted_spans if s.overall_severity in ("high", "medium"))

        return {
            "status": "SUCCESS",
            "model_version": self.artifact.model_version,
            "metadata": {
                "word_count": len(words),
                "char_count": len(raw_text),
                "paragraph_count": segmentation.total_paragraphs,
                "sentence_count": segmentation.total_sentences,
                "flagged_sentence_count": flagged_count,
            },
            "assessment": {
                "category": category,
                "verdict_code": verdict_code,
                "verdict_color": verdict_color,
                "confidence_level": confidence_level,
                "calibrated_ai_probability": round(ai_probability, 4),
                "logit_score": round(logit, 4),
                "summary": summary_desc,
            },
            "evidence": {
                "top_ai_evidence": top_ai_evidence,
                "top_human_evidence": top_human_evidence,
                "all_feature_measurements": {k: round(float(v), 4) for k, v in all_features.items()},
            },
            "highlighted_spans": [
                {
                    "span_id": span.span_id,
                    "sentence_idx": span.sentence_idx,
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "text": span.text,
                    "overall_severity": span.overall_severity,
                    "ai_evidence_count": span.ai_evidence_count,
                    "evidence_items": span.evidence_items,
                }
                for span in highlighted_spans
            ],
            "disclaimers": {
                "probabilistic_warning": (
                    "This analysis is based on statistical and linguistic evidence compared against a verified human reference corpus. "
                    "Statistical indicators do NOT constitute absolute proof of machine authorship. Results should be reviewed holistically by human readers."
                ),
                "esl_fairness_notice": (
                    "This system does not infer or evaluate applicant demographic background or English proficiency. Highly articulate or non-native "
                    "writing styles may exhibit idiosyncratic feature profiles."
                ),
                "no_llm_judge_guarantee": "All scores and evidence are computed strictly using deterministic feature extractors and calibrated statistical models. No LLM was prompted to judge this essay."
            }
        }

    def _empty_response(self, msg: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "error_type": "EMPTY_INPUT",
            "message": msg,
        }

    def _short_text_response(self, raw_text: str, word_count: int) -> Dict[str, Any]:
        return {
            "status": "INSUFFICIENT_LENGTH",
            "message": f"Essay contains only {word_count} words. Admissions stylometric and narrative discourse features require at least 20 words for meaningful measurement.",
            "raw_text": raw_text,
            "word_count": word_count,
        }
