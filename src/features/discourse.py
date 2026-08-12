"""Admissions essay discourse and narrative architecture feature extractor."""
import re
from typing import Dict, List, Optional
import numpy as np
import nltk

from src.features.base import BaseFeatureExtractor, FeatureMetadata
from src.segmentation.segmenter import HierarchicalSegmenter, EssaySegmentation


class DiscourseFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts admissions-specific discourse, agency, reflection, causal, and concrete/abstract features.
    """

    # Lexicons for Admissions Discourse Analysis
    ACTION_VERBS = {
        "built", "created", "founded", "organized", "designed", "coded", "launched",
        "decided", "chose", "led", "repaired", "wrote", "conducted", "researched",
        "initiated", "engineered", "assembled", "campaigned", "resolved", "struggled",
        "confronted", "negotiated", "volunteered", "tutored", "gathered", "analyzed"
    }

    PASSIVE_STATE_VERBS = {
        "felt", "seemed", "was told", "were told", "found myself", "made me", "appeared",
        "wished", "hoped", "wondered", "endured", "received", "was given", "witnessed"
    }

    REFLECTION_MARKERS = [
        r"\brealiz(?:ed|ing|e)\b",
        r"\bdiscover(?:ed|ing|e)\b",
        r"\bunderst(?:ood|and|anding)\b",
        r"\blearn(?:ed|t|ing)\b",
        r"\btaught me\b",
        r"\bperspective (?:shifted|changed|broadened)\b",
        r"\bconclud(?:ed|ing|e)\b",
        r"\bepiphany\b",
        r"\billuminat(?:ed|ing|e)\b",
        r"\binsight\b",
        r"\bcame to know\b",
        r"\bopened my eyes\b",
    ]

    FORMULAIC_MORAL_PATTERNS = [
        r"\bin conclusion\b",
        r"\bthis (?:experience )?taught me (?:that|how)\b",
        r"\bthe lesson (?:i|we) learned\b",
        r"\blooking back\b",
        r"\bhas shaped (?:who|what) i am\b",
        r"\ba testament to\b",
        r"\btestament of\b",
        r"\bnot only did .+, but (?:also)?\b",
        r"\bultimately,? (?:this|i)\b",
        r"\bforever changed\b",
    ]

    CAUSAL_CONNECTORS = [
        r"\bbecause\b", r"\btherefore\b", r"\bconsequently\b", r"\bas a result\b",
        r"\bthus\b", r"\bhence\b", r"\bdue to\b", r"\bled to\b", r"\bfueled by\b",
        r"\bwhich caused\b", r"\bsince\b"
    ]

    TEMPORAL_MARKERS = [
        r"\binitially\b", r"\bsuddenly\b", r"\bmeanwhile\b", r"\byears later\b",
        r"\bat that moment\b", r"\bsubsequently\b", r"\beventually\b", r"\bbefore long\b",
        r"\bthe next morning\b", r"\bover time\b", r"\bafter weeks\b"
    ]

    ABSTRACT_VOCABULARY = {
        "paradigm", "quintessential", "foster", "fostering", "inextricably", "dichotomy",
        "transformative", "embodiment", "myriad", "tapestry", "beacon", "catalyst",
        "profound", "pivotal", "delve", "nuance", "testament", "resonate", "resonates",
        "interconnectedness", "holistic", "multifaceted", "unwavering", "culmination",
        "testament", "epitome", "embark", "embarking", "testament"
    }

    CONCRETE_SENSORY_WORDS = {
        "smell", "scent", "sound", "screamed", "whispered", "wooden", "metal", "grease",
        "soldering", "kitchen", "street", "table", "clock", "sweat", "chalk", "brass",
        "rain", "dirt", "guitar", "wire", "microscope", "hospital", "notebook", "stumbled",
        "hands", "fingers", "cold", "hot", "bruise", "bandage", "canvas", "paint"
    }

    def __init__(self, segmenter: Optional[HierarchicalSegmenter] = None):
        self.segmenter = segmenter or HierarchicalSegmenter()

    def extract_features(self, text: str, segmentation: Optional[EssaySegmentation] = None) -> Dict[str, float]:
        if segmentation is None:
            segmentation = self.segmenter.segment(text)

        tokens = re.findall(r"\b\w+(?:'\w+)?\b", text.lower())
        total_tokens = max(1, len(tokens))
        text_lower = text.lower()

        # 1. Personal Agency: First Person Active Action vs Passive State
        agency_action_count = 0
        agency_passive_count = 0
        
        for i, tok in enumerate(tokens):
            if tok in ("i", "my", "me", "we", "our"):
                window = tokens[max(0, i):min(len(tokens), i + 4)]
                if any(w in self.ACTION_VERBS for w in window):
                    agency_action_count += 1
                if any(w in self.PASSIVE_STATE_VERBS for w in window):
                    agency_passive_count += 1

        agency_action_density = (agency_action_count / total_tokens) * 100
        agency_passive_density = (agency_passive_count / total_tokens) * 100
        agency_ratio = agency_action_density / max(1e-5, agency_action_density + agency_passive_density)

        # 2. Reflection Markers and Positional Distribution
        reflection_matches = []
        for pat in self.REFLECTION_MARKERS:
            for m in re.finditer(pat, text_lower):
                reflection_matches.append(m.start() / max(1, len(text_lower)))

        total_reflections = len(reflection_matches)
        reflection_density = (total_reflections / total_tokens) * 100

        if total_reflections > 0:
            intro_ref = sum(1 for pos in reflection_matches if pos < 0.33) / total_reflections
            body_ref = sum(1 for pos in reflection_matches if 0.33 <= pos < 0.67) / total_reflections
            conclusion_ref = sum(1 for pos in reflection_matches if pos >= 0.67) / total_reflections
            mean_reflection_pos = float(np.mean(reflection_matches))
        else:
            intro_ref = 0.0
            body_ref = 0.0
            conclusion_ref = 0.0
            mean_reflection_pos = 0.5

        # 3. Formulaic Moral / Lesson Summary Explicitness
        formulaic_count = 0
        for pat in self.FORMULAIC_MORAL_PATTERNS:
            formulaic_count += len(re.findall(pat, text_lower))
        formulaic_moral_density = (formulaic_count / total_tokens) * 100

        # 4. Causal & Temporal Cohesion
        causal_count = sum(len(re.findall(pat, text_lower)) for pat in self.CAUSAL_CONNECTORS)
        causal_density = (causal_count / total_tokens) * 100

        temporal_count = sum(len(re.findall(pat, text_lower)) for pat in self.TEMPORAL_MARKERS)
        temporal_density = (temporal_count / total_tokens) * 100

        # 5. Concrete Sensory vs Abstract Vocabulary Density
        abstract_count = sum(1 for t in tokens if t in self.ABSTRACT_VOCABULARY)
        abstract_density = (abstract_count / total_tokens) * 100

        concrete_count = sum(1 for t in tokens if t in self.CONCRETE_SENSORY_WORDS)
        # Add numbers/digits count as concrete anchoring
        digit_count = len(re.findall(r"\b\d+\b", text))
        concrete_count += digit_count
        concrete_density = (concrete_count / total_tokens) * 100

        concrete_abstract_ratio = concrete_density / max(1e-5, concrete_density + abstract_density)

        return {
            "discourse_agency_action_density": float(agency_action_density),
            "discourse_agency_passive_density": float(agency_passive_density),
            "discourse_agency_ratio": float(agency_ratio),
            "discourse_reflection_density": float(reflection_density),
            "discourse_reflection_intro_ratio": float(intro_ref),
            "discourse_reflection_body_ratio": float(body_ref),
            "discourse_reflection_conclusion_ratio": float(conclusion_ref),
            "discourse_reflection_mean_pos": float(mean_reflection_pos),
            "discourse_formulaic_moral_density": float(formulaic_moral_density),
            "discourse_causal_density": float(causal_density),
            "discourse_temporal_density": float(temporal_density),
            "discourse_abstract_vocab_density": float(abstract_density),
            "discourse_concrete_density": float(concrete_density),
            "discourse_concrete_abstract_ratio": float(concrete_abstract_ratio),
        }

    def get_metadata(self) -> List[FeatureMetadata]:
        return [
            FeatureMetadata("discourse_agency_action_density", "discourse", "First-person active decision/action verb density", "[0.0, 10.0]", "Active personal agency in storytelling", "Lexicon based"),
            FeatureMetadata("discourse_agency_passive_density", "discourse", "First-person passive feeling/experiential verb density", "[0.0, 10.0]", "Passive or state-based framing", "Lexicon based"),
            FeatureMetadata("discourse_agency_ratio", "discourse", "Ratio of active agency to total agency markers", "[0.0, 1.0]", "Direct protagonist agency vs passive observation", "Undefined if no agency markers"),
            FeatureMetadata("discourse_reflection_density", "discourse", "Epistemic/realization reflection markers per 100 words", "[0.0, 10.0]", "Frequency of self-reflection moments", "Pattern matching"),
            FeatureMetadata("discourse_reflection_intro_ratio", "discourse", "Fraction of reflections occurring in first 33% of essay", "[0.0, 1.0]", "Premature reflection structure", "Normalized position"),
            FeatureMetadata("discourse_reflection_body_ratio", "discourse", "Fraction of reflections occurring in middle 33% of essay", "[0.0, 1.0]", "Integrated narrative reflection", "Normalized position"),
            FeatureMetadata("discourse_reflection_conclusion_ratio", "discourse", "Fraction of reflections occurring in final 33% of essay", "[0.0, 1.0]", "End-loaded resolution structure", "Normalized position"),
            FeatureMetadata("discourse_reflection_mean_pos", "discourse", "Mean normalized position of reflection markers", "[0.0, 1.0]", "Chronological center of gravity of reflection", "Default 0.5 if none"),
            FeatureMetadata("discourse_formulaic_moral_density", "discourse", "Formulaic moral/takeaway phrases per 100 words", "[0.0, 5.0]", "Explicitness of stated life lessons", "Phrase pattern based"),
            FeatureMetadata("discourse_causal_density", "discourse", "Causal transition markers per 100 words", "[0.0, 10.0]", "Explanatory causal structure density", "Regex marker based"),
            FeatureMetadata("discourse_temporal_density", "discourse", "Narrative time transition markers per 100 words", "[0.0, 10.0]", "Chronological progression density", "Regex marker based"),
            FeatureMetadata("discourse_abstract_vocab_density", "discourse", "Frequency of abstract AI-prone vocabulary", "[0.0, 10.0]", "Conceptual abstractness vs grounding", "Fixed curated lexicon"),
            FeatureMetadata("discourse_concrete_density", "discourse", "Sensory, physical, and numerical grounding per 100 words", "[0.0, 10.0]", "Concrete situational grounding", "Lexicon and numeric based"),
            FeatureMetadata("discourse_concrete_abstract_ratio", "discourse", "Ratio of concrete grounding to total descriptive vocabulary", "[0.0, 1.0]", "Grounding vs abstraction balance", "Lexicon based"),
        ]
