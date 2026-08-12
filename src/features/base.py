"""Abstract base class and metadata schema for all feature extractors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class FeatureMetadata:
    name: str
    group: str  # 'surface', 'discourse', 'distributional'
    definition: str
    unit_range: str
    interpretation: str
    limitations: str


class BaseFeatureExtractor(ABC):
    """Base class for modular feature extraction."""

    @abstractmethod
    def extract_features(self, text: str, segmentation=None) -> Dict[str, float]:
        """Extracts a dictionary of numeric feature values from text."""
        pass

    @abstractmethod
    def get_metadata(self) -> List[FeatureMetadata]:
        """Returns metadata documentation for all features produced by this extractor."""
        pass
