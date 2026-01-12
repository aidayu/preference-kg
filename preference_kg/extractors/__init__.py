"""Multi-step preference extraction module.

This module provides a 3-step pipeline for extracting user preferences from dialogues:
1. EntityExtractor: Extract preference-target entities from dialogue
2. AxisClassifier: Classify each entity into preference axes
3. PreferenceBuilder: Build complete preference objects with details
"""

from .entity_extractor import EntityExtractor
from .axis_classifier import AxisClassifier
from .preference_builder import PreferenceBuilder
from .pipeline import MultiStepPreferenceExtractor

__all__ = [
    "EntityExtractor",
    "AxisClassifier",
    "PreferenceBuilder",
    "MultiStepPreferenceExtractor",
]
