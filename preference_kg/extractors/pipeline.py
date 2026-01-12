"""Multi-step preference extraction pipeline."""

import json
import os
from datetime import datetime

from .entity_extractor import EntityExtractor
from .axis_classifier import AxisClassifier
from .preference_builder import PreferenceBuilder


class MultiStepPreferenceExtractor:
    """
    3-step pipeline for preference extraction.

    Step 1: Extract entities from dialogue
    Step 2: Classify each entity into preference axes
    Step 3: Build complete preference objects for each (entity, axis) pair
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        entity_few_shot: str = "",
        axis_few_shot: str = "",
        preference_few_shot: str = "",
        output_dir: str | None = None,
    ):
        self.entity_extractor = EntityExtractor(model, entity_few_shot)
        self.axis_classifier = AxisClassifier(model, axis_few_shot)
        self.preference_builder = PreferenceBuilder(model, preference_few_shot)
        self.output_dir = output_dir
        
        # ステップごとの出力を保存するためのディレクトリを作成
        if self.output_dir:
            os.makedirs(os.path.join(self.output_dir, "step1_entities"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "step2_axes"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "step3_preferences"), exist_ok=True)

    def _save_step_output(self, step_name: str, dialogue_id: int, data: dict | list):
        """各ステップの出力をJSONファイルに保存"""
        if not self.output_dir:
            return
        
        filepath = os.path.join(
            self.output_dir, 
            step_name, 
            f"dialogue_{dialogue_id}.json"
        )
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def extract(self, dialogue: str, dialogue_id: int = 0) -> dict:
        """
        Extract preferences from dialogue using 3-step pipeline.

        Args:
            dialogue: The dialogue text
            dialogue_id: Optional dialogue ID

        Returns:
            Dict with dialogue_id, user_id, and list of preferences
        """
        preferences = []
        step2_results = []
        step3_results = []

        # Step 1: Extract entities
        entities = self.entity_extractor.extract(dialogue)
        self._save_step_output("step1_entities", dialogue_id, {
            "dialogue_id": dialogue_id,
            "entities": entities
        })

        for entity_info in entities:
            entity = entity_info.get("entity", "")
            original_mention_from_step1 = entity_info.get("original_mention", "")

            if not entity:
                continue

            # Step 2: Classify axes for this entity
            axes = self.axis_classifier.extract(dialogue, entity)
            step2_results.append({"entity": entity, "axes": axes})

            for axis_info in axes:
                axis = axis_info.get("combined_axis", "")

                if not axis:
                    continue

                # Step 3: Build complete preference
                preference = self.preference_builder.extract(dialogue, entity, axis)
                step3_results.append(preference)

                # Use original_mention from step 1 if step 3 didn't provide one
                if not preference.get("original_mention"):
                    preference["original_mention"] = original_mention_from_step1

                preferences.append(preference)

        # Step 2, 3の結果も保存
        self._save_step_output("step2_axes", dialogue_id, {
            "dialogue_id": dialogue_id,
            "axis_classifications": step2_results
        })
        self._save_step_output("step3_preferences", dialogue_id, {
            "dialogue_id": dialogue_id,
            "preferences": step3_results
        })

        return {
            "dialogue_id": dialogue_id,
            "user_id": "user",
            "preferences": preferences,
        }

    def extract_with_debug(self, dialogue: str, dialogue_id: int = 0) -> dict:
        """
        Extract preferences with intermediate step results for debugging.

        Returns:
            Dict with full debug information for each step
        """
        result = {
            "dialogue_id": dialogue_id,
            "dialogue": dialogue,
            "step1_entities": [],
            "step2_axes": [],
            "step3_preferences": [],
            "final_preferences": [],
        }

        # Step 1
        entities = self.entity_extractor.extract(dialogue)
        result["step1_entities"] = entities
        self._save_step_output("step1_entities", dialogue_id, {
            "dialogue_id": dialogue_id,
            "entities": entities
        })

        for entity_info in entities:
            entity = entity_info.get("entity", "")
            if not entity:
                continue

            # Step 2
            axes = self.axis_classifier.extract(dialogue, entity)
            result["step2_axes"].append({"entity": entity, "axes": axes})

            for axis_info in axes:
                axis = axis_info.get("combined_axis", "")
                if not axis:
                    continue

                # Step 3
                preference = self.preference_builder.extract(dialogue, entity, axis)
                result["step3_preferences"].append(preference)
                result["final_preferences"].append(preference)

        # Step 2, 3の結果も保存
        self._save_step_output("step2_axes", dialogue_id, {
            "dialogue_id": dialogue_id,
            "axis_classifications": result["step2_axes"]
        })
        self._save_step_output("step3_preferences", dialogue_id, {
            "dialogue_id": dialogue_id,
            "preferences": result["step3_preferences"]
        })

        return result

