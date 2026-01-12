"""Step 1: Entity Extractor - Extract preference-target entities from dialogue."""

import os

from .base import BaseExtractor

PROMPT_TEMPLATE = """# Role
Extract entities that are targets of user preferences from the dialogue.

# Rules
1. Extract ONLY entities that express explicit preference (liking, wanting, needing)
2. Keep intrinsic attributes ("spicy food", "classical music")
3. Drop extrinsic attributes (price, quantity)
4. Output normalized noun phrases

# Exclusion
- Pure facts without evaluation ("I smoke", "I use Mac")
- Procedural necessities ("Need to login")
- General plans without emotion

# Inclusion
- Evaluated facts ("I *love* coffee")
- Volitions ("I want to *quit* smoking")
- Leisure plans with desire ("I plan to play tennis")
"""

SCHEMA = {
    "name": "entity_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "entity": {
                            "type": "string",
                            "description": "Normalized noun phrase (entity name)",
                        },
                        "original_mention": {
                            "type": "string",
                            "description": "Original text from dialogue",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this entity expresses preference",
                        },
                    },
                    "required": ["entity", "original_mention", "reasoning"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
}


class EntityExtractor(BaseExtractor):
    """Step 1: Extract preference-target entities from dialogue."""

    def __init__(self, model: str = "gpt-4o-mini", few_shot_examples: str = ""):
        super().__init__(model)
        self.few_shot_examples = few_shot_examples

    def extract(self, dialogue: str) -> list[dict]:
        """
        Extract entities from dialogue.

        Args:
            dialogue: The dialogue text

        Returns:
            List of entity dicts with keys: entity, original_mention, reasoning
        """
        system_prompt = PROMPT_TEMPLATE
        if self.few_shot_examples:
            system_prompt += f"\n\n# Examples\n{self.few_shot_examples}"

        user_prompt = f"Extract preference-target entities from this dialogue:\n\n{dialogue}"

        result = self._call_llm(system_prompt, user_prompt, SCHEMA)

        if "error" in result:
            return []

        return result.get("entities", [])
