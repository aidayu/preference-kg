"""Step 2: Axis Classifier - Classify entity into preference axes."""

from .base import BaseExtractor

PROMPT_TEMPLATE = """# Role
Determine which preference axis/axes apply to the given entity based on the dialogue context.

# Axis Definitions
## Liking (Past/Present Experience)
- **aesthetic_sensory**: Five senses, aesthetics, physical comfort
- **stimulation**: Excitement, immersion, stress relief
- **identification**: Values, beliefs, identity match
- **general**: Unspecified affection, default positive

## Wanting (Future-oriented)
- **interest**: Curiosity, understanding, analysis (no direct action)
- **goal**: Acquisition, achievement, active intent with action

## Need (Deficit-based)
- **functional**: Specs, efficiency, cost/budget, utility
- **personal**: Constraints (health/money), self-maintenance

# Rules
1. An entity can have MULTIPLE axes if the dialogue supports it
2. Choose axes based on HOW the entity is mentioned, not what it is
3. Look for evaluative expressions (adjectives, emotions) for Liking
4. Look for future-oriented expressions for Wanting
5. Look for necessity/requirement expressions for Need
"""

SCHEMA = {
    "name": "axis_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "axes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "combined_axis": {
                            "type": "string",
                            "enum": [
                                "liking__aesthetic_sensory",
                                "liking__stimulation",
                                "liking__identification",
                                "liking__general",
                                "wanting__interest",
                                "wanting__goal",
                                "need__functional",
                                "need__personal",
                            ],
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Why this axis applies, quoting evidence from dialogue",
                        },
                    },
                    "required": ["combined_axis", "reasoning"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entity", "axes"],
        "additionalProperties": False,
    },
}


class AxisClassifier(BaseExtractor):
    """Step 2: Classify entity into preference axes."""

    def __init__(self, model: str = "gpt-4o-mini", few_shot_examples: str = ""):
        super().__init__(model)
        self.few_shot_examples = few_shot_examples

    def extract(self, dialogue: str, entity: str) -> list[dict]:
        """
        Classify entity into preference axes.

        Args:
            dialogue: The dialogue text
            entity: The entity to classify

        Returns:
            List of axis dicts with keys: combined_axis, reasoning
        """
        system_prompt = PROMPT_TEMPLATE
        if self.few_shot_examples:
            system_prompt += f"\n\n# Examples\n{self.few_shot_examples}"

        user_prompt = (
            f"Dialogue:\n{dialogue}\n\n"
            f"Entity: {entity}\n\n"
            f"Determine which preference axis/axes apply to this entity."
        )

        result = self._call_llm(system_prompt, user_prompt, SCHEMA)

        if "error" in result:
            return []

        return result.get("axes", [])
