"""Step 3: Preference Builder - Build complete preference objects."""

from .base import BaseExtractor

PROMPT_TEMPLATE = """# Role
Build a complete preference object for the given entity and axis based on the dialogue.

# Field Definitions
## polarity
- **positive**: User likes/wants/needs the entity
- **negative**: User dislikes/doesn't want/doesn't need the entity
- **neutral**: Ambiguous or balanced sentiment

## intensity  
- **high**: Strong expressions (love, hate, must, definitely)
- **medium**: Moderate expressions (like, want, prefer)
- **low**: Weak expressions (somewhat, a bit, maybe)

## context_tags (select ALL that apply, or empty list if none)
Categories:
- activity-*: Action during which entity is consumed
- social-*: Solo or group context
- condition-*: Physical/mental state
- temporal-*: Time of day, season, weekday/weekend
- location-*: Where the preference applies

# Rules
1. Only include context if EXPLICITLY mentioned in dialogue
2. If entity == activity, don't add activity as context
3. Be conservative - when in doubt, leave context_tags empty
"""

CONTEXT_TAG_ENUM = [
    "activity-working_studying",
    "activity-eating",
    "activity-drinking",
    "activity-driving",
    "activity-relaxing",
    "activity-shopping",
    "social-solo",
    "social-group",
    "condition-tired",
    "condition-hungry_thirsty",
    "condition-sick_pain",
    "condition-stressed",
    "condition-busy",
    "condition-bored",
    "temporal-morning",
    "temporal-noon",
    "temporal-afternoon",
    "temporal-night",
    "temporal-spring",
    "temporal-summer",
    "temporal-fall",
    "temporal-winter",
    "temporal-weekday",
    "temporal-weekend",
    "temporal-holiday",
    "location-home",
    "location-workplace",
    "location-school",
    "location-restaurant_bar",
    "location-shop",
    "location-travel",
    "location-outdoor",
]

SCHEMA = {
    "name": "preference_building",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entity": {"type": "string"},
            "combined_axis": {"type": "string"},
            "original_mention": {
                "type": "string",
                "description": "Original text from dialogue expressing this preference",
            },
            "polarity": {
                "type": "string",
                "enum": ["positive", "negative", "neutral"],
            },
            "intensity": {
                "type": "string",
                "enum": ["high", "medium", "low"],
            },
            "context_tags": {
                "type": "array",
                "items": {"type": "string", "enum": CONTEXT_TAG_ENUM},
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation for polarity, intensity, and context choices",
            },
        },
        "required": [
            "entity",
            "combined_axis",
            "original_mention",
            "polarity",
            "intensity",
            "context_tags",
            "reasoning",
        ],
        "additionalProperties": False,
    },
}


class PreferenceBuilder(BaseExtractor):
    """Step 3: Build complete preference objects."""

    def __init__(self, model: str = "gpt-4o-mini", few_shot_examples: str = ""):
        super().__init__(model)
        self.few_shot_examples = few_shot_examples

    def extract(self, dialogue: str, entity: str, axis: str) -> dict:
        """
        Build complete preference object.

        Args:
            dialogue: The dialogue text
            entity: The entity
            axis: The combined_axis (e.g., "liking__aesthetic_sensory")

        Returns:
            Complete preference dict
        """
        system_prompt = PROMPT_TEMPLATE
        if self.few_shot_examples:
            system_prompt += f"\n\n# Examples\n{self.few_shot_examples}"

        user_prompt = (
            f"Dialogue:\n{dialogue}\n\n"
            f"Entity: {entity}\n"
            f"Axis: {axis}\n\n"
            f"Build the complete preference object."
        )

        result = self._call_llm(system_prompt, user_prompt, SCHEMA)

        if "error" in result:
            return {
                "entity": entity,
                "combined_axis": axis,
                "polarity": "neutral",
                "intensity": "medium",
                "context_tags": [],
                "original_mention": "",
                "reasoning": f"Error: {result.get('error')}",
            }

        return result
