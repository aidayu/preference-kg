"""Base extractor class for multi-step preference extraction."""

import json
import os

import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


class BaseExtractor:
    """Base class for all extraction steps."""

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = openai.OpenAI()

    def _call_llm(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        """Call LLM with structured output."""
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_schema", "json_schema": schema},
            )
            return json.loads(completion.choices[0].message.content)
        except Exception as e:
            print(f"LLM call error: {e}")
            return {"error": str(e)}

    def extract(self, **kwargs) -> dict:
        """Override in subclasses."""
        raise NotImplementedError
