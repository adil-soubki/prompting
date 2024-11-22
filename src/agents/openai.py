# -*- coding: utf-8 -*
from typing import Any

import openai

from ..agents.core import Agent
from ..core import keychain


class GPTAgent(Agent):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI(api_key=keychain.get("IACS"))

    def generation_hook(self, prompt: str) -> openai.types.chat.ChatCompletion:
        return self.client.chat.completions.create(
            model=self.model_name, messages=[dict(role="user", content=prompt)]
        )

    def post_generation_hook(self, output: Any) -> dict[str, Any]:
        return {
            "model_name": output.model,
            "generation": output.choices[0].message.content,
        }
