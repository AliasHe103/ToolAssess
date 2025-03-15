import os

from openai import OpenAI

from tool_assess.agents.compatible_agent import CompatibleAgent
from tool_assess.utils.api_string import to_api_model_string


class TogetherAgent(CompatibleAgent):
    def __init__(self, name):
        key = os.environ.get("TOGETHER_API_KEY")
        url = "https://api.together.xyz/v1"
        super().__init__(key, url, name)
        self.client = OpenAI(
            api_key=key,
            base_url=url
        )
        self.name = to_api_model_string(name)

    def predict(self, messages):
        response = self.client.chat.completions.create(
            model=self.name,
            messages=messages,
            temperature=1.0,
        )
        return response.choices[0].message.content.strip()

    def test(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"}
        ]
        return self.predict(messages)