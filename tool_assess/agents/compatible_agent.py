from openai import OpenAI

from tool_assess.agents.agent import Agent


class CompatibleAgent(Agent):
    def __init__(self, key, url, name):
        super().__init__(name)
        self.client = OpenAI(
            api_key=key,
            base_url=url
        )
        self.name = name
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
