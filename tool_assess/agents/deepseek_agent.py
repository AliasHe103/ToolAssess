import os

from tool_assess.agents.compatible_agent import CompatibleAgent


class DeepseekAgent(CompatibleAgent):
    def __init__(self, name):
        key = os.environ.get("DEEPSEEK_API_KEY")
        url = "https://api.deepseek.com"
        super().__init__(key, url, name)

    def predict(self, messages):
        return super().predict(messages)

    def test(self):
        return super().test()