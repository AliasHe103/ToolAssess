import os

from tool_assess.agents.compatible_agent import CompatibleAgent


class QwenAgent(CompatibleAgent):
    def __init__(self, name):
        key = os.environ.get("DASHSCOPE_API_KEY")
        url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        super().__init__(key, url, name)

    def predict(self, messages):
        return super().predict(messages)

    def test(self):
        return super().test()