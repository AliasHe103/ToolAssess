"""
This is an example for using qwen api.
"""
import os
from openai import OpenAI

from tool_assess.agents.qwen_agent import QwenAgent

key = os.environ.get("DASHSCOPE_API_KEY")
if key is None:
    raise ValueError("DASHSCOPE_API_KEY is not set!")

client = OpenAI(api_key=key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
qwen = QwenAgent(name="qwen-max")
response = qwen.test()

print(response)