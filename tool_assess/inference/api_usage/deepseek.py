"""
This is an example for using deepseek api.
"""
import os

from tool_assess.agents.deepseek_agent import DeepseekAgent

key = os.environ.get("DEEP_SEEK_API_KEY")
if key is None:
    raise ValueError("DEEP_SEEK_API_KEY is not set!")

deepseek = DeepseekAgent(name="deepseek-chat")
response = deepseek.test()

print(response)