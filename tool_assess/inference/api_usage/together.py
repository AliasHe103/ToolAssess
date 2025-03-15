"""
This is an example for using together api.
"""
import os

from tool_assess.agents.together_agent import TogetherAgent

key = os.environ.get("TOGETHER_API_KEY")
if key is None:
    raise ValueError("TOGETHER_API_KEY is not set!")

together = TogetherAgent(name="llama-3.3-70B")
response = together.test()

print(response)