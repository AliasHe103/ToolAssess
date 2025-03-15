"""
This is an example for using openai api.
"""
import os
from openai import OpenAI

from tool_assess.agents.gpt_agent import GPTAgent

if os.environ.get("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set!")

chatgpt = GPTAgent("gpt-4o-mini")
response = chatgpt.test()

print(response)