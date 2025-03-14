import os
from openai import OpenAI

key = os.environ.get("TOGETHER_API_KEY")
if key is None:
    raise ValueError("TOGETHER_API_KEY is not set!")

client = OpenAI(api_key=key, base_url="https://api.together.xyz/v1",)
completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Hello! Describe yourself."
        }
    ],
)

print(completion.choices[0].message.content)