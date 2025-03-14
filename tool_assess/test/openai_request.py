import os

from huggingface_hub import InferenceClient

key = os.environ.get("HF_KEY")
if key is None:
    raise ValueError("HF_KEY environment variable not set")
client = InferenceClient(
    "deepseek-ai/DeepSeek-R1",
    token=key,
)

client.text_classification("Today is a great day")