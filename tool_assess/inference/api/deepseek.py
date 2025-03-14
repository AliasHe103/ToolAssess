import os
from openai import OpenAI

key = os.environ.get("DEEP_SEEK_API_KEY")
if key is None:
    raise ValueError("DEEP_SEEK_API_KEY is not set!")

models_mapping = {
    "deepseek-chat": "DeepSeek-V3",
    "deepseek-reasoner": "DeepSeek-R1"
}

tool_description_prompt = '''
Please provide a brief introduction about a tool designed for LLMs to use, namely [Tool Name]: [Tool Description]. Specifically, explain how it serves the needs of a large language model (LLM) and describe its main functions or applications. For example, for the tool "Amadeus Toolkit", your response could be:
"Amadeus Toolkit": "The Amadeus Toolkit integrates LangChain with the Amadeus travel APIs, allowing LLMs to assist with travel-related tasks such as searching for flights and booking trips. LLMs can leverage this toolkit to help users plan travel, check flight availability, compare prices, and suggest optimal travel options based on user preferences, improving the travel booking experience with AI-driven recommendations and automation."
Now I will give you some relevant tools.
'''

client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {
            "role": "system",
            "content": tool_description_prompt
        },
        {
            "role": "user",
            "content": "Gmail Toolkit"
        }
    ],
    temperature=1.0,
)

print(completion.choices[0].message.content)