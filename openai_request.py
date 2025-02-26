from openai import OpenAI
client = OpenAI()

model_mapping = [
    "gpt-4",
    "gpt-3.5-turbo.5",
]

tool_description_prompt = '''
Please provide a brief introduction about a tool designed for LLMs to use, namely [Tool Name]: [Tool Description]. Specifically, explain how it serves the needs of a large language model (LLM) and describe its main functions or applications. For example, for the tool "Amadeus Toolkit", your response could be:
"Amadeus Toolkit": "The Amadeus Toolkit integrates LangChain with the Amadeus travel APIs, allowing LLMs to assist with travel-related tasks such as searching for flights and booking trips. LLMs can leverage this toolkit to help users plan travel, check flight availability, compare prices, and suggest optimal travel options based on user preferences, improving the travel booking experience with AI-driven recommendations and automation."
Now I will give you some relevant tools.
'''
completion = client.chat.completions.create(
    model="gpt-4o-mini",
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
    max_tokens=256,
    temperature=0.7,
)

print(completion.choices[0].message.content)