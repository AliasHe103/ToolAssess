import os, json
from openai import OpenAI

from config import settings

task_prompt = '''
You are tasked with summarizing tool descriptions. For each tool, you are provided with a long description that includes its name, features, and use cases. Your goal is to condense the original description into a concise, clear, and machine-readable summary that focuses on the core functionality of the tool. The summary should highlight what the tool does, its main purpose, and how it helps a Large Language Model (LLM) deal with a user's query. Avoid including unnecessary details or redundant information.

Please format the output as follows:
Tool Name: Core Functionality (Concise summary focusing on the main use case and how the tool helps the LLM with a user's query).

Example:
[Input] "Gmail Toolkit": "Gmail Toolkit interacts with the Gmail API to read messages, draft and send emails, and perform other email-related tasks. For LLMs, this toolkit can be used to automate email management, classify emails, or generate smart replies based on the content of incoming messages. LLMs can leverage the Gmail Toolkit to streamline tasks like responding to emails, sorting messages into folders, and generating summaries or follow-ups, making email communication more efficient and personalized."
[Your Output] "Gmail Toolkit": "Interacts with the Gmail API to read, send, and manage emails, automate tasks like sorting and classifying emails, and generate smart replies, helping the LLM efficiently respond to email-related user queries."

Instructions:
1.Focus on the main purpose of the tool and what it does.
2.Mention how the tool helps the LLM handle a user query.
3.Remove unnecessary explanations, extraneous features, and complex wording.
4.Make sure the summary is clear, focused, and easily understandable by machines.
5.Repeat this process for each tool description provided.
6.The actual input is a JSON serialized string. You need to compose all the outputs into a corresponding JSON serialized string and return it.
'''

models_mapping = {
    "deepseek-chat": "DeepSeek-V3",
    "deepseek-reasoner": "DeepSeek-R1"
}

with open(settings.ORG_TOOLS_DATA_PATH, 'r', encoding='utf-8') as jf:
    data = json.load(jf)
    inputs = json.dumps(data, ensure_ascii=False)

print("Start generating.")
client = OpenAI(api_key=settings.deepseek_api_key, base_url="https://api.deepseek.com")
completion = client.chat.completions.create(
    model="deepseek-reasoner",
    messages=[
        {
            "role": "system",
            "content": task_prompt
        },
        {
            "role": "user",
            "content": inputs
        }
    ]
)

results = completion.choices[0].message.content
print(results)

store_path = settings.SUMMARIZED_TOOLS_PATH.split('.')[0] + "_by_deepseek" + ".json"
with open(store_path, 'w', encoding='utf-8') as jf:
    json.dump(results, jf, ensure_ascii=False, indent=4)

