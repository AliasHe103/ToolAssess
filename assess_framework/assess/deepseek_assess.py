import json
import os
from openai import OpenAI

from config import settings

key = os.environ.get("DEEP_SEEK_API_KEY")
if key is None:
    raise ValueError("DEEP_SEEK_API_KEY is not set!")

models_mapping = {
    "deepseek-chat": "DeepSeek-V3",
    "deepseek-reasoner": "DeepSeek-R1"
}

system_prompt = """
You are tasked with helping the User with his questions. 
For each user query, you are provided with a toolset containing tool names and descriptions about them.
You should determine if a tool is needed for the task.

## Response Format
Your response **must** strictly follow one of these three formats:
1. "requires tool: [Tool Name]." → If the task **must and can** be completed with a tool, **select only one tool** from the provided list.
2. "no tool." → If the task can be **completed without any tool**.
3. "cannot be completed." → If the task is **impossible even with the provided tools**.

## Rules
1. If multiple tools seem relevant, **only select the most suitable one**.
2. **Do not** include any extra explanation or reasoning.
3. Your response **must** strictly follow the formats above **without any additional text**.
"""

sample_file = settings.SINGLE_TASK_DATA_PATH
output_path = settings.SINGLE_TASK_OUTPUT_PATH

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(sample_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

client = OpenAI(api_key=key, base_url="https://api.deepseek.com")

results = {}
def assess_on_deepseek():
    response = ""
    for op_id, sample in samples.items():
        query = sample["query"]
        tools = json.dumps(sample["tools"], indent=2)

        print(f"Running test on {op_id}.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nAvailable Tools: {tools}"}
        ]

        try:
            completion = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,  # 每次请求都是新的，不受历史影响
                temperature=1.0,
            )

            response = completion.choices[0].message.content.strip()
            if "requires tool" in response:
                response_type = "requires tool"
            elif "no tool" in response:
                response_type = "no tool"
            elif "cannot be completed" in response:
                response_type = "cannot be completed"
            else:
                raise ValueError("Invalid response format.")

            results[op_id] = {
                "type": response_type,
                "response": response
            }

        except Exception as e:
            print(f"Error on {op_id}: {e}")
            results[op_id] = {
                "type": "error",
                "response": response
            }
            continue

assess_on_deepseek()
output_file = os.path.join(output_path, "deepseek.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")