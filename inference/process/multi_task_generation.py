import os, json
from openai import OpenAI
from config import settings

if os.environ.get("OPENAI_API_KEY") is None:
    raise ValueError("OPENAI_API_KEY is not set!")

# Load tool descriptions
with open(settings.SUMMARIZED_TOOLS_PATH, 'r', encoding='utf-8') as jf:
    data = json.load(jf)
    base_toolset = json.dumps(data, ensure_ascii=False)

# Multi-task prompt
multi_task_prompt = '''
You are tasked with generating evaluation data for testing the multi-task handling capabilities of Large Language Models. Each task in your generated dataset should consist of multiple user queries that are logically connected but require independent resolutions. The dataset should be formatted as a JSON object, containing:

- **A realistic user scenario** consisting of 2 to 4 interrelated tasks.
- **A set of 3-5 tools** with strictly defined capabilities.

The possible task outcomes for each sub-task should fall into three categories:
1. **Task requires a tool(s)** - The query can only be solved with the provided tools. **Only one tool should be able to complete each sub-task, while the others must be completely unrelated. Return the name of the tool that can solve the sub-task.**
2. **Task does not require a tool** - The query can be answered based on general knowledge without using any tool.
3. **Task cannot be completed, even with tools** - The query cannot be solved, even if the provided tools are used.

Ensure:
- **Generate exactly {num_scenarios} multi-task scenarios.**
- **Each scenario should contain 2-4 logically related tasks.**
- **Each scenario should have 3-5 tools assigned.**
- **In "requires tool" cases, exactly one tool should be relevant per sub-task. Other tools must be unrelated.**
- **At least 25% of the scenarios should have exactly 5 tools.**
- **Tasks requiring basic reasoning or common knowledge (e.g., simple math, general facts) should be classified as "no tool".**
- **"Cannot be completed" cases should be tasks that remain impossible, even with the provided tools.**

Format:

{{
    "scenario1": {{
        "tasks": {{
            "task1": {{
                "query": "[User Request 1]",
                "result": "[Expected Result: requires tool / no tool / cannot be completed]",
                "solving_tool": "[Tool Name (if task requires a tool)]"
            }},
            "task2": {{
                "query": "[User Request 2]",
                "result": "[Expected Result: requires tool / no tool / cannot be completed]",
                "solving_tool": "[Tool Name (if task requires a tool)]"
            }}
        }},
        "tools": {{
            "[Tool Name1]": "[Tool Description1]",
            "[Tool Name2]": "[Tool Description2]",
            "[Tool Name3]": "[Tool Description3]",
            "[Tool Name4]": "[Tool Description4]",
            "[Tool Name5]": "[Tool Description5]"
        }}
    }}
}}
'''.format(num_scenarios=settings.MULTI_TASK_DATA_SIZE)

print("Start generating multi-task dataset.")
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": multi_task_prompt
        },
        {
            "role": "user",
            "content": base_toolset
        }
    ]
)

results = completion.choices[0].message.content
print(results)

with open(settings.MULTI_TASK_DATA_PATH, 'w', encoding='utf-8') as jf:
    json.dump(results, jf, ensure_ascii=False, indent=4)
