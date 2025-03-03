import os, json
from openai import OpenAI
from config import settings

# Load tool descriptions
with open(settings.SUMMARIZED_TOOLS_PATH, 'r', encoding='utf-8') as jf:
    data = json.load(jf)
    base_toolset = json.dumps(data, ensure_ascii=False)

# Multi-task prompt
multi_task_prompt = '''
You are tasked with generating evaluation data for testing the multi-task handling capabilities of Large Language Models. Each task in your generated dataset should consist of multiple user queries that are logically connected but require independent resolutions. The dataset should be formatted as a JSON object, containing:

- **A realistic user scenario** consisting of 2 to 4 interrelated tasks.
- **A set of 3-5 tools**, ensuring that:
  - **Each tool serves a distinct purpose, with no obvious overlap.**
  - **At least one tool must be completely unrelated to the tasks, serving as a distractor.**
  - **Tools should be selected strategically to test the model's tool selection awareness.**

The possible task outcomes for each sub-task should fall into three categories:
1. **Task requires a tool(s)** - The query can only be solved with the provided tools. **Only one tool should be able to complete each sub-task, while the others must be completely unrelated. Return the name of the tool that can solve the sub-task.**
2. **Task does not require a tool** - The query can be answered based on general knowledge without using any tool. **Only assign "no tool" to tasks that involve common knowledge, basic arithmetic, or general facts that an LLM can answer confidently. Always include "solving_tool": "" explicitly.**
3. **Task cannot be completed, even with tools** - The query cannot be solved, even if the provided tools are used. **This category should only include tasks that are fundamentally impossible, such as predicting future lottery numbers or determining the exact stock market price next year. Always include "solving_tool": "" explicitly.**

Ensure:
- **Generate exactly {num_scenarios} multi-task scenarios.**
- **Each scenario should contain 2-4 logically related tasks.**
- **Each scenario should have 3-5 tools assigned, following these constraints:**
  - **No two tools should have a significant overlap in functionality.**
  - **At least one tool must be irrelevant to the given tasks.**
  - **In "requires tool" cases, exactly one tool should be relevant per sub-task. Other tools must be unrelated.**
- **Tasks requiring basic reasoning or common knowledge (e.g., simple math, general facts) should be classified as "no tool" and explicitly include "solving_tool": ""**
- **"Cannot be completed" cases should be tasks that remain impossible, even with the provided tools. Always include "solving_tool": "" explicitly.**
- **All tasks within a scenario must be logically connected. The relationships should follow at least one of these patterns:**
  1. **Sequential (Progressive) Relationship**: The tasks should be different steps of a larger process.  
     - Example: **"Find a flight to Paris" → "Book a hotel in Paris" → "Check the weather forecast in Paris."**
  2. **Parallel (Categorical) Relationship**: The tasks should share a common characteristic.  
     - Example: **"Calculate 5 factorial" → "Find the square root of 144" → "Convert 25% to a fraction."**
  3. **Domain-Specific Relationship**: The tasks should belong to the same field or industry.  
     - Example: **"Retrieve the latest stock price of Tesla" → "Analyze the electric vehicle stock market trend" → "Find the most traded tech stocks this week."**

Format:

{{
    "scenario1": {{
        "tasks": {{
            "task1": {{
                "query": "[User Request 1]",
                "result": "[Expected Result: requires tool / no tool / cannot be completed]",
                "solving_tool": "[Tool Name (if task requires a tool) or empty string]"
            }},
            "task2": {{
                "query": "[User Request 2]",
                "result": "[Expected Result: requires tool / no tool / cannot be completed]",
                "solving_tool": "[Tool Name (if task requires a tool) or empty string]"
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

Example:

{{
    "scenario1": {{
        "tasks": {{
            "task1": {{
                "query": "Find the latest stock price of Tesla",
                "result": "requires tool",
                "solving_tool": "Yahoo Finance News"
            }},
            "task2": {{
                "query": "Analyze the recent market trend of electric vehicle stocks",
                "result": "requires tool",
                "solving_tool": "Google Trends"
            }},
            "task3": {{
                "query": "Find the most traded tech stocks this week",
                "result": "requires tool",
                "solving_tool": "FinancialDatasets Toolkit"
            }},
            "task4": {{
                "query": "What is 5 multiplied by 8?",
                "result": "no tool",
                "solving_tool": ""
            }},
            "task5": {{
                "query": "What will be the winning lottery numbers for tomorrow?",
                "result": "cannot be completed",
                "solving_tool": ""
            }}
        }},
        "tools": {{
            "Yahoo Finance News": "Retrieves financial news and market analysis.",
            "Google Trends": "Tracks and analyzes trending financial and economic data.",
            "FinancialDatasets Toolkit": "Offers structured financial datasets for deep analysis.",
            "Dall-E Image Generator": "Generates images from text prompts.",
            "Slack Toolkit": "Automates collaboration tasks within Slack, allowing message management and team communication."
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
