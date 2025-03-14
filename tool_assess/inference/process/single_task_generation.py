import os, json
from openai import OpenAI
from tool_assess.config import settings

DOMAIN_DIST = {
    'information_retrieval': 0.35,
    'data_processing': 0.25,
    'api_tasks': 0.2,
    'technical': 0.15,
    'daily_life': 0.05
}

with open(settings.SUMMARIZED_TOOLS_PATH, 'r', encoding='utf-8') as jf:
    data = json.load(jf)
    base_toolset = json.dumps(data, ensure_ascii=False)

single_task_prompt = '''
You are tasked with generating evaluation data for testing task-handling capabilities of Large Language Models. Each task in your generated dataset should be formatted as a JSON object, containing:

- A **realistic user query** (1 clear task).
- **2-3 tools** with strictly defined capabilities.

The possible task outcomes should fall into three categories:
1. **Task requires a tool(s)** - The query can only be solved with the provided tools. **Only one tool should be able to complete the task, while the others must be completely unrelated. Return the name of the tool that can solve the task.**
2. **Task does not require a tool** - The query can be answered based on general knowledge without using any tool.
3. **Task cannot be completed, even with tools** - The query cannot be solved, even if the provided tools are used.

Ensure:
- **Generate exactly {num_scenarios} multi-task scenarios.**
- **Each scenario should contain 2-4 logically related tasks.**
- **Each scenario should have 3-5 tools assigned.**
- **In "requires tool" cases, exactly one tool should be relevant per sub-task. Other tools must be unrelated.**
- **Scenarios with 3 tools must not exceed 50% of the total scenarios and 30% at least.**
- **Scenarios with 4 tools and 5 tools should have approximately equal distribution.**
- **Tasks requiring basic reasoning or common knowledge (e.g., simple math, general facts) should be classified as "no tool" and explicitly include "solving_tool": ""**
- **"Cannot be completed" cases should be tasks that remain impossible, even with the provided tools. Always include "solving_tool": "" explicitly.**

Format:

{{
    "op1": {{
        "query": "[User Request]",
        "tools": {{
            "[Tool Name1]": "[Tool Description1]",
            "[Tool Name2]": "[Tool Description2]",
            "[Tool Name3]": "[Tool Description3]"
        }},
        "result": "[Expected Result: requires tool / no tool / cannot be completed]",
        "solving_tool": "[Tool Name (if task requires a tool)]"
    }}
}}

Example:

{{
    "op1": {{
        "query": "Check flight status from NYC to London",
        "tools": {{
            "FlightAPI": "Provides real-time airline schedule data",
            "Dall-E Image Generator": "Creates digital images from text prompts."
        }},
        "result": "requires tool",
        "solving_tool": "FlightAPI"
    }},
    "op2": {{
        "query": "What is the capital city of France?",
        "tools": {{
            "Wikipedia": "Accesses extensive articles and knowledge.",
            "Currency Converter": "Calculates exchange rates between currencies."
        }},
        "result": "no tool",
        "solving_tool": ""
    }},
    "op3": {{
        "query": "Tell me the winning lottery numbers for tomorrow",
        "tools": {{
            "LotteryAPI": "Provides past lottery results, but cannot predict future numbers.",
            "WeatherAPI": "Provides weather forecasts but is unrelated to lottery results."
        }},
        "result": "cannot be completed",
        "solving_tool": ""
    }}
}}
'''.format(num_scenarios=settings.SINGLE_TASK_DATA_SIZE)


print("Start generating single-task dataset.")
client = OpenAI()
completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": single_task_prompt
        },
        {
            "role": "user",
            "content": base_toolset
        }
    ]
)

results = completion.choices[0].message.content
print(results)

with open(settings.SINGLE_TASK_DATA_PATH, 'w', encoding='utf-8') as jf:
    json.dump(results, jf, ensure_ascii=False, indent=4)