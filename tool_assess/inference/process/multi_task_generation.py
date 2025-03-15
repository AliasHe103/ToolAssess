import os
import json
from openai import OpenAI
from tool_assess.config import settings


def load_toolset():
    """Load tool descriptions from the specified JSON file."""
    with open(settings.SUMMARIZED_TOOLS_PATH, 'r', encoding='utf-8') as jf:
        return json.load(jf)


def generate_multi_task_prompt(num_scenarios):
    """Generate the refined multi-task prompt with multi-turn improvements."""
    return  f"""
    You are tasked with generating multi-task scenario data for testing the capabilities of Large Language Models. Each scenario **must be formatted as a JSON object**, following these guidelines:

    1. Number of Scenarios
       - Generate exactly {{num_scenarios}} scenarios when the user specifies the quantity.
       - Each scenario is identified by a key (e.g., "scenario1", "scenario2", etc.).

    2. User Role (Context)
       - Each scenario must include a "role" field that briefly describes the user or context (e.g., "role": "Software Developer", "role": "Travel Blogger", etc.).
       - The tasks in this scenario should logically reflect the chosen role’s needs and interests.

    3. Tasks
       - Each scenario should contain 2 to 4 tasks, presented under a "tasks" object.
       - Every task has:
         - "query": A user request or question.
         - "result": Must be one of:
           1. "requires tool" — This task can only be solved with a specific tool.
           2. "no tool" — This task can be solved by the LLM’s general knowledge without tools.
           3. "cannot be completed" — Even with tools, the task can’t be fulfilled.
         - "solving_tool":
           - The name of the tool if "requires tool".
           - An empty string "" if "no tool" or "cannot be completed".

    4. Tools
       - Each scenario has a "tools" object, listing 3 to 5 tools (key-value pairs of toolName: description).
       - Tools must be selected strategically:
         - Include at least one tool that cannot solve any task (a distractor).
         - Avoid using multiple tools with the same functionality.
       - If a task is "requires tool", that tool must appear in "tools".

    5. Logic & Realism
       - The tasks should be logically connected to the user’s "role".
       - If a task has result: "cannot be completed", ensure the reason is that:
         - The scenario is missing the required tool, or
         - The request is inherently impossible.

    6. JSON Format
       - The entire output must be a well-formed JSON object containing all scenarios.
       - Example structure:
         {{
           "scenario1": {{
             "role": "Some Role",
             "tasks": {{
               "task1": {{ "query": "...", "result": "...", "solving_tool": "..." }},
               "task2": {{ ... }}
             }},
             "tools": {{
               "Tool A": "Description",
               "Tool B": "Description"
             }}
           }},
           "scenario2": {{ ... }}
         }}
       - Realistic output example:
        {{
            "scenario1": {{
                "role": "Space Hobbyist",
                "tasks": {{
                  "task1": {{
                    "query": "I'd love to see the latest images from the Mars rover mission.",
                    "result": "requires tool",
                    "solving_tool": "NASA Toolkit"
                  }},
                  "task2": {{
                    "query": "Analyze the rover's temperature data logs for any extreme values or anomalies.",
                    "result": "requires tool",
                    "solving_tool": "E2B Data Analysis"
                  }},
                  "task3": {{
                    "query": "Could you explain what a 'light-minute' means in space travel?",
                    "result": "no tool",
                    "solving_tool": ""
                  }},
                  "task4": {{
                    "query": "Let's schedule a private video call with NASA next week to discuss rover updates.",
                    "result": "cannot be completed",
                    "solving_tool": ""
                  }}
                }},
                "tools": {{
                  "NASA Toolkit": "Accesses NASA data for space-related inquiries, enabling LLMs to provide informed responses on space exploration topics.",
                  "E2B Data Analysis": "Processes end-to-end business data for actionable insights, helping LLMs generate recommendations or reports from business operations.",
                  "CDP": "Consolidates customer data for tailored responses, enabling LLMs to enhance user interactions with personalized support and recommendations.",
                  "Twilio": "Enables integrations with messaging platforms to automate messaging tasks, assisting LLMs in managing communications."
                }}
            }}
        }}

    7. Minimize Repetition
       - Avoid reusing the same tool combinations, question styles, or scenario types too frequently.
       - Each scenario’s tasks and toolset should be distinct enough to ensure broad coverage.

    8. Generation Logic(excluded in the final result)
       - Consider this high-level process:
         1) Search possible tool combinations (ensure they differ from previously used sets).
         2) Set the user's role (context).
         3) Define question tasks that align with the role and the chosen tools, ensuring variety in phrasing and content.
         4) Check each question to confirm logical consistency with the user’s role and the selected tools.

    By following these guidelines, you will produce coherent, logical multi-task scenarios in JSON format, each involving:
    - A specific user role
    - Multiple tasks that either require a tool, need no tool, or cannot be completed
    - A set of available tools (including at least one irrelevant distractor)
    - Minimal duplication across scenarios
    """

def generate_multi_task_dataset():
    """Generate the multi-task dataset using OpenAI's API with multi-turn interaction refinement."""
    toolset = load_toolset()
    toolset_str = "I will give you a toolset for your generation job.\n" + json.dumps(toolset, indent=4)
    prompt = generate_multi_task_prompt(settings.MULTI_TASK_DATA_SIZE)

    client = OpenAI()
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": toolset_str},
    ]

    print("Starting multi-task dataset generation...")
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=messages
    )

    results = completion.choices[0].message.content
    print(results)

    with open(settings.MULTI_TASK_DATA_PATH, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, ensure_ascii=False, indent=4)

    print("Dataset successfully saved!")


if __name__ == "__main__":
    generate_multi_task_dataset()
