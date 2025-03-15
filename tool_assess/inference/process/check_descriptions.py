import json
from tool_assess.config import settings

sample_file = settings.SINGLE_TASK_DATA_PATH
tools_file = settings.SUMMARIZED_TOOLS_PATH
output_file = sample_file

def update_tool_descriptions(sample_file, tools_file, output_file):
    with open(tools_file, "r", encoding="utf-8") as f:
        tools_data = json.load(f)
    tools_data = json.loads(tools_data)

    with open(sample_file, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    for op, details in sample_data.items():
        if "tools" in details:
            updated_tools = {}
            for tool, description in details["tools"].items():
                if tool in tools_data:
                    updated_tools[tool] = tools_data[tool]
                else:
                    updated_tools[tool] = description
            details["tools"] = updated_tools

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)

    print(f"Updated file {output_file}")

update_tool_descriptions(sample_file, tools_file, output_file)
