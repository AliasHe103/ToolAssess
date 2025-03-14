import json
from config import settings

# 定义文件路径
sample_file = settings.SINGLE_TASK_DATA_PATH   # 样本文件
tools_file = settings.SUMMARIZED_TOOLS_PATH     # 工具集文件
output_file = sample_file  # 输出文件

def update_tool_descriptions(sample_file, tools_file, output_file):
    # 读取工具集文件
    with open(tools_file, "r", encoding="utf-8") as f:
        tools_data = json.load(f)
    tools_data = json.loads(tools_data)

    # 读取样本文件
    with open(sample_file, "r", encoding="utf-8") as f:
        sample_data = json.load(f)

    # 遍历样本数据，替换工具描述
    for op, details in sample_data.items():
        if "tools" in details:
            updated_tools = {}
            for tool, description in details["tools"].items():
                if tool in tools_data:
                    updated_tools[tool] = tools_data[tool]  # 用 tools.json 中的描述替换
                else:
                    updated_tools[tool] = description  # 如果工具集没有该工具，则保留原描述
            details["tools"] = updated_tools

    # 保存更新后的样本文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sample_data, f, indent=4, ensure_ascii=False)

    print(f"Updated file {output_file}")

update_tool_descriptions(sample_file, tools_file, output_file)
