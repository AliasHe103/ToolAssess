import argparse
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument("-model", required=False)
args = parser.parse_args()

if args.model is not None:
    model_name = args.model
    with open("tool_assess/config/model_name.json", "w", encoding="utf-8") as f:
        json.dump({"name": model_name}, f, ensure_ascii=False, indent=4)
else:
    with open("tool_assess/config/model_name.json", "r", encoding="utf-8") as f:
        model_name = json.load(f)["name"]

available_models = [
    "deepseek",
    "deepseek-r1",
    "deepseek-v3",
    "gpt-4o",
    "o1",
    "qwen-max",
    "qwen2.5-7b-instruct-1m",
    "llama3.2-3b-instruct",
    "glm-4-plus",
    "Baichuan4-Turbo"
]

if model_name not in available_models:
    raise ValueError(f"Model {model_name} is not supported!")

# check api keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set!")

deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if deepseek_api_key is None:
    raise ValueError("DEEPSEEK_API_KEY is not set!")

qwen_api_key = os.environ.get("DASHSCOPE_API_KEY")

together_api_key = os.environ.get("TOGETHER_API_KEY")

zhipu_api_key = os.environ.get("ZHIPU_API_KEY")

baichuan_api_key = os.environ.get("BAICHUAN_API_KEY")

# set variables
TEST_DATA_PATH = "tool_assess/data/simple_test.json"
TEST_OUTPUT_PATH = "tool_assess/result/"
# summarization
ORG_TOOLS_DATA_PATH = "tool_assess/data/tools_with_original_descriptions.json"
SUMMARIZED_TOOLS_PATH = "tool_assess/data/tools_with_summarized_descriptions.json"
# single task
SINGLE_TASK_DATA_SIZE = 50
SINGLE_TASK_DATA_PATH = "tool_assess/data/single_task.json"
SINGLE_TASK_OUTPUT_PATH = "tool_assess/result/single_task/"
# multi task
MULTI_TASK_DATA_SIZE = 30
MULTI_TASK_DATA_PATH = "tool_assess/data/multi_task.json"
MULTI_TASK_OUTPUT_PATH = "tool_assess/result/multi_task/"
# assess score
ASSESS_SCORE_PATH = "tool_assess/result/assess_score/"