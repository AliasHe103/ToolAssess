# set variables
import os

TEST_DATA_PATH = "data/simple_test.json"
TEST_OUTPUT_PATH = "result/"
# summarization
ORG_TOOLS_DATA_PATH = "data/tools_with_original_descriptions.json"
SUMMARIZED_TOOLS_PATH = "data/tools_with_summarized_descriptions.json"
# single task
SINGLE_TASK_DATA_SIZE = 50
SINGLE_TASK_DATA_PATH = "data/single_task.json"
SINGLE_TASK_OUTPUT_PATH = "result/single_task/"
# multi task
MULTI_TASK_DATA_SIZE = 3
MULTI_TASK_DATA_PATH = "data/multi_task.json"
MULTI_TASK_OUTPUT_PATH = "result/multi_task/"

SUPPORTED_MODELS = {
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna-7b-v1.3": "lmsys/vicuna-7b-v1.3",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat"
}

# check api keys
openai_api_key = os.environ.get("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY is not set!")

deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if deepseek_api_key is None:
    raise ValueError("DEEPSEEK_API_KEY is not set!")

def get_model_name(model: str):
    # return [name for name, model_path in SUPPORTED_MODELS.items() if model_path == model][0]
    return model.split('/')[1]