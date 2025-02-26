TEST_DATA_PATH = "data/simple_test.json"
TEST_OUTPUT_PATH = "result/"
ORG_TOOLS_DATA_PATH = "data/tools_with_original_descriptions.json"
SUMMARIZED_TOOLS_PATH = "data/tools_with_summarized_descriptions.json"

SUPPORTED_MODELS = {
    "Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b-chat-hf",
    "vicuna-7b-v1.3": "lmsys/vicuna-7b-v1.3",
    "deepseek-llm-7b-chat": "deepseek-ai/deepseek-llm-7b-chat"
}

def get_model_name(model: str):
    # return [name for name, model_path in SUPPORTED_MODELS.items() if model_path == model][0]
    return model.split('/')[1]