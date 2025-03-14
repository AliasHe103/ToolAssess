def get_model_name(model: str):
    # return [name for name, model_path in SUPPORTED_MODELS.items() if model_path == model][0]
    return model.split('/')[1]

together_model_map = {
    "llama-3-70B": "meta-llama/Meta-Llama-3-70B-Instruct-Turbo",
}
def to_api_model_string(name):
    return together_model_map.get(name)