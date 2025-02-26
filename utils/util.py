import os
import json
import config.settings as settings
from config.settings import get_model_name


def load_messages():
    with open(settings.TEST_DATA_PATH, 'r', encoding='utf-8') as jf:
        data = json.load(jf)
        messages = data['messages']
        return messages

def save_results(results, model_path: str):
    model_name = get_model_name(model_path)

    with open(settings.TEST_OUTPUT_PATH + model_name + ".json", 'w', encoding='utf-8') as jf:
        json.dump(results, jf, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    msg = load_messages()
    save_results(msg, "test-model")