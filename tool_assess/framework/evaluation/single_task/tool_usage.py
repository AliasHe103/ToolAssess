import json
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tool_assess.config import settings
from tool_assess.config.settings import model_name

result_path = settings.SINGLE_TASK_OUTPUT_PATH
result_file = os.path.join(result_path, model_name + ".json")

def get_true_results():
    _results = []
    with open(settings.SINGLE_TASK_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _id, content in data.items():
            if content["result"] == "requires tool" or content["result"] == "cannot be completed" or content["result"] == "no tool":
                _results.append(content["result"])
            else:
                raise ValueError("Invalid true result type.")

    return _results

def get_predicted_results():
    _results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _id, content in data.items():
        if content["type"] == "requires tool" or content["type"] == "cannot be completed" or content["type"] == "no tool":
            _results.append(content["type"])
        else:
            raise ValueError("Invalid test result type.")

    return _results

def evaluate_model(r_true, r_pred):
    accuracy = accuracy_score(r_true, r_pred)
    precision = precision_score(r_true, r_pred, average='weighted')
    recall = recall_score(r_true, r_pred, average='weighted')
    f1 = f1_score(r_true, r_pred, average='weighted')

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


true_results = get_true_results()
test_results = get_predicted_results()

metrics = evaluate_model(true_results, test_results)
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")
