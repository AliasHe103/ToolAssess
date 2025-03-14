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
            if content["result"] == "requires tool":
                _results.append(content["solving_tool"])
            elif content["result"] == "cannot be completed" or content["result"] == "no tool":
                _results.append(content["result"])
            else:
                raise ValueError("Invalid result type.")

    return _results

def get_predicted_results():
    _results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _id, content in data.items():
        if content["type"] == "requires tool":
            solve_tool = content["response"].split(":")[1].split(".")[0].strip()
            _results.append(solve_tool)
        elif content["type"] == "cannot be completed" or content["type"] == "no tool":
            _results.append(content["type"])
        else:
            raise ValueError("Invalid result type.")

    return _results

def optimize_results(r_true, r_pred):
    if len(r_true) != len(r_pred):
        raise ValueError("The length of true results and predicted results must be the same.")
    for i in range(len(r_true)):
        if r_true[i] == "no tool" or r_true[i] == "cannot be completed":
            r_true[i] = "false tool"
            r_pred[i] = "false tool" if (r_pred[i] == "no tool" or r_pred[i] == "cannot be completed") else "true tool"
        elif r_pred[i] == "no tool" or r_pred[i] == "cannot be completed":
            # true: requires tool
            r_true[i] = "true tool"
            r_pred[i] = "false tool"
            continue
        else:
            # true: requires tool, pred: requires tool
            # true: tool name1, pred: tool name2
            if r_true[i] == r_pred[i]:
                r_pred[i] = "true tool"
            else:
                r_pred[i] = "false tool"
            r_true[i] = "true tool"

def evaluate_model(r_true, r_pred):
    accuracy = accuracy_score(r_true, r_pred)
    precision = precision_score(r_true, r_pred, pos_label="true tool")
    recall = recall_score(r_true, r_pred, pos_label="true tool")
    f1 = f1_score(r_true, r_pred, pos_label="true tool")

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


true_results = get_true_results()
test_results = get_predicted_results()

optimize_results(true_results, test_results)
# print(true_results)
# print(test_results)

metrics = evaluate_model(true_results, test_results)
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")
