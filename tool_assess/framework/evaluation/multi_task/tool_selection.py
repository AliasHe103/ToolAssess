import json
import os

from tool_assess.config import settings
from tool_assess.config.settings import model_name

result_path = settings.MULTI_TASK_OUTPUT_PATH
result_file = os.path.join(result_path, model_name + "_multi.json")

def get_true_results():
    _results = []
    with open(settings.MULTI_TASK_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for _id, content in data.items():
            scenario_result = []
            tasks = content["tasks"]
            for task_id, task in tasks.items():
                if task["result"] in ["cannot be completed", "no tool"]:
                    scenario_result.append(task["result"])
                elif task["result"] == "requires tool":
                    scenario_result.append(task["solving_tool"])
                else:
                    raise ValueError("Invalid task result type in true results.")

            _results.append(scenario_result)

    return _results


def optimize_response(result):
    return [
        name if name in ["cannot be completed", "no tool"] else name.split(":")[1].strip()
        for name in result
    ]


def get_predicted_results():
    _results = []
    with open(result_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    for _id, content in data.items():
        result = json.loads(content["response"])
        if content["type"] == "success":
            result = optimize_response(result)
            _results.append(result)
        else:
            raise ValueError("Invalid type in predicted results.")

    return _results

def ASA_score(r_true, r_pred):
    score = 0
    total_size = len(r_true)
    for i in range(total_size):
        if r_true[i] == r_pred[i]:
            score += 1
    score /= total_size

    return score

def PSA_score(r_true, r_pred):
    score = 0
    total_size = len(r_true)
    for i in range(total_size):
        l_true, l_pred = r_true[i], r_pred[i]
        local_score = 0
        local_size = len(l_true)

        for j in range(local_size):
            if l_true[j] == l_pred[j]:
                local_score += 1
        local_score /= local_size
        score += local_score

    score /= total_size
    return score

def evaluate_model(r_true, r_pred):
    asa = ASA_score(r_true, r_pred)
    psa = PSA_score(r_true, r_pred)

    return {
        "ASA": asa,
        "PSA": psa
    }

true_results = get_true_results()
predicted_results = get_predicted_results()

metrics = evaluate_model(true_results, predicted_results)
# for metric, value in metrics.items():
#     print(f"{metric}: {value:.4f}")
