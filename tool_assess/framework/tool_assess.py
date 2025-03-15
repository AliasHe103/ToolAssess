import json
import os.path
import numpy as np

from tool_assess.config import settings
from tool_assess.config.settings import model_name
from tool_assess.framework.evaluation.multi_task.tool_selection import metrics as multi_task_tool_selection
from tool_assess.framework.evaluation.multi_task.tool_usage import metrics as multi_task_tool_usage
from tool_assess.framework.evaluation.single_task.tool_selection import metrics as single_task_tool_selection
from tool_assess.framework.evaluation.single_task.tool_usage import metrics as single_task_tool_usage

if not os.path.exists(settings.ASSESS_SCORE_PATH):
    os.makedirs(settings.ASSESS_SCORE_PATH)
models_scores_file = os.path.join(settings.ASSESS_SCORE_PATH, "models_scores.json")

def load_models_scores(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {}

def save_models_scores(updated_models_scores):
    with open(models_scores_file, 'w', encoding='utf-8') as f:
        json.dump(updated_models_scores, f, ensure_ascii=False, indent=4)

# Update models_scores.json
models_scores = load_models_scores(models_scores_file)
models_scores[model_name] = {
    "SingleTaskToolUsage": single_task_tool_usage,
    "SingleTaskToolSelection": single_task_tool_selection,
    "MultiTaskToolUsage": multi_task_tool_usage,
    "MultiTaskToolSelection": multi_task_tool_selection
}
save_models_scores(models_scores)
models_scores = load_models_scores(models_scores_file)
model_names = list(models_scores.keys())

def extract_scores(models_scores, metric_type):
    return [[model[metric_type][key] for key in model[metric_type]] for model in models_scores.values()]

def compute_variance_based_weights(scores_list):
    values = np.array(scores_list)
    variances = np.var(values, axis=0)
    total_variance = np.sum(variances)
    weights = variances / total_variance if total_variance > 0 else np.ones_like(variances) / len(variances)
    return weights

def compute_weighted_score(metrics, weights):
    return round(np.dot(weights, np.array(list(metrics.values()))) * 100, 2)

# Extract scores
st_tu = extract_scores(models_scores, "SingleTaskToolUsage")
st_ts = extract_scores(models_scores, "SingleTaskToolSelection")
mt_tu = extract_scores(models_scores, "MultiTaskToolUsage")
mt_ts = extract_scores(models_scores, "MultiTaskToolSelection")

# Compute variance-based weights
st_tus_weight = compute_variance_based_weights(st_tu)
st_tss_weight = compute_variance_based_weights(st_ts)
mt_tus_weight = compute_variance_based_weights(mt_tu)
mt_tss_weight = compute_variance_based_weights(mt_ts)

models_scores_dict = {}
def compute_tas_weights():
    models_scores_matrix = []
    for name in model_names:
        scores = models_scores[name]
        model_st_tu, model_st_ts, model_mt_tu, model_mt_ts = scores["SingleTaskToolUsage"], scores["SingleTaskToolSelection"], scores["MultiTaskToolUsage"], scores["MultiTaskToolSelection"]
        # Compute weighted scores for each model
        model_st_tus = compute_weighted_score(model_st_tu, st_tus_weight)
        model_st_tss = compute_weighted_score(model_st_ts, st_tss_weight)
        model_mt_tus = compute_weighted_score(model_mt_tu, mt_tus_weight)
        model_mt_tss = compute_weighted_score(model_mt_ts, mt_tss_weight)
        model_scores_list = [model_st_tss, model_st_tus, model_mt_tss, model_mt_tus]
        models_scores_matrix.append(model_scores_list)
        models_scores_dict[name] = model_scores_list

    # Compute Tool Assess weights and score
    tas_weights = compute_variance_based_weights(models_scores_matrix)
    return tas_weights

tas_dict = {}
def compute_tool_assess_score(model_name):
    tas_weights = compute_tas_weights()
    for name in model_names:
        tas = round(np.dot(tas_weights, np.array(models_scores_dict[name])), 2)
        tas_dict[name] = tas

    return tas_dict[model_name]

tool_assess_score = compute_tool_assess_score(model_name)
print(f"Tool Assess score for {model_name}: {tool_assess_score}")

tas_scores_file = os.path.join(settings.ASSESS_SCORE_PATH, "tas_scores.json")
models_scenario_scores = os.path.join(settings.ASSESS_SCORE_PATH, "models_scenario_scores.json")

with open(models_scenario_scores, 'w', encoding='utf-8') as f:
    json.dump(models_scores_dict, f, ensure_ascii=False, indent=4)
with open(tas_scores_file, 'w', encoding='utf-8') as f:
    json.dump(tas_dict, f, ensure_ascii=False, indent=4)