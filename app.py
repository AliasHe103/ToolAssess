import os
import streamlit as st
import json
import pandas as pd
import altair as alt

from tool_assess.config import settings
from tool_assess.config.settings import model_name

st.title("Tool Assess Benchmark Result")
st.write(f"Displaying test results of {model_name}")

output_path = settings.ASSESS_SCORE_PATH
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_file = os.path.join(output_path, "tas_scores.json")

with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

model_scores = []
for name, score in data.items():
    if name == model_name:
        st.write(f"Tool Assess Score: {score}")
    model_scores.append({"Model": name, "Score": score})

df = pd.DataFrame(model_scores)

chart = (
    alt.Chart(df)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", sort="-y", title="Model Name"),
        y=alt.Y("Score:Q", title="Test Score"),
        color=alt.value("steelblue"),
        tooltip=["Model", "Score"]
    )
    .properties(
        width=800,
        height=400,
        title="Tool Assessment Scores of All Agents"
    )
)

st.altair_chart(chart, use_container_width=True)

# detailed task scores
detailed_scores_file = os.path.join(output_path, "models_scenario_scores.json")

with open(detailed_scores_file, "r", encoding="utf-8") as f:
    detailed_data = json.load(f)

detailed_scores = []
for model, scores in detailed_data.items():
    detailed_scores.append({
        "Model": model,
        "Task 1": scores[0],
        "Task 2": scores[1],
        "Task 3": scores[2],
        "Task 4": scores[3]
    })

df_detailed = pd.DataFrame(detailed_scores)

st.subheader("Detailed Model Performance on Different Tasks")
st.dataframe(df_detailed)

df_melted = df_detailed.melt(id_vars=["Model"], var_name="Task", value_name="Score")

detailed_chart = (
    alt.Chart(df_melted)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", title="Model Name", sort="-y"),
        y=alt.Y("Score:Q", title="Score"),
        color="Task:N",
        tooltip=["Model", "Task", "Score"]
    )
    .properties(
        width=800,
        height=400,
        title="Model Performance on Individual Tasks"
    )
)

st.altair_chart(detailed_chart, use_container_width=True)