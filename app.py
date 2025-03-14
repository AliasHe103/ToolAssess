import os

import streamlit as st
import json
import pandas as pd

from tool_assess.config import settings
from tool_assess.config.settings import model_name

st.title("Tool Assess Benchmark Result")
st.write(f"Displaying test results of {model_name}")

# read results
output_path = settings.ASSESS_SCORE_PATH
if not os.path.exists(output_path):
    os.makedirs(output_path)
output_file = os.path.join(output_path, "tas_scores.json")
with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)

st.json(data)

df = pd.DataFrame([data])
st.dataframe(df)
