"""
Usage:
all: python -m start gpt-4o assess
predict: python -m start gpt-4o predict
display results: python -m start gpt-4o score
"""

import argparse
import os
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Specify the model name")
parser.add_argument("task", nargs="?", default="score",
                    choices=["score", "assess", "predict"],
                    help="Choose the task to run: 'score' (fetch final scores only), 'assess' (full process), 'predict' (only predictions).")
args = parser.parse_args()

# settings
try:
    cmd = f"python -m tool_assess.config.settings -model {args.model}"
    result = subprocess.run(cmd, shell=True, check=True)
    print("Config settings successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error: Config settings failed with exit code {e.returncode}")
    exit(1)

# Skip prediction steps if 'score' is provided
if args.task in ["assess", "predict"]:
    # single_task prediction
    try:
        cmd = f"python -m tool_assess.framework.prediction.single_task -model {args.model}"
        result = subprocess.run(cmd, shell=True, check=True)
        print("Single task evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Single task evaluation failed with exit code {e.returncode}")
        exit(1)

    # multi_task prediction
    try:
        cmd = f"python -m tool_assess.framework.prediction.multi_task -model {args.model}"
        result = subprocess.run(cmd, shell=True, check=True)
        print("Multi task evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Multi task evaluation failed with exit code {e.returncode}")
        exit(1)

# assess
if args.task in ["assess", "score"]:
    try:
        cmd = f"python -m tool_assess.framework.tool_assess -model {args.model}"
        result = subprocess.run(cmd, shell=True, check=True)
        print("Tool assessment completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Tool assessment failed with exit code {e.returncode}")
        exit(1)

# display
if args.task in ["assess", "score"]:
    try:
        cmd = f"streamlit run app.py"
        result = subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print()