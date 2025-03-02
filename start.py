import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("task")
args = parser.parse_args()

if args.task == "summarization":
    os.system("python -m generation_test")
elif args.task == "task_generation":
    os.system("python -m inference.process.task_generation")
elif args.task == "test_json":
    os.system("python -m test.test_json")
elif args.task == "assess":
    os.system("python -m assess_framework.assess.deepseek_assess")
else:
    raise ValueError("Invalid task!")
