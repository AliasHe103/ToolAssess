import os

model_path = "lmsys/vicuna-7b-v1.3"
print("You are running the test on model {}".format(model_path))
os.system("python -m generation_test --model {}".format(model_path))