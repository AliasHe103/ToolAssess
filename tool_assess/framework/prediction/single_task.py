import json
import os

from tool_assess.agents.compatible_agent import CompatibleAgent
from tool_assess.agents.deepseek_agent import DeepseekAgent
from tool_assess.agents.gpt_agent import GPTAgent
from tool_assess.agents.qwen_agent import QwenAgent
from tool_assess.agents.together_agent import TogetherAgent
from tool_assess.config import settings
from tool_assess.config.settings import model_name


def make_single_task_prompt(rule=""):
    return f"""
    You are tasked with helping the User with his questions. 
    For each user query, you are provided with a toolset containing tool names and descriptions about them.
    You should determine if a tool is needed for the task.
    
    ## Response Format
    Your response **must** strictly follow one of these three formats:
    1. "requires tool: [Tool Name]." → If the task **must and can** be completed with a tool, **select only one best matched tool** from the provided list.
    2. "no tool." → If the task can be **completed without any tool**.
    3. "cannot be completed." → If the task is **impossible even with the provided tools**(you cannot achieve it with any single tool), or **the task itself is impossible**.
    
    ## Rules
    1. If multiple tools seem relevant, **only select the most suitable one**.
    2. **Do not** include any extra explanation or reasoning.
    3. Your response **must** strictly follow the formats above **without any additional text**.
    4. You can **only** use the tools provided by the User.
    5. If you can solve the task with or without tools, then you best choose "no tool".
    {rule}
    """

def get_response_type(response):
    response_type = "error"
    if "requires tool" in response:
        response_type = "requires tool"
    elif "no tool" in response:
        response_type = "no tool"
    elif "cannot be completed" in response:
        response_type = "cannot be completed"
    else:
        raise ValueError("Invalid response format.")
    return response_type

def assess_single_task(agent, system_prompt):
    response = ""
    for op_id, sample in samples.items():
        query = sample["query"]
        tools = json.dumps(sample["tools"], indent=2)

        print(f"Assessing {op_id}...")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query: {query}\nAvailable Tools: {tools}"}
        ]

        try:
            response = agent.predict(messages)
            response_type = get_response_type(response)

            results[op_id] = {
                "type": response_type,
                "response": response
            }

        except Exception as e:
            print(f"Error on {op_id}: {e}.")
            results[op_id] = {
                "type": "error",
                "response": response
            }
            continue

def assess_on_deepseek_single():
    deepseek_agent = DeepseekAgent(name="deepseek-chat")
    system_prompt = make_single_task_prompt()
    assess_single_task(agent=deepseek_agent, system_prompt=system_prompt)
def assess_on_openai_single():
    gpt_agent = GPTAgent(model_name)
    system_prompt = make_single_task_prompt()
    assess_single_task(agent=gpt_agent, system_prompt=system_prompt)

def assess_on_openai_compatible_single(key, url, rule=""):
    compatible_agent = CompatibleAgent(key, url, model_name)
    system_prompt = make_single_task_prompt(rule)
    assess_single_task(agent=compatible_agent, system_prompt=system_prompt)

def assess_on_qwen_single(rule=""):
    qwen_agent = QwenAgent(model_name)
    system_prompt = make_single_task_prompt(rule)
    assess_single_task(agent=qwen_agent, system_prompt=system_prompt)

def assess_on_together_single(rule=""):
    together_agent = TogetherAgent(model_name)
    system_prompt = make_single_task_prompt(rule)
    assess_single_task(agent=together_agent, system_prompt=system_prompt)

sample_file = settings.SINGLE_TASK_DATA_PATH
output_path = settings.SINGLE_TASK_OUTPUT_PATH

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(sample_file, "r", encoding="utf-8") as f:
    samples = json.load(f)

results = {}


if model_name in ["test"]:
    assess_on_deepseek_single()
elif model_name in ["gpt-4o", "o1"]:
    assess_on_openai_single()
elif model_name in ["qwen-max", "qwen2.5-7b-instruct-1m", "deepseek-r1", "deepseek-v3"]:
    assess_on_qwen_single()
elif model_name in ["llama-3.2-3B", "llama-3.3-70B"]:
    assess_on_together_single(rule="6. Use double quotes, **avoid single quotes** in your reply.")
elif model_name in ["glm-4-plus"]:
    assess_on_openai_compatible_single(key=settings.zhipu_api_key, url="https://open.bigmodel.cn/api/paas/v4/")
elif model_name in ["Baichuan4-Turbo"]:
    assess_on_openai_compatible_single(key=settings.baichuan_api_key, url="https://api.baichuan-ai.com/v1/")
else:
    raise ValueError("Invalid model name.")

output_file = os.path.join(output_path, model_name + ".json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Results saved to {output_file}")