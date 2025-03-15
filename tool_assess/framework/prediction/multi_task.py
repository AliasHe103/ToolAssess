import json
import os

from tool_assess.agents.compatible_agent import CompatibleAgent
from tool_assess.agents.deepseek_agent import DeepseekAgent
from tool_assess.agents.gpt_agent import GPTAgent
from tool_assess.agents.qwen_agent import QwenAgent
from tool_assess.agents.together_agent import TogetherAgent
from tool_assess.config import settings
from tool_assess.config.settings import model_name


def make_multi_task_prompt(rule=""):
    return f"""
    You are tasked with helping the User handle multiple tasks in a scenario.
    For each sub-task, you are provided with a toolset containing tool names and descriptions.
    You should determine if a tool is needed for that sub-task.
    
    ## Response Format
    Your response **must** strictly follow one of these three formats for each sub-task:
    1. "requires tool: [Tool Name]" → If the sub-task **must and can** be completed with a tool, **select only one** from the provided list.
    2. "no tool" → If the sub-task can be **completed without any tool**.
    3. "cannot be completed" → If the sub-task is **impossible even with the provided tools**, for example, you need other tools but they are not listed.
    
    ## Rules
    1. If multiple tools seem relevant, **only select the single most suitable** one.
    2. **Do not** include any extra explanation or reasoning.
    3. Sub-tasks are given by a Python list.
    4. For each scenario, organize your answers for all sub-tasks by a Python list, like: ["requires tool: Google Map", "no tool", "cannot be completed"].
    5. Your response **must** strictly follow the formats above **without any additional text**.
    {rule}
    """

def extract_sub_tasks(tasks):
    sub_tasks = []
    for task_id, task_data in tasks.items():
        sub_tasks.append(task_data["query"])
    return sub_tasks

def assess_multi_task(agent, system_prompt):
    for scenario_id, scenario_data in scenarios.items():
        role = scenario_data.get("role", "Unknown Role")
        tasks = scenario_data.get("tasks", {})
        sub_tasks = extract_sub_tasks(tasks)
        tools_data = scenario_data.get("tools", {})

        # Convert tools_data (dict) to JSON string for display
        tools_json = json.dumps(tools_data, indent=2)

        print(f"Assessing {scenario_id}.")

        # Build messages for the model
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User Role: {role}\nTask: {sub_tasks}\nAvailable Tools: {tools_json}"
            }
        ]

        # print(f"User Role: {role}\nTask: {sub_tasks}\nAvailable Tools: {tools_json}")

        response = ""
        try:
            response = agent.predict(messages)

            parsed_response = json.loads(response)
            if not isinstance(parsed_response, list):
                raise ValueError("Invalid response format.")

            results[scenario_id] = {
                "type": "success",
                "response": response
            }
            print(response)

        except Exception as e:
            print(f"Error on {scenario_id}: {e}.")
            print(response)
            results[scenario_id] = {
                "type": "error",
                "response": response
            }

def assess_on_deepseek_multi():
    deepseek_agent = DeepseekAgent(name="deepseek-chat")
    system_prompt = make_multi_task_prompt()
    assess_multi_task(agent=deepseek_agent, system_prompt=system_prompt)

def assess_on_openai_multi():
    gpt_agent = GPTAgent(model_name)
    system_prompt = make_multi_task_prompt()
    assess_multi_task(agent=gpt_agent, system_prompt=system_prompt)

def assess_on_openai_compatible_multi(key, url, rule=""):
    compatible_agent = CompatibleAgent(key, url, model_name)
    system_prompt = make_multi_task_prompt(rule)
    assess_multi_task(agent=compatible_agent, system_prompt=system_prompt)

def assess_on_qwen_multi(rule=""):
    qwen_agent = QwenAgent(model_name)
    system_prompt = make_multi_task_prompt(rule)
    assess_multi_task(agent=qwen_agent, system_prompt=system_prompt)

def assess_on_together_multi(rule=""):
    together_agent = TogetherAgent(model_name)
    system_prompt = make_multi_task_prompt(rule)
    assess_multi_task(agent=together_agent, system_prompt=system_prompt)

multi_task_file = settings.MULTI_TASK_DATA_PATH  # Path to multi-task scenarios
output_path = settings.MULTI_TASK_OUTPUT_PATH  # Path to save the results

if not os.path.exists(output_path):
    os.makedirs(output_path)

with open(multi_task_file, "r", encoding="utf-8") as f:
    scenarios = json.load(f)

results = {}

if model_name in ["test"]:
    assess_on_deepseek_multi()
elif model_name in ["gpt-4o", "o1"]:
    assess_on_openai_multi()
elif model_name in ["qwen-max", "qwen2.5-7b-instruct-1m", "deepseek-r1", "deepseek-v3"]:
    assess_on_qwen_multi()
elif model_name in ["llama-3.2-3B", "llama-3.3-70B"]:
    # model llama-3.2-3B has a poor ability of understanding the prompt, so errors may occur.
    assess_on_together_multi(rule="""
    6. Use double quotes, **avoid single quotes** in your reply.
    7. Your result **must** be exactly a **single line**.
    8. Be enclosed within square brackets [ ] to represent a valid Python list.
    9. Each item in the list must be a complete string and enclosed in double quotes like ["requires tool: Google Search", "no tool"].
    10. The tool name must NOT be enclosed in additional quotes.
    """)
elif model_name in ["glm-4-plus"]:
    assess_on_openai_compatible_multi(key=settings.zhipu_api_key, url="https://open.bigmodel.cn/api/paas/v4/")
elif model_name in ["Baichuan4-Turbo"]:
    assess_on_openai_compatible_multi(key=settings.baichuan_api_key, url="https://api.baichuan-ai.com/v1/",
                                      rule="6.Ensure that all JSON responses are returned as a single-line string without newline characters. The output should be compact and properly formatted to prevent parsing errors.")
else:
    raise ValueError("Invalid model name.")

output_file = os.path.join(output_path, model_name + "_multi.json")
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Multi-task results saved to {output_file}")
