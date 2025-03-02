import os

from langchain_deepseek import ChatDeepSeek
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_community.chat_models.tongyi import ChatTongyi

key = os.environ.get("DEEPSEEK_API_KEY")
if key is None:
    raise ValueError("DEEPSEEK_API_KEY is not set!")

gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini")
deep_seek_v3 = ChatDeepSeek(model="deepseek-chat", base_url="https://api.deepseek.com/v1",)
qwen_max = ChatTongyi(model="qwen-max", api_key=os.environ.get("DASHSCOPE_API_KEY"))

# vicuna_7b = ChatHuggingFace(llm=HuggingFaceEndpoint(
#     repo_id="lmsys/vicuna-7b-v1.3",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# ), verbose=True)

# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# response = tongyi.invoke(messages)
# print(response)