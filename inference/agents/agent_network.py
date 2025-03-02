# secure you have the corresponding api keys
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver

from inference.agents.models_configuration import gpt_4o_mini, deep_seek_v3, qwen_max
from inference.agents.prepare_tools import available_tools

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute to generate your chart."],
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )


def make_system_prompt(suffix: str) -> str:
    return (
        "You are a helpful AI assistant, collaborating with other assistants."
        " The User will ask questions, and you should handle them to the best of your ability."
        " You are provided with a list of tools and their descriptions."
        " For each question, try to identify the most suitable tool that can help solve the task."
        " Your answer must follow one of the three formats:"
        "\n\n"
        " 1.If you work it out and need a tool for the task(or if you do use a tool), first mention the tool name followed by '<FINAL ANSWER>' and a brief explanation."
        "For example: '<FINAL ANSWER>Gmail toolkit (the tool name). I can use this tool to send an email for the user.'"
        "\n\n"
        " 2. If you work it out and don’t need any tool for the task, reply with '<FINAL ANSWER>I don't need a tool for this task.' and provide your solution."
        "\n\n "
        " 3.If none of the available tools can help you with the task, start with 'No tool can solve this task.' and then pass the task to the next assistant."
        "\n\n"
        " Remember to include the prefix '<FINAL ANSWER>' if you work it out, so the team knows to stop."
        " If not, feel free to pass it to your colleague(s)."
         "\n\n"
        " Now you can see the tools."
        f"{available_tools}"
        "\n\n"
        f"{suffix}"
    )

from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

def get_next_node(last_message: BaseMessage, goto: str):
    if "FINAL ANSWER" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


# Research agent and node
collaborating_agent1 = create_react_agent(
    model=gpt_4o_mini,
    # qwen_max,
    tools=[],
    prompt=make_system_prompt(
        "You are working with a colleague and will handle the task first."
    ),
)

# START->colleague1v->colleague2
def collaborating_node1(
    state: MessagesState,
) -> Command[Literal["colleague2", END]]:
    result = collaborating_agent1.invoke(state)
    goto = get_next_node(result["messages"][-1], "colleague2")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="colleague1"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

collaborating_agent2 = create_react_agent(
    gpt_4o_mini,
    tools=[],
    prompt=make_system_prompt(
        "You are working with a researcher colleague. You are the second to handle the task."
    ),
)

# colleague2->END
def collaborating_node2(
    state: MessagesState,
) -> Command[Literal[END]]:
    result = collaborating_agent2.invoke(state)
    goto = get_next_node(result["messages"][-1], END)
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="colleague2"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("colleague1", collaborating_node1)
workflow.add_node("colleague2", collaborating_node2)

workflow.add_edge(START, "colleague1")
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)

events = graph.stream(
    {
        "messages": [
            (
                "user",
                "How is the weather in WuHan? "
            )
        ],
    },
    # Maximum number of steps to take in the graph
    {"configurable": {"thread_id": "1"}, "recursion_limit": 150},
)
for s in events:
    role, msg = next(iter(s.items()))
    if 'messages' in msg and isinstance(msg['messages'], list):
        print(f"{role}: {msg['messages'][-1].content}")
    else:
        print(s)
    print("----")