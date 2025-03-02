本研究聚焦于大语言模型（LLM）驱动的智能代理（Agent）在执行任务时的工具规划与选择能力，旨在评估其在实际应用中的效率与可靠性。
考察多任务处理、信息检索和模型选择？
主要考察智能代理在以下三个方面的表现：
1. 根据任务需求规划工具（Tool Selection）
2. 选择合适的检索工具（Retrieval）
3. LLM模型选择

在LangGraph中，工具使用并不是模型的自主调用，而是通过预先设计工作流实现的。所以，要测试模型的工作使用能力，应该要考虑其他方式。

single_task.json:
1. 负类举例：不需要工具，但是给定的工具也能辅助（强干扰）
```json
{
  "query": "Calculate the square root of 2025",
  "tools": {
    "Wolfram Alpha": "Acts as a computational knowledge engine for LLMs, enabling complex calculations and structured data retrieval.",
    "Human as a tool": "Enhances LLM functionality with human expertise for nuanced decision-making and improved accuracy in specialized tasks."
  },
  "result": "no tool",
  "solving_tool": ""
}
```  

现在的问题在于如果预测结果是"requires tool"，那么结果可能是各种各样的工具，这样不利于分类；因此，在此基础上，引入两个新类型，分别为"true tool"和"false tool"，这样，原本的三分类问题变成了["true tool", "false tool", "no tool","cannot be completed"]的四分类问题;
如果正确结果为"no tool"或者"cannot be completed"，且预测结果为"requires tool"，那么预测结果更新为"false tool";
1. "true tool": 预测结果和正确结果都为"requires tool"且工具名相同
2. "false tool": 预测结果和正确结果都为"requires tool"且工具名不同，或预测结果为"requires tool"而正确结果为"no tool"或者"cannot be completed"