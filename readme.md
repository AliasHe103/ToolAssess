# ToolAssess
***


## 1. 背景
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
***

## 2. 评估标准

### 2.1 单任务场景

现在的问题在于如果预测结果是"requires tool"，那么结果可能是各种各样的工具，这样不利于分类；因此，在此基础上，引入两个新类型，分别为"true tool"和"false tool"，这样，原本的三分类问题变成了["true tool", "false tool", "no tool","cannot be completed"]的四分类问题;
如果正确结果为"no tool"或者"cannot be completed"，且预测结果为"requires tool"，那么预测结果更新为"false tool";
1. "true tool": 预测结果和正确结果都为"requires tool"且工具名相同
2. "false tool": 预测结果和正确结果都为"requires tool"且工具名不同，或预测结果为"requires tool"而正确结果为"no tool"或者"cannot be completed"

#### 2.1.1 工具使用意识
在**单任务场景**下，
评估模型的**工具使用意识**，即 测试模型**能否正确分类任务为 "requires tool"、"no tool"、"cannot be completed"**。 
这是一个三分类问题。
测试模型的工具使用意识，通过下面四个指标：

🔍 **如何解释 Accuracy、Precision、Recall 和 F1 Score 在“工具使用意识”测试中的现实意义？**  
1️⃣ **准确率（Accuracy）**  
✅ 解释  
$$Accuracy=\frac{TP+TN}{TP+TN+FP+FN}$$
 
✅ 现实意义：衡量 模型对任务是否需要工具的整体判断准确性，即：

能否正确分类为 "requires tool"、"no tool"、"cannot be completed"。
但不能区分具体错误类型（如误分类 "requires tool" 为 "no tool"）。  
2️⃣ **精确率（Precision）**  
✅ 解释  
$$Precision=\frac{TP}{TP+FP}$$
 
✅ 现实意义：衡量 模型预测某个类别时的可靠性，即：

预测为 "requires tool" 的任务中，有多少实际需要工具？
预测为 "no tool" 的任务中，有多少实际不需要工具？
预测为 "cannot be completed" 的任务中，有多少实际无法完成？  
⚠️ 如果 Precision 低：

模型误报较多（比如错误地把 "no tool" 预测为 "requires tool"）。  
3️⃣ **召回率（Recall）**  
✅ 解释  
$$Recall=\frac{TP}{TP+FN}$$
 
✅ 现实意义：衡量 模型能否正确识别所有任务类别，即：

所有真正 "requires tool" 的任务中，模型能正确预测多少？
所有真正 "no tool" 的任务中，模型能正确预测多少？
所有真正 "cannot be completed" 的任务中，模型能正确预测多少？  
⚠️ 如果 Recall 低：

模型容易漏掉需要工具的任务（如把 "requires tool" 误分类为 "no tool"）。  
4️⃣ **F1 分数（F1 Score）**  
✅ 解释  
$$F1=\frac{2*Precision*Recall}{Precision+Recall}$$
 
✅ 现实意义：衡量 模型在精确率（避免误报）和召回率（避免漏报）之间的平衡，即：

如果 Precision 高，Recall 低 → 说明模型更谨慎，误报少但可能漏掉一些任务
如果 Recall 高，Precision 低 → 说明模型更激进，能识别大部分 "requires tool"，但误报较多  
⚠️ 如果类别不均衡，F1 Score 可以更公平地评估模型性能。

📌 结论  
✅ 如果关注模型是否能准确识别任务类型，关注 Accuracy  
✅ 如果关注模型是否误报 "requires tool"，关注 Precision  
✅ 如果关注模型是否漏报 "requires tool"，关注 Recall  
✅ 如果类别分布不均衡（某一类任务占比较少），关注 F1 Score  

#### 2.1.2 工具选择能力

🔍 **评估“工具选择能力”**  
在**单任务场景**下，
评估模型的**工具选择能力**，即测试模型**能否正确识别出任务需要使用工具，并选择正确的工具"**。  
这不在是一个三分类问题，因为"requires tool"的具体结果可能是各种各样的工具名，事实上，这是一个多分类问题。  
为了简化该问题，我将"requires tool"的分类结果细化为"true tool"和"false tool"，其中"true tool"表示模型预测结果和正确结果都为"requires tool"且工具名相同，"false tool"表示模型预测结果和正确结果都为"requires tool"且工具名不同。  
这样，问题就变成了["true tool", "false tool", "no tool","cannot be completed"]的四分类问题。  
不过，这种定义下，仍然存在一个问题，真实标签中只含有"true tool"而不会含有"false tool"。其后果是：  

1. FN = 0。
2. Recall恒为1。
3. Precision 会偏高。

🔍**类别重映射（Re-mapping Labels）**  
这样的后果当时不是所期望的，因此，做出如下规定：

1. 真实标签和预测标签都为"requires tool"，真实集为"true tool"。如果工具名相同，则预测集为"true tool"；否则，预测集为"false tool"。
2. 真实标签为"requires tool"，预测标签为"no tool"或者"cannot be completed"，则真实集为"true tool"，预测集为"false tool"。
3. 真实标签为"no tool"或者"cannot be completed"，则真实集为"true tool"。如果预测标签为"requires tool"，则预测集为"true tool"，否则预测集为"false tool"。

经过如上规定，问题就变成了["true tool", "false tool"]的二分类问题。  
这样做的好处有：

1. 解决了FN = 0的缺点和Recall恒为1的问题。
2. Precision, Recall, F1 Score 都变得更直观。 
3. 使计算出的指标更具有现实意义。

🔍 **评估“工具选择能力”时，Accuracy、Precision、Recall 和 F1 Score 的现实意义**  
1️⃣ 准确率（Accuracy）  
✅ 现实意义：  
衡量模型整体判断任务是否需要工具的能力和工具选择能力。  
如果 Accuracy 高，说明：
模型大部分情况下能正确判断是否需要工具。
模型在 "requires tool" 任务中，大部分情况下能选择正确的工具。
但 Accuracy 无法区分：
模型是完全错分类，还是只是选错了工具。  

2️⃣ 精确率（Precision）  
✅ 现实意义：

衡量 模型在预测 "requires tool" 时，选择正确工具的概率。
如果 Precision 高，说明：
模型预测 "requires tool" 时，通常能选对工具。
如果 Precision 低：
模型经常错误地预测 "requires tool" 或选择了错误的工具（误报 False Positive 过多）。

3️⃣ 召回率（Recall）  
✅ 现实意义：

衡量 在所有 "requires tool" 任务中，模型能正确预测多少。
如果 Recall 高，说明：
模型能很好地识别 "requires tool" 任务，并正确选择工具。
如果 Recall 低：
模型可能错误地将 "requires tool" 预测为 "false tool"（即 "no tool" 或 "cannot be completed"）。
说明模型 低估了工具的必要性，无法正确识别需要工具的任务。

4️⃣ F1 分数（F1 Score）  
✅ 现实意义：  
衡量 模型在 Precision 和 Recall 之间的平衡。
如果 F1 高：
模型既能正确预测 "requires tool" 任务，又能正确选择工具。
如果 F1 低：
说明 Precision 或 Recall 其中一个较差，模型要么误报过多，要么漏报过多。

📌 结论  
✅ 如果关注模型是否误报 "requires tool"（避免误选工具），关注 Precision  
✅ 如果关注模型是否漏报 "requires tool"（避免忽略工具的必要性），关注 Recall  
✅ 如果类别不均衡（部分任务较少），关注 F1 Score

## 2.2 多任务场景

