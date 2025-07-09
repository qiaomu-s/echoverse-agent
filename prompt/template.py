ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES = """Respond to the human as helpfully and accurately as possible.

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {{tool_names}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
{{historic_messages}}
Question: {{query}}
{{agent_scratchpad}}
Thought:"""  # noqa: E501


ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES = """Observation: {{observation}}
Thought:"""

ENGLISH_REACT_CHAT_PROMPT_TEMPLATES = """Respond to the human as helpfully and accurately as possible.

{{instruction}}

You have access to the following tools:

{{tools}}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {{tool_names}}

Provide only ONE action per $JSON_BLOB, as shown:

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{
  "action": "Final Answer",
  "action_input": "Final response to human"
}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation:.
"""  # noqa: E501


ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES = ""

CHINESE_REACT_COMPLETION_PROMPT_TEMPLATES = """尽可能有帮助和准确地回应人类。

{{instruction}}

您可以使用以下工具：

{{tools}}

使用 json blob 通过提供操作键（工具名称）和操作输入键（工具输入）来指定工具。
有效的“操作”值：“最终答案”或 {{tool_names}}

每个 $JSON_BLOB 只提供一个操作，如下所示：

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

请遵循以下格式：

问题：需要回答的输入问题
思考：考虑先前和后续的步骤
操作：
```
$JSON_BLOB
```
观察：操作结果
...（重复思考/操作/观察 N 次）
思考：我知道该如何回应
操作：
```
{
  "action": "Final Answer",
  "action_input": "给人类的最终答复"
}
```

开始！提醒始终使用单个操作的有效 json blob 进行响应。如有必要，请使用工具。如果合适，请直接回应。格式为 Action:```$JSON_BLOB```then Observation:.
{{historic_messages}}
问题：{{query}}
{{agent_scratchpad}}
思考："""  # noqa: E501


CHINESE_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES = """观察：{{observation}}
思考："""

CHINESE_REACT_CHAT_PROMPT_TEMPLATES = """尽可能有帮助和准确地回应人类。

{{instruction}}

您可以使用以下工具：

{{tools}}

使用 json blob 通过提供操作键（工具名称）和操作输入键（工具输入）来指定工具。
有效的“操作”值：“最终答案”或 {{tool_names}}

每个 $JSON_BLOB 只提供一个操作，如下所示：

```
{
  "action": $TOOL_NAME,
  "action_input": $ACTION_INPUT
}
```

请遵循以下格式：

问题：需要回答的输入问题
思考：考虑先前和后续的步骤
操作：
```
$JSON_BLOB
```
观察：操作结果
...（重复思考/操作/观察 N 次）
思考：我知道该如何回应
操作：
```
{
  "action": "Final Answer",
  "action_input": "给人类的最终答复"
}
```

开始！提醒始终使用单个操作的有效 json blob 进行响应。如有必要，请使用工具。如果合适，请直接回应。格式为 Action:```$JSON_BLOB```then Observation:.
"""  # noqa: E501


CHINESE_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES = ""

REACT_PROMPT_TEMPLATES = {
    "english": {
        "chat": {
            "prompt": ENGLISH_REACT_CHAT_PROMPT_TEMPLATES,
            "agent_scratchpad": ENGLISH_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES,
        },
        "completion": {
            "prompt": ENGLISH_REACT_COMPLETION_PROMPT_TEMPLATES,
            "agent_scratchpad": ENGLISH_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES,
        },
    },
    "chinese": {
        "chat": {
            "prompt": CHINESE_REACT_CHAT_PROMPT_TEMPLATES,
            "agent_scratchpad": CHINESE_REACT_CHAT_AGENT_SCRATCHPAD_TEMPLATES,
        },
        "completion": {
            "prompt": CHINESE_REACT_COMPLETION_PROMPT_TEMPLATES,
            "agent_scratchpad": CHINESE_REACT_COMPLETION_AGENT_SCRATCHPAD_TEMPLATES,
        },
    }
}

# MCP Function Calling Prompt Templates

ENGLISH_MCP_FUNCTION_CALLING_SYSTEM_PROMPT = """You are an intelligent assistant that can help users accomplish various tasks through tool calling.

{instruction}

## Available Tools

You have access to two types of tools:
1. **Built-in Tools**: Standard plugin tools with predefined functionality
2. **MCP Tools**: Dynamic tools fetched from Model Context Protocol (MCP) servers that provide real-time capabilities

### MCP Tools Guidelines

**What are MCP Tools?**
MCP (Model Context Protocol) tools are dynamically available tools that can interact with external systems, APIs, and services. These tools are fetched in real-time and may include:
- IoT device controls (smart home devices, sensors, etc.)
- External API integrations
- Real-time data sources
- Custom automation scripts

**How to Use MCP Tools:**
1. **Analyze the request**: Understand what the user wants to accomplish
2. **Choose appropriate tools**: Select the most relevant tool(s) from available options
3. **Parameter accuracy**: Ensure all required parameters are provided with correct values
4. **Handle responses**: Process tool outputs and provide meaningful feedback to users

**Best Practices:**
- Always read tool descriptions carefully before calling
- Provide all required parameters as specified in the tool schema
- For IoT controls: Be specific about device names, values, and states
- For data queries: Use appropriate filters and parameters
- If a tool call fails, explain the error and suggest alternatives
- Combine multiple tool calls when necessary to complete complex tasks

**Error Handling:**
- If an MCP tool is unavailable, suggest alternative approaches
- If parameters are missing, ask the user for clarification
- If a tool returns an error, explain what went wrong in user-friendly terms

Remember: MCP tools enable real-time interaction with external systems, so always consider the current context and user's environment when making tool calls."""

CHINESE_MCP_FUNCTION_CALLING_SYSTEM_PROMPT = """你是一个智能助手，能够通过工具调用帮助用户完成各种任务。

{instruction}

## 可用工具

你可以使用两种类型的工具：
1. **内置工具**：具有预定义功能的标准插件工具
2. **MCP工具**：从模型上下文协议(MCP)服务器动态获取的工具，提供实时功能

### MCP工具使用指南

**什么是MCP工具？**
MCP（模型上下文协议）工具是动态可用的工具，可以与外部系统、API和服务交互。这些工具是实时获取的，可能包括：
- 物联网设备控制（智能家居设备、传感器等）
- 外部API集成
- 实时数据源
- 自定义自动化脚本

**如何使用MCP工具：**
1. **分析请求**：理解用户想要完成的任务
2. **选择合适的工具**：从可用选项中选择最相关的工具
3. **参数准确性**：确保提供所有必需的参数及正确的值
4. **处理响应**：处理工具输出并向用户提供有意义的反馈

**最佳实践：**
- 调用前务必仔细阅读工具描述
- 按照工具架构提供所有必需的参数
- 对于物联网控制：明确指定设备名称、数值和状态
- 对于数据查询：使用适当的过滤器和参数
- 如果工具调用失败，解释错误并建议替代方案
- 必要时组合多个工具调用以完成复杂任务

**错误处理：**
- 如果MCP工具不可用，建议替代方法
- 如果缺少参数，请向用户询问澄清
- 如果工具返回错误，用用户友好的术语解释出了什么问题

记住：MCP工具能够与外部系统进行实时交互，因此在进行工具调用时，请始终考虑当前上下文和用户的环境。"""

# MCP Prompt Templates Dictionary
MCP_FUNCTION_CALLING_PROMPT_TEMPLATES = {
    "english": {
        "system_prompt": ENGLISH_MCP_FUNCTION_CALLING_SYSTEM_PROMPT,
    },
    "chinese": {
        "system_prompt": CHINESE_MCP_FUNCTION_CALLING_SYSTEM_PROMPT,
    }
}
