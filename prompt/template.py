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
