# 概述

Dify Chatflow/Workflow 中的 Agent 节点使大型语言模型 (LLM) 能够自主使用工具。该插件具有两种官方 Dify Agent 推理策略，使 LLM 能够在运行时动态选择和运行工具，以解决多步骤问题。

## 策略

### 1. 函数调用

函数调用将用户命令映射到特定的函数或工具。LLM 识别用户的意图，决定调用哪个函数，并提取所需的参数。这是一种调用外部功能的直接机制。

![](./_assets/function_calling.png)

#### 优点：

- **精确：** 直接为已定义的任务调用正确的工具，避免复杂的推理。
- **易于外部集成：** 将外部 API 和工具集成为可调用函数。
- **结构化输出：** 提供结构化的函数调用信息，便于处理。
- **MCP 集成：** 支持动态获取和执行 MCP（Model Context Protocol）工具，提供更丰富的外部能力。

### 2. ReAct (推理 + 行动)

ReAct 在 LLM 推理情况和采取行动之间交替进行。LLM 分析当前状态和目标，选择并使用一个工具，然后使用该工具的输出进行下一步的思考和行动。此循环重复进行，直到问题得到解决。

![](./_assets/react.png)

#### 优点：

- **利用外部信息：** 有效地使用外部工具收集模型自身无法处理的任务所需的信息。
- **可解释的推理：** 交织的推理和行动步骤允许对 Agent 的过程进行一定程度的跟踪。
- **广泛的适用性：** 适用于需要外部知识或特定操作的任务，例如问答、信息检索和任务执行。

## MCP 工具接口说明

> 当前仅函数调用（Function Calling）方式支持 MCP 相关功能，ReAct 策略暂不支持。

> 注意：文档中的 `http://www.label-studio.top` 仅为示例，实际接口地址应由系统动态配置。

Agent 策略中支持通过 API 动态获取 MCP 工具，使用前需要配置以下参数：

- **API Host**：MCP 服务的主机地址
- **API Key**：用于身份验证的 API 密钥
- **Device ID**：设备标识符

相关接口如下：

### 1. 获取 MCP 工具列表

- **接口地址**：`/open/iot/device/mcpTools`
- **请求方式**：GET
- **请求头**：
  - `X-API-Key`: 用户 API Key
  - `X-Device-ID`: 设备 ID
- **返回示例**：

```json
{
  "code": 1000,
  "message": "success",
  "data": {
    "count": 3,
    "tools": [
      {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {
              "type": "string",
              "description": "城市名称"
            },
            "unit": {
              "type": "string",
              "description": "温度单位",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["city"]
        }
      },
      {
        "name": "calculate",
        "description": "执行数学计算",
        "inputSchema": {
          "type": "object",
          "properties": {
            "expression": {
              "type": "string",
              "description": "数学表达式"
            }
          },
          "required": ["expression"]
        }
      },
      {
        "name": "search_web",
        "description": "搜索网络信息",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "搜索关键词"
            },
            "limit": {
              "type": "number",
              "description": "返回结果数量限制"
            }
          },
          "required": ["query"]
        }
      }
    ]
  }
}
```

- 返回数据包含 `count`（工具总数）和 `tools`（工具列表）。
- 每个工具包含 `name`（工具名）、`description`（描述）、`inputSchema`（输入参数定义）。

### 2. 执行 MCP 工具

- **接口地址**：`/open/iot/device/executeMcpTool`
- **请求方式**：POST
- **请求头**：
  - `X-API-Key`: 用户 API Key
  - `X-Device-ID`: 设备 ID
  - `Content-Type`: application/json
- **请求体示例**：

```json
{
  "toolName": "get_weather",
  "params": {
    "city": "北京",
    "unit": "celsius"
  }
}
```

- **返回示例**：

```json
{
  "code": 1000,
  "message": "success",
  "data": {
    "result": "北京今日天气晴，温度25°C，湿度60%"
  }
}
```

- 其中 `toolName` 字段包含要执行的工具名称，`params` 包含工具参数。
- 返回 code 为 1000 表示执行成功，`data` 字段包含工具执行结果。

---
