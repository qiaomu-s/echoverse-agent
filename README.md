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

### 2. ReAct (推理 + 行动)

ReAct 在 LLM 推理情况和采取行动之间交替进行。LLM 分析当前状态和目标，选择并使用一个工具，然后使用该工具的输出进行下一步的思考和行动。此循环重复进行，直到问题得到解决。

![](./_assets/react.png)

#### 优点：

- **利用外部信息：** 有效地使用外部工具收集模型自身无法处理的任务所需的信息。
- **可解释的推理：** 交织的推理和行动步骤允许对 Agent 的过程进行一定程度的跟踪。
- **广泛的适用性：** 适用于需要外部知识或特定操作的任务，例如问答、信息检索和任务执行。

## IoT 设备接口说明

> 当前仅函数调用（Function Calling）方式支持 IoT 相关功能，ReAct 策略暂不支持。

> 注意：文档中的 `http://www.label-studio.top` 仅为示例，实际接口地址应由系统动态配置。

Agent 策略中支持通过 API 动态获取 IoT 设备工具和设备状态，相关接口如下：

### 1. 获取可控设备列表

- **接口地址**：`/open/iot/device/controlDevices`
- **请求方式**：GET
- **请求头**：
  - `X-API-Key`: 用户 API Key
  - `X-Device-ID`: 设备 ID
- **返回示例**：

```json
[
  {
    "type": "function",
    "function": {
      "name": "zhaomnigdeng",
      "description": "洗手间照明灯",
      "parameters": {
        "type": "object",
        "properties": {
          "methods": {
            "type": "string",
            "description": "打开或者关闭洗手间照明灯",
            "enum": ["open", "close"]
          }
        },
        "required": ["methods"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "taideng",
      "description": "家庭卧室台灯",
      "parameters": {
        "type": "object",
        "properties": {
          "methods": {
            "type": "string",
            "description": "打开或者关闭台灯",
            "enum": ["open", "close"]
          }
        },
        "required": ["methods"]
      }
    }
  },
  {
    "type": "function",
    "iot_name": "Speaker",
    "function": {
      "name": "SetVolume",
      "description": "扬声器 - 设置音量",
      "parameters": {
        "type": "object",
        "properties": {
          "volume": {
            "description": "0到100之间的整数",
            "type": "number"
          },
          "iot_name": {
            "type": "string",
            "description": "Speaker的标识名称"
          }
        },
        "required": ["volume"]
      }
    }
  },
  {
    "type": "function",
    "iot_name": "Lamp",
    "function": {
      "name": "TurnOn",
      "description": "一个测试用的灯 - 打开灯",
      "parameters": {
        "type": "object",
        "properties": {
          "iot_name": {
            "type": "string",
            "description": "Lamp的标识名称"
          }
        },
        "required": []
      }
    }
  },
  {
    "type": "function",
    "iot_name": "Lamp",
    "function": {
      "name": "TurnOff",
      "description": "一个测试用的灯 - 关闭灯",
      "parameters": {
        "type": "object",
        "properties": {
          "iot_name": {
            "type": "string",
            "description": "Lamp的标识名称"
          }
        },
        "required": []
      }
    }
  }
]
```

- 每个元素代表一个可控设备的操作能力，包含 type、function 及可选的 iot_name 字段。
- function 字段内包含 name（操作名）、description（描述）、parameters（参数定义，含类型、描述、可选枚举等）。

### 2. 获取设备状态

- **接口地址**：`/open/iot/device/deviceState`
- **请求方式**：GET
- **请求头**：
  - `X-API-Key`: 用户 API Key
  - `X-Device-ID`: 设备 ID
- **返回示例**：

```json
[
  { "name": "Speaker", "state": { "volume": 80 } },
  { "name": "Lamp", "state": { "power": false } }
]
```

- 其中 `name` 表示设备名称，`state` 为设备的状态信息（如音量、开关等）。

### 3. 控制设备（IOT 设备调用）

- **接口地址**：`/open/iot/device/executeControl`
- **请求方式**：POST
- **请求头**：
  - `X-API-Key`: 用户 API Key
  - `X-Device-ID`: 设备 ID
  - `Content-Type`: application/json
- **请求体示例**：

```json
{
  "function": {
    "name": "TurnOn",
    "iot_name": "Lamp",
    "arguments": {
      "iot_name": "Lamp"
    }
  }
}
```

- **返回示例**：

```json
{
  "code": 1000,
  "message": "success"
}
```

- 其中 `function` 字段包含要调用的操作名、iot_name 及参数。
- 返回 code 为 1000 表示调用成功。

---
