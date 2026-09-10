# Alibaba Cloud Token Plan Provider Proposal

## 目标

Pygent 增加独立 Provider `aliyun_token_plan`，覆盖 Token Plan 个人版官方当前提供的全部模型，并用统一 capabilities 准确描述文本、图像、视频和音频模型。Provider、模型目录、协议和 Adapter 继续彼此解耦：

- Provider 表示模型服务来源和套餐边界；
- protocol 表示实际 wire contract；
- capabilities 表示该 Provider、Model ID 和 protocol 组合公开的模型能力；
- Adapter 决定 Pygent 是否能够执行该 protocol。

Token Plan 与阿里云按量付费、Coding Plan 使用不同的 API Key 和 Base URL，因此不合并为同一个 Provider。第一版复用现有 OpenAI Chat Completions 和 Anthropic Messages Adapter；多模态专用协议只进入目录，不在本次实现 Adapter。

## Provider 与连接 preset

`ProviderCatalog.builtin()` 增加：

| Provider | Protocol | Base URL | Credential environment |
| --- | --- | --- | --- |
| `aliyun_token_plan` | `openai_chat_completions` | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` | `ALIYUN_TOKEN_PLAN_OPENAI_API_KEY` |
| `aliyun_token_plan` | `anthropic_messages` | `https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic` | `ALIYUN_TOKEN_PLAN_ANTHROPIC_API_KEY` |

默认 protocol 为 `openai_chat_completions`。两个连接 preset 使用 Bearer credential 引用，真实 API Key 继续只在部署资源装配时读取。

Provider catalog 只提供配置默认值，不持有 client，不注册 Adapter，也不表示目录中的全部模型都可由当前 Pygent 执行。`BuiltinModelProtocol` 继续只枚举已经内置 Adapter 的 `openai_chat_completions` 和 `anthropic_messages`。

连接信息以 [Token Plan 快速开始](https://help.aliyun.com/zh/model-studio/token-plan-personal-quick-start) 和 [阿里云百炼 Base URL](https://help.aliyun.com/en/model-studio/base-url) 为依据。

## 能力契约

统一 capabilities 不增加 `task` 字段。模型用途由输入模态、输出模态和 protocol 表达。

`modalities.input` 和 `modalities.output` 使用封闭取值：

```text
text, image, audio, video
```

流式能力改为输出模态集合：

```yaml
streaming:
  output: [text]
```

`streaming.output` 必须是 `modalities.output` 的子集。解析后的值不可变；旧 `streaming.text` 结构不再接受。

Token limits 允许未知或不适用：

```yaml
limits:
  context_tokens: null
  max_output_tokens: null
```

两个字段分别接受正整数或 `null`。现有四个文本能力模板仍要求 materialize 调用者显式提供两个正整数，不增加图像、视频或音频模板。

其余能力字段保持现有结构：

```yaml
tools:
  call: true
  choice: [none, auto, required, named]
  parallel: true
structured_output:
  json_object: true
  json_schema: true
reasoning:
  supported: true
  controllable: true
```

## 官方模型目录

第一版以 [Token Plan 个人版官方模型列表](https://help.aliyun.com/zh/model-studio/token-plan-personal-overview) 为目录来源。账号 `/models` 只用于发布前核验，不把账号返回的历史名称或额外模型自动收入内置目录。

### 文本与视觉理解

以下九个 Model ID 分别建立 `openai_chat_completions` 和 `anthropic_messages` 两条目录记录：

| Model ID | Input | Output | Streaming output | Tools | Reasoning | Context |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen3.8-max` | text, image | text | text | yes | supported, controllable | 1,000,000 |
| `qwen3.8-flash` | text, image | text | text | yes | supported, controllable | 1,000,000 |
| `qwen3.7-max` | text | text | text | yes | supported, controllable | 1,000,000 |
| `qwen3.7-plus` | text, image | text | text | yes | supported, controllable | 1,000,000 |
| `qwen3.6-flash` | text, image | text | text | yes | supported, controllable | 1,000,000 |
| `deepseek-v4-pro` | text | text | text | yes | supported, controllable | 1,000,000 |
| `deepseek-v4-pro-0813` | text | text | text | yes | supported, controllable | 1,000,000 |
| `deepseek-v4-flash-0731` | text | text | text | yes | supported, controllable | 1,000,000 |
| `glm-5.2` | text | text | text | yes | supported, controllable | 1,000,000 |

`max_output_tokens` 只在官方资料能够明确确认时保存正整数，否则保存 `null`，不从 context window 推算。

Structured output 按 protocol 分别记录：

- OpenAI Chat Completions 下，Qwen 3.7/3.8 支持 JSON Object 与 JSON Schema；Qwen 3.6、DeepSeek 和 GLM 按普通 JSON 能力记录，不把 JSON Object 推断成严格 JSON Schema。
- Anthropic Messages 下，Qwen 3.7/3.8、DeepSeek 和 GLM 记录严格 JSON Schema；Qwen 3.6 只记录普通 JSON 能力。

Tools 的 `call`、`choice` 和 `parallel` 同样按每个 Model ID 和 protocol 的官方接口约束分别填写，不按模型家族统一猜测。文本能力依据 [阿里云百炼文本生成模型表](https://help.aliyun.com/zh/model-studio/text-generation-model)、[结构化输出](https://help.aliyun.com/zh/model-studio/qwen-structured-output) 和 [Anthropic Messages](https://help.aliyun.com/zh/model-studio/anthropic-api-messages)。

### 图像

| Model ID | Protocol | Input | Output | Streaming output |
| --- | --- | --- | --- | --- |
| `qwen-image-3.0-pro` | `dashscope_multimodal_generation` | text, image | image | empty |
| `wan2.7-image` | `dashscope_multimodal_generation` | text, image | image | empty |
| `wan2.7-image-pro` | `dashscope_multimodal_generation` | text, image | image | empty |

三条记录的 tools、structured output 和 reasoning 为 false，limits 为 null。输入同时包含 text 和 image，因为官方明确支持生成与编辑。能力依据 [图片生成与编辑](https://help.aliyun.com/zh/model-studio/image-model)。

### 视频

| Model ID | Protocol | Input | Output | Streaming output |
| --- | --- | --- | --- | --- |
| `happyhorse-1.1-t2v` | `dashscope_video_generation` | text | video | empty |
| `happyhorse-1.1-i2v` | `dashscope_video_generation` | text, image | video | empty |
| `happyhorse-1.1-r2v` | `dashscope_video_generation` | text, image | video | empty |

三条记录的 tools、structured output 和 reasoning 为 false，limits 为 null。异步任务不等同于流式输出。协议与调用方式依据 [Token Plan 接入多模态生成模型](https://help.aliyun.com/zh/model-studio/token-plan-multimodal-gen)。

### 音频

| Model ID | Protocol | Input | Output | Streaming output | Tools | Limits |
| --- | --- | --- | --- | --- | --- | --- |
| `qwen-audio-3.0-tts-plus` | `dashscope_speech_synthesis` | text | audio | audio | no | null |
| `qwen-audio-3.0-realtime-plus` | `dashscope_realtime` | text, audio | text, audio | text, audio | yes | context 40,960; output 8,192 |
| `qwen-audio-3.0-asr-flash` | `dashscope_speech_recognition` | audio | text | empty | no | null |

Realtime 模型的 reasoning 为 false。其他两条音频记录的 tools、structured output 和 reasoning 为 false。能力依据 [Qwen Audio Realtime](https://help.aliyun.com/zh/model-studio/qwen-audio-3-0-realtime-plus)、[Qwen Audio TTS](https://help.aliyun.com/zh/model-studio/qwen-audio-3-0-tts-plus) 和 [Qwen Audio ASR](https://help.aliyun.com/zh/model-studio/qwen-audio-3-0-asr-flash)。

目录最终包含 18 个官方 Model ID 和 27 条 `(provider, model_id, protocol)` 记录。五个 DashScope protocol 是开放字符串，不加入 `BuiltinModelProtocol`，第一版也不提供对应 Adapter。

## 用户配置与 UI

普通 Agent 开发者和 Runtime 不读取目录。应用或 UI 使用 Provider catalog 和能力目录把用户选择展开成完整 `ModelConfig` Mapping。

选择 `qwen3.8-max` 的 OpenAI 接口后，保存结果形如：

```yaml
models:
  aliyun_qwen_primary:
    provider: aliyun_token_plan
    model_id: qwen3.8-max
    protocol: openai_chat_completions
    connection:
      base_url: https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
      credential:
        env: ALIYUN_TOKEN_PLAN_OPENAI_API_KEY
      verify_ssl: true
    provider_options: {}
    capabilities:
      modalities:
        input: [text, image]
        output: [text]
      streaming:
        output: [text]
      tools:
        call: true
        choice: [none, auto, required, named]
        parallel: true
      structured_output:
        json_object: true
        json_schema: true
      reasoning:
        supported: true
        controllable: true
      limits:
        context_tokens: 1000000
        max_output_tokens: null

model_groups:
  assistant:
    models: [aliyun_qwen_primary]
```

同一个 Model ID 使用 Anthropic Messages 时保存为另一个本地模型条目，使用 Anthropic Base URL、credential 引用及该 protocol 对应的完整 capabilities。Pygent 不自动切换协议。

目录 ID、能力模板名和 override 不进入用户配置。用户可以逐项修改 UI 展开的值；保存结果始终是完整 capabilities。目录升级只影响之后新建或重新编辑的条目，不修改已经保存或正在使用的 `ModelSpec`。

专用模态模型也可由 UI 展示。应用根据自己实际装配的 Adapter 标记其是否可执行；这个状态不写进 Provider 或模型能力目录。用户或第三方 Adapter 可以为开放 protocol 构造完整连接配置，但 Pygent 第一版不会把目录存在误报为内置执行支持。

## Alibaba OpenAI Provider options

`OpenAICompatibleAdapter` 为 `aliyun_token_plan` 接受并严格校验当前目录模型所需的官方扩展字段：

- `enable_thinking`；
- `preserve_thinking`；
- `reasoning_effort`；
- `thinking`；
- `thinking_budget`；
- `tool_stream`。

字段类型、枚举和值域遵循阿里云 OpenAI Chat Completions 协议。未知字段继续在 Provider I/O 前拒绝。Responses API 的 Harness tools、代码解释器和联网搜索不通过这些选项伪装支持。

OpenAI Chat Completions Adapter 不维护支持 `reasoning_content` 的 Provider 白名单。任何使用该 protocol 的模型，只要响应实际返回合法的 `reasoning_content`，都使用现有版本化 `ModelContinuation` 保存它。以 Alibaba Token Plan 为例：

```python
ModelContinuation(
    provider="aliyun_token_plan",
    protocol="openai_chat_completions",
    data={"version": 1, "reasoning_content": "..."},
)
```

流式与非流式调用采用同一规则：字段不存在时不产生 continuation；字段存在时必须是字符串，否则按非法 Provider 响应拒绝。Continuation 的 `provider` 来自当前 `ModelSpec`，后续工具调用只在 Provider 与 protocol 同时匹配时回传。逻辑不依赖 Provider 白名单或具体 Model ID，也不跨 Provider 或 protocol 转换。现有 DeepSeek continuation 自然落入同一条协议规则，行为保持不变。

## Invoker 与能力警告

Invoker 是否使用流式文本 transport 改为检查：

```python
"text" in model.capabilities.streaming.output
```

当前 `ModelCallLayer` 的消息与结果契约仍然是文本，因此这项修改不实现图像、视频或音频执行。

`limits.max_output_tokens` 为 `null` 时跳过输出上限能力警告；为正整数时继续执行现有比较。能力检查只读取已经解析的不可变值。匹配路径不加载目录、不探测网络、不创建警告事件对象。

## 必要迁移

`streaming.text` 到 `streaming.output` 以及 nullable limits 会机械影响：

- capabilities 值对象和严格 Mapping 解析；
- capabilities codec、definition/effect digest 与持久化 snapshot；
- Invoker 的流式 transport 选择和输出上限警告；
- 三份包内目录 JSON；
- 相关测试、LLM 第一原则、README 和 SDK 示例。

这是 born-as-new 的配置结构：不保留 `streaming.text` 别名、双格式 codec 或旧目录 snapshot reader。Runtime profile、Worker、durable replay 和 SQLite 的执行语义不重新设计，只使用统一的新能力投影。

## 测试与交付

实现遵循测试先行，并覆盖：

- modality 封闭取值、`streaming.output` 子集校验和不可变性；
- nullable limits 的解析、codec 和警告行为；
- 既有文本目录迁移后的调用行为不变；
- `aliyun_token_plan` 的两个 Provider preset；
- 18 个 Model ID、27 条能力记录及逐 protocol 差异；
- Alibaba Provider options 的合法投影和失败关闭；
- OpenAI Chat Completions `reasoning_content` 的响应驱动保存、非法值拒绝、流式与非流式工具循环续传，以及无字段时的既有行为；
- 未安装的专用协议在网络请求前明确失败；
- Runtime、Worker、SQLite 和 durable replay 的新能力投影 round trip；
- wheel 与 sdist 中的目录 JSON 和 checksum。

完整验证执行：

```text
uv run pytest -q
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
uv build
uvx twine check dist/*
```

最后从主工作区 `.env` 显式读取 Token Plan credential 引用，分别通过 OpenAI Chat Completions 和 Anthropic Messages 执行文本冒烟测试。不输出 API Key、响应正文或 thinking 内容。第一版不对没有 Adapter 的多模态协议报告调用成功。

## 实施边界

本次只增加 Alibaba Token Plan Provider、官方模型能力目录、通用输出模态 streaming 表达、nullable token limits、Alibaba OpenAI 私有字段，以及 OpenAI Chat Completions 响应驱动的 reasoning continuation。

本次不增加图像、视频、TTS、ASR、Realtime 或 OpenAI Responses Adapter，不增加多模态 Message，不增加 Adapter registry、自动协议选择、能力路由、远程目录同步、Runtime 配置、容量系统或 Provider client 生命周期。新增契约按 born-as-new 方式实现，同时保持已有 OpenAI、Anthropic、DeepSeek、fallback、managed Runtime 和非模型功能的行为。
