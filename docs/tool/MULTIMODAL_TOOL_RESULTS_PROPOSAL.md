# 多模态工具结果设计

> 状态：公开契约与标准 `FileTools.read_image`、`read_video` 已实现；文件路径工具通过
> synthetic 与 `glm-5.3-flash` live-provider 验收。
>
> 本文服从 [Pygent 第一原则](../FEATURES.md)、[Tool 第一原则](FEATURES.md)、
> [Context 第一原则](../context/FEATURES.md)、[LLM 第一原则](../llm/FEATURES.md)
> 以及 [Execution contract](../EXECUTION.md)。实现前需要把本文冻结的新增类型、
> wire schema 和 capability 字段同步到对应 SDK。

## 目标与范围

Pygent 允许一次工具调用返回由多个文本、JSON、图片和视频块组成的模型可见结果。
ToolResult 保留原 `call_id`，ReAct 仍把它作为 ToolMessage 加入模型投影；Provider Adapter
负责把中立内容块转换成具体接口接受的 `role: tool` 消息。业务工具不拼接
`image_url`、`video_url`、data URI 或其他 Provider 请求结构。

首期范围包括：

- 混合文本、JSON、图片和视频内容块；
- URL、稳定资源引用和二进制输入的规范化；
- OpenAI Chat Completions compatible 接口的结构化 tool content 编码；
- Context、Worker、effect 和 durable history 的无损往返；
- 模型输入模态与协议 tool-message 能力的独立检查；
- 无效、失效、不兼容和超限媒体的明确失败。

音频、媒体输出生成、自动转码、抽帧、OCR 和字幕提取不属于首期。标准
`read_image(file_path)` 支持 PNG、JPEG、GIF、WebP，`read_video(file_path)` 支持 MP4；
两者按 workspace 路径规则读取有界字节并产生 inline 媒体内容块。

## 与现有原则的关系

本能力不改变 Tool、Context、LLM 和 Runtime 的责任：

- ToolResult 和内容块是不可变、封闭、具有稳定 schema 的公开值；
- ToolCallLayer 继续按原 ToolCall 顺序把多个 ToolResult 聚合成一个 ToolMessage；
- ReAct 或应用组合层决定 ToolMessage 何时进入 Context，工具层不隐式提交历史；
- Context 保存模型投影，不持有文件句柄、Store、resolver、client 或其他活资源；
- Provider Adapter 负责 wire 映射，Runtime 不构造或解释 Provider 请求；
- 长期会话和媒体存储仍由应用负责，Runtime 只保存其声明支持的执行恢复事实。

原始 `bytes`、文件对象、PIL 对象、视频 reader 和 Provider client 不能进入 ToolResult、
Message 或 Context。应用入口可以接收二进制，但必须在公开值构造前规范化为严格 JSON
可表达的媒体源。跨进程和恢复不使用 pickle，也不按 Python 类名恢复对象。

## 公开内容模型

当前 SDK 提供封闭的 `ToolResultContent` 联合类型：

```python
ToolResultContent = ToolResultText | ToolResultJson | ToolResultMedia


@dataclass(frozen=True, slots=True)
class ToolResultText:
    text: str


@dataclass(frozen=True, slots=True)
class ToolResultJson:
    value: JsonValue


@dataclass(frozen=True, slots=True)
class ToolResultMedia:
    media_type: Literal["image", "video"]
    mime_type: str
    source: MediaSource
    detail: Literal["auto", "low", "high"] | None = None
```

`detail` 是中立的质量提示。Adapter 仅在目标协议存在等价字段时发送；不能表达时应明确
忽略这一非语义提示，而不能忽略媒体本身。首期不允许应用通过任意 Mapping 自定义未知
内容块类型，避免 Provider 字段穿透公共契约。

ToolResult 增加一个默认空元组的模型可见投影：

```python
@dataclass(frozen=True, slots=True)
class ToolResult:
    # 现有字段保持不变
    output: JsonValue = None
    content: tuple[ToolResultContent, ...] = ()
```

`output` 继续保存工具的严格 JSON 业务结果，并参与任务查询、effect 重放和持久化。
`content` 是显式的模型可见投影：

- `content == ()` 时沿用现有行为，Adapter 把 status、output 和安全错误字段编码为文本；
- `content != ()` 时 Adapter 使用内容块，不再把整个数组 JSON 序列化成字符串；
- `ToolResultJson` 在没有原生 JSON tool block 的协议上转换为单独的 text block；
- 普通 list/dict 返回值不会因形状类似内容块而被自动识别为多模态结果；
- `call_id`、name、status、task 和错误字段仍属于外层 ToolResult，不在每个块中复制。

当 `content` 非空时，`output` 不会被隐式复制进模型投影；工具若希望模型同时看到业务
JSON，必须显式增加 `ToolResultJson`。这样持久任务查询可以保留完整业务结果，而模型只
接收工具明确选择的安全内容。

首期只有 `status="succeeded"` 的结果可以携带媒体块。拒绝、失败、取消、unknown 和
detached 结果继续使用现有安全文本投影，避免失败结果夹带未经验证的媒体。

应用工具可以返回由 Python adapter 识别的 `ToolOutput`：

```python
return ToolOutput(
    output={"width": 640, "height": 480},
    content=(
        ToolResultText("工具读取到的图片。"),
        ToolResultMedia(
            media_type="image",
            mime_type="image/png",
            source=MediaSource.resource(
                uri="media://session-1/image-7",
                sha256="...",
                size_bytes=12345,
            ),
        ),
    ),
)
```

`ToolOutput` 只是 executor 到 ToolResult 的类型安全转换入口，不是第四层工具身份，也不
拥有 resolver 或存储服务。

## 媒体源与资源解析

`MediaSource` 是严格、不可变的描述值，并且必须恰好选择一种来源：

| source kind | 必需字段 | 语义 |
| --- | --- | --- |
| `resource` | `uri` | 由部署显式注册的 MediaResolver 解析；可以表示本地或共享资源 |
| `url` | `url` | Provider 可直接读取的 URL，或由受信 resolver 拉取后再编码 |
| `inline` | `base64_data` | 已规范化的 Base64 内容；只适合有界小媒体 |

所有来源都可以携带 `sha256` 和 `size_bytes`；持久恢复要求二者存在。MIME 类型放在
`ToolResultMedia` 上，不从文件扩展名或 URL 猜测。解析后必须校验实际大小、摘要、声明
MIME 与允许格式；不匹配时在 Provider I/O 前失败。

URL 不能内嵌用户名、密码、API key、签名 token 或其他 secret。需要认证、短时签名或
刷新 URL 的媒体应保存为稳定 `resource`，由部署本地 resolver 在请求边界取得当前可用
内容。易过期 URL 不能单独满足 durable restore 契约。

本地路径不是可移植资源身份。应用可以把路径交给本地 builder，但 builder 必须在结果
进入 ToolResult 前完成以下一种转换：

1. 写入应用拥有的媒体存储并产生稳定 `resource` URI；
2. 产生有界 `inline` 数据；
3. 在仅限当前部署的 direct execution 中产生由显式 MediaResolver 识别的本地 URI。

第三种形式不自动获得 Worker 或 durable 能力。保存路径只证明引用被保存，不证明恢复
进程仍能读取同一文件。需要恢复后读取同一内容时，应用必须使用不可变资源、内容寻址或
版本化引用，并验证摘要。

MediaResolver 是部署资源，不进入 Module 定义、Context、ExecutionPlan 或 ToolResult。
它负责有界读取、URL 安全策略、凭据、生命周期和关闭。Adapter 只消费解析结果，不能把
任意本地路径直接发送给 Provider，也不能绕过现有网络边界替 Provider 下载任意 URL。

## 两层能力检查

媒体请求必须同时通过两个彼此独立的检查：

1. `ModelCapabilities.modalities.input` 包含所需的 `image` 或 `video`，表示模型语义支持
   这种输入；
2. 当前 Adapter/endpoint 声明 `tool_result_content.modalities` 包含该模态，表示具体协议
   接受该媒体出现在 tool 消息中。

Adapter capability 还应声明可直接发送的 source kinds 和 MIME 限制。例如，同一个模型
可能接受 user message 图片，但某个 Chat Completions endpoint 不接受 tool message 图片；
这种配置必须在 Provider I/O 前报告为不支持。

OpenAI Compatible 不是 capability。框架不能根据 Provider 名、模型名、URL 或一次成功
响应推断所有兼容接口都支持多模态 tool content。第三方接口由应用或 Provider preset
显式声明，内置声明必须有对应的文档或测试证据。

能力不足属于当前调用的非重试 invalid-request 失败。它不触发协议自动探测，不把媒体
改写成 user message，也不为了寻找支持媒体的模型而改变 ModelGroup 顺序或静默路由。

## OpenAI Compatible 编码

支持该能力的 Chat Completions Adapter 把一个 ToolResult 编码为一个保持关联的消息：

```json
{
  "role": "tool",
  "tool_call_id": "call_read_1",
  "content": [
    {"type": "text", "text": "工具读取到的图片。"},
    {
      "type": "image_url",
      "image_url": {"url": "data:image/png;base64,..."}
    }
  ]
}
```

视频块对应：

```json
{
  "type": "video_url",
  "video_url": {"url": "data:video/mp4;base64,..."}
}
```

编码规则如下：

- text block 映射为 `type: text`；
- image/video 分别映射为 `image_url`/`video_url`；
- endpoint 允许直接 URL 时可以保留 URL，否则 Adapter 经 resolver 读取后生成 data URI；
- `name` 是否发送继续服从现有 OpenAI Compatible 兼容策略；
- 一个 ToolMessage 中的多个 ToolResult 仍分别产生自己的 `role: tool` 消息和 call ID；
- `ToolMessage.content` 的追加文本作为最后一个结果的 text block 追加，不再假定原 content
  是字符串；
- 不支持多模态 tool content 的 Adapter 收到媒体时必须失败，不能调用 `json.dumps()` 把
  数组降级为文本。

纯文本和 JSON ToolResult 保持当前 wire 行为，避免没有使用新内容类型的应用产生请求
差异。各协议可以使用自己的等价结构，但必须保持块顺序、媒体语义和工具调用关联。

## Context、持久化与恢复

ToolResult content、MediaSource 和外层 call ID 必须进入同一个 Message codec。Context、
Worker 请求、managed effect、ToolTask 最终结果和 durable history 的往返不能丢失块类型、
顺序、MIME、资源身份、摘要或 call ID。未知内容类型和未知 source kind 必须 fail closed。

媒体字节与引用分开保存：

- Context 和执行记录保存内容块及稳定 source descriptor；
- 应用媒体存储保存资源内容并负责 retention、权限和删除；
- 大段 Base64 不复制到普通日志、公开错误或重复的事件字段；
- `model.request.prepared` 保存完整的逻辑请求、媒体 source descriptor、大小和摘要，不保存
  Adapter 临时生成的 data URI；稳定 request digest 必须包含媒体内容身份和块顺序；
- effect identity 使用规范 source descriptor 和摘要，不能依赖临时文件路径或签名 URL 的
  易变查询参数。

这需要在实现时同步澄清 LLM 第一原则中的“完整 provider-neutral 请求”：完整指逻辑模型
请求及可验证媒体身份，不是 Provider transport 展开后的 Base64 副本。权威 durable 记录
必须足以重新解析资源或明确判定资源已失效。

恢复后的后续调用在发送前重新解析资源。资源缺失、过期、权限变化、摘要不匹配或目标
Worker 没有 resolver capability 时明确失败。框架不能丢弃该块、退化为文件名文本，或
假装恢复成功。

若上下文压缩器不支持投影中的媒体，压缩必须明确失败或由应用先执行显式、可审计的媒体
摘要/替换操作；Pygent 不自动删除历史媒体。token/size 估算也必须把媒体计入 Provider
限制，无法可靠估算时采用声明的保守预算或在调用前拒绝。

## 失败语义

建议使用稳定、脱敏的 reason code：

| reason code | 条件 |
| --- | --- |
| `model_input_modality_unsupported` | 模型 capabilities 不包含媒体模态 |
| `tool_result_content_unsupported` | Adapter/endpoint 不接受 tool 消息中的该模态 |
| `media_source_unsupported` | 当前 Adapter/Resolver 不支持 source kind |
| `media_source_unresolvable` | 资源不存在、过期、无权限或无法读取 |
| `media_integrity_mismatch` | size 或 sha256 与解析结果不一致 |
| `media_mime_type_invalid` | MIME 缺失、不允许或与内容不符 |
| `media_too_large` | 超过配置或 Provider 上限 |
| `media_content_invalid` | Base64、内容块或媒体结构无效 |

错误必须保留原执行和 tool call 关联，同时遵守现有 Provider-neutral、脱敏错误边界。公开
错误不包含本地绝对路径、签名 URL、原始媒体、Provider 响应体或异常链。

## 首期验收

- 图片工具返回 text + image blocks 后，模型能识别图片内容；
- 视频工具返回 text + video blocks 后，模型能识别短视频画面及顺序；
- 捕获的请求证明媒体位于 `role: tool` 的 content 数组，并保留正确 `tool_call_id`；
- 多块顺序、多 ToolResult 顺序和 `ToolMessage.content` 追加文本保持稳定；
- direct、managed、Worker、effect replay 和 durable restore 往返保持内容块与资源身份；
- 恢复后资源仍有效时可以再次请求，失效或摘要不匹配时返回上述明确错误；
- 模型支持媒体但 endpoint 不支持 tool media，以及反向情况，都在 I/O 前分别失败；
- 不支持的 Adapter 不降级为字符串或 user message；
- 既有纯文本、JSON、拒绝和失败 ToolResult 的构造与 wire 输出保持兼容；
- 日志、事件和公开错误中不重复写入大段 Base64。

第三方 `glm-5.3-flash` Chat Completions endpoint 已使用标准 `FileTools.read_image` 与
`read_video` 文件路径工具通过图片和视频验收：图片返回 `BLUE SQUARE`，256×256 的红→蓝
短视频返回 `RED THEN BLUE`。这只证明被测 endpoint；同一网关上的 `glm-5.3` 对两种
tool-media 都返回 invalid request。音频不进入首期通过条件。
扩展到 64 个模型、95 个模型×模态场景的结果见
[多模态 ToolResult Live 模型矩阵](MULTIMODAL_TOOL_RESULTS_LIVE_MATRIX.md)。

## 交付状态

首期已交付内容块、MediaSource、ToolOutput、严格 codec、effect/durable 往返、
MediaResolver、两层 capability preflight、OpenAI Compatible 编码、prepared-request
脱敏投影和 synthetic tests。标准 `read_image`/`read_video` 与可重复的图片、视频
fixture 位于 `tests/live/multimodal_tool_result_probe.py`；标准工具可在后续按应用需求提供。
