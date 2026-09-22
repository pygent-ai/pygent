# 多模态上下文保护需求

> 状态：已确认，并由当前 LLM 媒体投影契约实现。
>
> 本文只记录本轮已经讨论的上下文保护、媒体能力描述和目标模型投影需求，不展开
> Provider 逐型号参数表、Token 公式或完整媒体处理实现。

## 1. 目标

Pygent 的原生工具可以返回 canonical 图片或视频。该媒体可以是原始媒体，也可以是工具按照
自身公开契约生成的规范化媒体。Invoker 在模型调用、失败重试和模型切换时，
必须保护这些多模态内容，使已经存在的历史文件不会因为一次调用失败或目标模型不支持该
模态而丢失。

上下文必须能够回答两个问题：

1. 模型在一次成功调用中实际接收了什么媒体；
2. 历史图片或视频当前保存在哪里，后续怎样重新读取或处理。

## 2. 工具结果

工具返回 canonical 图片或视频及其可移植来源。来源可以是资源位置、URL 或 inline Base64。
如果后续需要从原始媒体重新生成其他投影，工具或外部媒体 Store 还必须保留稳定的原始资源
引用。

工具结果保留媒体的基本事实，包括：

- 媒体类型与 MIME；
- canonical 来源，以及已有的原始来源或持久化后的稳定引用；
- 已知的宽高、时长等元数据；
- 已知的大小与内容摘要。

工具本身执行成功、但某个模型无法查看媒体时，不能把整个工具结果记为执行失败。

### 2.1 Context 最小补充

本需求不新增 `Context` 顶层字段。`Context.messages` 中的 `MediaBlock.source` 表示工具
产生的 canonical media artifact，不能被某个模型专属的缩放、转码、抽帧或传输编码结果
覆盖。

允许进入 Context 的媒体来源遵守以下最小准入规则：

- 有界 inline Base64 本身是可重放内容，可以进入 Context；其摘要和大小根据解码后的媒体
  字节自动生成；
- 需要跨进程、持久化或后续重放的 resource 必须具有稳定 URI、`sha256` 和 `size_bytes`；
- 只在当前请求中直接透传的 URL 可以没有摘要；可能过期的 URL 不能作为唯一历史来源，进入
  durable Context 前必须保存稳定快照；
- 临时文件路径不能直接进入可恢复 Context，必须先转换成稳定 resource 或 inline Base64。

模型专属 Base64 是 request-local 投影，不写回 Context：

```text
Context 中的 canonical media
    ↓
Invoker 选择目标媒体投影计划
    ↓
MediaProjector 解析、校验和转换
    ↓
生成 ProjectedMedia
    ↓
Adapter 编码到目标协议并调用模型
```

切换模型时，优先从稳定原始资源生成新模型的媒体投影；没有原始资源时，从 Context 保存的
canonical media 生成。不能从前一个模型的 request-local 派生媒体继续转换。

每个真实 attempt 在 Provider I/O 前生成 prepared-request 或投影记录，保存该 attempt 将要
使用的媒体事实，包括 canonical 或原始摘要、派生摘要、派生媒体稳定引用、最终 MIME、
尺寸、编码和模型身份。attempt 是否成功由后续事件表示。这些事实不新增为 Context 字段，
不复制 inline Base64 或 Adapter 生成的 data URI，也不把 Provider 专属请求 JSON 写入
Context。

模型调用失败时，不提交失败尝试产生的 Assistant，Context 保持不变并继续正常
retry/fallback。模型调用成功时，由调用方明确追加 Assistant，原 `MediaBlock`
继续保留。

因此 Context 不新增 media store、media projections、活跃 resolver 或 Provider 请求；最小
补充是强化 `MediaSource` 的可重放要求，并由真实 attempt 的外部投影记录保存模型实际请求
使用的媒体事实。

## 3. 结构化媒体能力

现有 `modalities.input` 继续说明模型是否支持 `image` 或 `video`。为了让 Invoker 在请求前
判断 canonical 媒体能否直接使用或需要转换，模型能力还需要提供机器可读的媒体细节，而不能只写
自然语言注释。

建议表达的图片能力包括：

- 支持的 MIME；
- 最大字节数、宽、高和像素数；
- 支持的分辨率模式；
- 是否支持动画图片。

建议表达的视频能力包括：

- 是否支持原生视频输入；
- 支持的 MIME；
- 最大字节数、时长、宽、高和 FPS；
- 是否接收视频中的音频。
- 请求投影使用原生视频 URL 还是有序图像帧。

`media_input` 是 `ModelCapabilities` 的公开能力结构；旧配置可以省略该字段。当前严格
schema 如下：

```yaml
capabilities:
  modalities:
    input: [text, image, video]
    output: [text]

  media_input:
    image:
      mime_types: [image/png, image/jpeg, image/webp]
      max_bytes: 20971520
      max_width: 8192
      max_height: 8192
      max_pixels: 30000000
      resolution_modes: [low, high, original]
      animated: false

    video:
      native: true
      mime_types: [video/mp4]
      max_bytes: 104857600
      max_duration_seconds: 300
      max_width: 1920
      max_height: 1080
      max_fps: 30
      audio: true
      delivery_modes: [video_url]
```

无法从可靠资料确认的限制使用 `null` 表示未知，不能解释为无限制。
`delivery_modes` 可使用 `video_url` 或 `image_frames`，并按具体模型记录；不能因为同一
Provider 中某个模型需要帧序列，就把整个模型族都改成帧序列。

模型媒体能力与协议传输能力必须分开。Endpoint 传输能力使用
`MediaTransportCapabilities`，由 Adapter/endpoint 装配提供，并声明：

- tool result 中支持的媒体模态；
- 支持的来源类型；
- 支持 Base64、URL 或文件上传中的哪些形式；
- inline 内容的最大字节数；
- wire 层接受的图片和视频 MIME。

以下是能力结构示意，不表示当前存在独立的顶层 `media_transport` YAML 配置：

```yaml
media_transport:
  tool_result_modalities: [image, video]
  source_kinds: [inline, resource, url]
  inline_encodings: [base64]
  max_inline_bytes: 20971520
  image_mime_types: [image/png, image/jpeg]
  video_mime_types: [video/mp4]
  file_upload: false
  remote_url: true
```

Invoker 使用模型媒体能力与 Adapter/endpoint 传输能力的交集作为本次请求的有效媒体能力。
能力记录可以附带资料来源、资料版本和核验时间，但运行时不能解析自然语言注释来决定行为。

## 4. 请求前分类

Invoker 必须在 Provider I/O 前区分以下三类情况。

### 4.1 确定的模型或协议能力问题

模型不支持对应模态、endpoint 不接受 tool result 媒体，或者已知限制明确无法满足时，
Invoker 不发送探测请求。它可以继续选择兼容模型，或者为当前模型生成 `not_viewed` 的历史
媒体说明。

### 4.2 媒体资源问题

Base64 无效、资源无法解析、MIME 不匹配、文件确实不存在或媒体超过无法转换的硬限制，
必须报告真实的媒体资源原因，不能说成模型不支持。

### 4.3 模型响应或服务问题

超时、限流、认证失败、Provider 不可用、流中断或响应格式无效继续使用现有模型失败处理。
这类失败不能自动推断为模型不支持媒体，也不能永久修改模型能力记录。

如果 Provider 明确返回媒体模态或格式不受支持，Adapter 可以把它归一为本次调用的媒体能力
冲突，并记录配置与实际服务不一致；Pygent 不通过主动发送请求来探测能力。

## 5. 面向目标模型的媒体投影

Invoker 根据本次有效媒体能力选择 provider-neutral 媒体投影计划；可替换的 MediaResolver
解析媒体资源，MediaProjector 按计划执行媒体转换，Adapter 只负责把投影编码到目标协议：

```text
稳定原始资源或 Context 中的 canonical media
    ↓
Invoker：模型媒体能力 + Adapter/endpoint 传输能力
    ↓
MediaProjectionPlan
    ↓
MediaProjector：保持原样、缩放、转码或抽帧
    ↓
ProjectedMedia
    ↓
Adapter：Base64、URL 或目标协议支持的文件引用
    ↓
模型请求
```

Base64 是支持的传输形式之一，不是唯一形式。图片和有界小媒体可以转换为 Base64；大视频
在目标协议支持时可以使用文件上传、文件引用或 URL。

### 5.1 Base64 解析与模型调用

当目标模型和 endpoint 支持 inline Base64 时，Pygent 必须在 Invoker 编排下完成以下步骤，
而不是要求业务工具构造 Provider 请求：

1. Invoker 根据目标模型媒体能力与 Adapter/endpoint 传输能力选择投影计划；
2. MediaProjector 读取稳定原始资源或工具结果中的 canonical media；inline 来源先解码
   Base64，resource 来源通过 MediaResolver 解析为字节；
3. MediaProjector 校验 Base64、真实字节数、内容摘要、MIME 和可取得的宽高、时长等元数据；
4. 媒体符合要求时直接使用当前字节；不符合但允许转换时，执行所需的缩放、转码或抽帧；
5. MediaProjector 生成 ProjectedMedia，并在需要 inline 传输时把最终媒体字节重新编码为
   规范 Base64；
6. Adapter 把该 Base64 放入目标协议的原生图片或视频字段，并发起模型调用。

```text
MediaBlock
    ↓
Invoker 选择 MediaProjectionPlan
    ↓
MediaProjector 解析来源或解码 Base64
    ↓
校验媒体事实并保持原样或转换
    ↓
ProjectedMedia
    ↓
Adapter 构造原生请求
    ↓
调用目标模型
```

inline Base64 的 SHA-256 对解码后的媒体字节计算，不对 Base64 字符串计算。需要持久重放的
resource 在解析后校验字节摘要和大小。每份发生字节变化的派生媒体使用自己的摘要，不能用
canonical 或原始媒体摘要代替。

如果 Base64、媒体内容或资源引用校验失败，Pygent 在 Provider I/O 前按媒体资源问题处理。
如果目标 endpoint 不支持 inline Base64，Invoker 按有效传输能力选择 URL、文件上传或文件
引用，不能仍然强制发送 Base64。

切换模型后优先从稳定原始资源生成新模型的媒体投影；没有原始资源时，从 Context 中的
canonical media 生成。不能把前一个模型已经缩放或有损转换的 request-local 派生媒体继续
转换。例如：

```text
稳定原始图片或 canonical 图片
  ├─ 模型 A → JPEG 2048px → Base64
  ├─ 模型 B → PNG 原分辨率 → Base64
  └─ 模型 C → JPEG 768px → Base64
```

派生媒体可以缓存。缓存身份至少包含 canonical 或原始媒体摘要、目标能力、处理策略和转换器
版本；缓存结果记录派生媒体自己的摘要、大小、MIME 和稳定引用。这样可以避免模型 fallback
或切回时重复转换，同时保证能力或转换规则更新后不会错误复用旧结果。

## 6. 成功模型调用

模型调用成功不会把 request-local 媒体投影写回 Context。Context 继续保留原
`MediaBlock`，调用方只按现有规则提交成功的 Assistant。每个真实 attempt 的
prepared-request trace 记录该 attempt 使用的媒体投影身份和转换事实。

成功后的模型上下文形态为：

```text
User
Assistant(tool_call=image_tool)
ToolResult(canonical media source)
Assistant(model answer)
```

模型实际接收的 Base64 或其他派生表示属于 attempt trace 和派生媒体存储，不替换上述
ToolResult。

## 7. 模型调用失败与正常回退

媒体能力匹配在请求前完成。进入真实调用的模型已经被判定能够接收本次媒体，因此超时、
限流、认证失败、Provider 不可用、流中断或响应格式无效等调用失败，统一沿用现有
retry/fallback，不增加多模态专用恢复流程。

是否继续 retry/fallback 完全服从现有 RetryPolicy、输出可见性、`model.output.reset`、
deadline、清理确认和 `OUTCOME_UNKNOWN` 规则。

模型调用失败时：

1. 失败尝试产生的 Assistant 内容不能提交到历史；已经通过流式事件可见的暂存输出继续遵守
   现有 reset 和终态规则；
2. Context 中的 canonical 图片或视频以及已有的稳定原始资源引用继续保留；
3. 同一模型 retry 时继续使用该模型的有效媒体投影；
4. fallback 到新模型时，先检查新模型与 endpoint 的媒体能力；
5. 新模型兼容时，优先从稳定原始资源生成它需要的投影；没有原始资源时，从 canonical media
   生成；
6. 新模型不兼容时，不创建真实 attempt，继续选择下一个兼容候选；
7. 成功完成后，由调用方按现有 Agent 提交规则提交新的 Assistant。

普通调用失败不能把媒体改写成 `not_viewed`，也不能表述为“模型不支持图片或视频”。
Base64 或稳定媒体引用的保存属于上下文可重放要求，不属于模型失败恢复机制。

如果能力配置声明支持媒体，但 Provider 明确返回模态或格式不受支持，Adapter 将该结果
归一为本次调用的媒体能力冲突。Invoker 随后重新执行候选能力筛选，不把它当作普通超时或
服务失败，也不自动永久修改模型能力记录。

## 8. 模型切换

中途切换模型时，Invoker 必须检查目标模型能否读取历史中的图片或视频。

如果目标模型支持该媒体，但限制或传输形式不同，Invoker 选择新的媒体投影计划。
MediaProjector 优先从稳定原始资源生成；没有原始资源时，从 Context 中的 canonical media
生成该模型接受的 Base64、URL 或文件引用，并将新投影发送给模型。

如果目标模型不能查看某类历史媒体，或者媒体无法转换到其支持范围：

- Context 中的 canonical media 和已有的稳定原始资源引用继续保留；
- 发送给该模型的上下文不再附带它无法处理的媒体内容；
- 对应工具结果明确说明当前模型没有查看该图片或视频的能力；
- 工具结果同时告诉模型历史文件保存在哪里，以便后续通过工具重新读取或处理；
- Invoker 使用处理后的上下文重新调用模型并获取新的 Assistant。

图片示例：

```json
{
  "type": "image",
  "status": "not_viewed",
  "reason_code": "model_input_modality_unsupported",
  "media_ref": "media://<stable-reference>",
  "message": "当前模型没有查看图片的能力。历史图片保存在 media://<stable-reference>。"
}
```

视频继续使用 `model_input_modality_unsupported`，由媒体类型生成“当前模型没有查看视频的
能力”的明确说明。

不能使用“文件不存在”“媒体不可用”等容易误解为资源丢失的表述。只有文件确实不存在、
损坏或引用无法解析时，才能报告对应的资源错误。

## 9. 上下文保存规则

多模态上下文遵守以下提交规则：

- 成功完成的模型调用才提交对应 Assistant；
- 失败尝试的 Assistant 不进入历史；
- 当前 Context 投影中的 `MediaBlock` 必须可重放；
- 成功请求的 request-local 媒体投影不写回 Context；
- 失败或切换模型不会删除 canonical media 或已有的稳定原始资源引用；
- 模型调用失败沿用正常 retry/fallback，不创建多模态专用恢复上下文；
- 换模型时优先从稳定原始资源生成目标模型投影；没有原始资源时，从 canonical media 生成，
  不级联转换 request-local 派生媒体；
- 确定的能力问题在请求前处理，不通过真实请求探测；
- 模型响应失败不能自动归类为模型能力问题；
- 模型可见结果必须区分“当前模型不能查看”和“文件确实不存在”；
- 历史工具调用的 `call_id`、工具结果和媒体引用保持关联。

这些规则只保证当前模型投影中的 `MediaBlock` 可重放。Context 经压缩或显式 replacement
后，长期完整历史和媒体资源位置由外部 History/Media Store 保存。

## 10. 验收场景

### 10.1 图片调用成功

- 工具返回图片；
- 模型成功查看并回答；
- Context 继续保留工具返回的 canonical 图片；
- attempt trace 能够标识本次成功请求使用的图片投影。

### 10.2 视频调用成功

- 工具返回视频；
- 模型成功查看并回答；
- Context 继续保留工具返回的 canonical 视频；
- attempt trace 能够标识本次成功请求使用的视频投影。

### 10.3 携带媒体的模型调用失败

- 失败尝试的 Assistant 不提交；
- Context 中的 canonical 图片或视频仍然存在；
- inline Base64 的摘要和大小根据解码后的媒体字节生成；
- 需要跨进程、持久化或后续重放的媒体具有稳定引用；
- 同模型 retry 继续使用该模型的有效媒体投影；
- fallback 候选重新执行请求前能力判断；
- 兼容候选优先从稳定原始资源生成自己的媒体投影；没有原始资源时，从 canonical media
  生成；
- 不兼容候选不创建真实 attempt；
- 响应失败不被错误记录为模型能力不足；
- 成功候选产生并提交新的 Assistant。

### 10.4 切换到支持图片但限制不同的模型

- canonical 图片和已有的稳定原始资源引用不丢失；
- Invoker 依据新模型与 endpoint 的有效能力选择投影计划，MediaProjector 生成图片投影；
- 新请求使用目标协议支持的 Base64、URL 或文件引用；
- 新投影优先从稳定原始资源生成；没有原始资源时，从 canonical 图片生成，不从上一个模型
  的 request-local 派生图片继续转换。

### 10.5 切换到不支持图片的模型

- canonical 图片和已有的稳定原始资源引用不丢失；
- 新模型不接收无法处理的图片附件；
- 工具结果明确说明当前模型没有查看图片的能力；
- 工具结果提供历史图片位置；
- 新模型能够继续生成 Assistant。

### 10.6 切换到不支持视频的模型

- canonical 视频和已有的稳定原始资源引用不丢失；
- 新模型不接收无法处理的视频附件；
- 工具结果明确说明当前模型没有查看视频的能力；
- 工具结果提供历史视频位置；
- 新模型能够继续生成 Assistant。

### 10.7 请求前能力判断

- 已知不支持的图片、视频、格式或硬限制在 Provider I/O 前识别；
- Invoker 不发送能力探测请求；
- 媒体资源错误与模型能力问题使用不同原因；
- 超时、限流和 Provider 响应错误按现有 RetryPolicy、输出可见性、reset、deadline、清理确认
  和 `OUTCOME_UNKNOWN` 规则决定是否 retry/fallback。

### 10.8 Base64 解析与调用

- inline Base64 在请求前被解码并校验；
- inline Base64 的 SHA-256 对解码后的媒体字节计算，不对 Base64 字符串计算；
- 需要持久重放的 resource 在解析后校验字节摘要和大小；
- resource 或 URL 在需要 inline 传输时被解析为媒体字节；
- Pygent 依据目标模型与 endpoint 能力决定保持原样或执行转换；
- 最终媒体字节被编码为目标请求使用的 Base64；
- 每份发生字节变化的派生媒体具有自己的摘要；
- Adapter 将 Base64 写入对应协议的原生媒体字段并调用模型；
- 无效 Base64、MIME、摘要或媒体内容不会进入 Provider I/O；
- 不支持 inline Base64 的 endpoint 使用其声明支持的其他传输形式。

## 11. 本草案不决定的内容

本文仍不决定：

- 媒体存储由 Pygent 内置还是由应用提供；
- Base64 是直接持久化还是转换为内容寻址的媒体引用；
- 各 Provider、模型和 endpoint 的具体能力参数值；
- 派生媒体缓存是否由 Pygent 内置。
