# 多模态降级清单

> 状态：公开契约的降级点登记。每项记录当前行为、丢失的信息与恢复条件；
> 模型与厂商能力补齐后按条件恢复，恢复时删除对应条目并更新目录。

本文登记 Pygent 多模态链路中所有已知的降级、缺口与受限行为。降级一律显式
（not_viewed 注记、明确报错或文档化边界），不允许静默丢弃。通用降级机制
（媒体按模型能力门控、不支持时转 not_viewed 文本）见
[多模态工具结果设计](../tool/MULTIMODAL_TOOL_RESULTS_PROPOSAL.md)。

## 1. chat 协议 assistant 音频重放降级

- **现状**：`openai_chat_completions` 模型（gpt-audio、qwen3.5-omni）本轮生成的
  音频会进入 `AIMessage.media` 并持久化，但**下一轮不会作为音频回传给模型**；
  历史重放时降级为 not_viewed 文本注记。语音模型听不见自己上一轮说过什么，
  多轮语音连续性不成立。
- **原因**：官方续接契约（assistant 消息按 audio id 回传）尚未查证实现。
- **恢复条件**：确认官方 assistant audio id 回传 wire 后，按 `ModelContinuation`
  或 assistant media replay 实现，并补 conformance 场景。

## 2. chat 协议音频输入编码缺口

- **现状**：内置 `OpenAICompatibleAdapter` 的 media transport 默认关闭，用户/
  工具结果的音频输入需显式装配 `MediaTransportCapabilities(modalities=("audio",))`；
  且当前 `_encode_media_part` 对 audio 块产生 `audio_url` 形状，官方 chat 端点
  期望 `input_audio`（`{data, format}`，Alibaba 兼容端点使用 data-uri 变体）。
  显式开启 audio transport 的部署会把用户音频编码为官方不接受的形状。
- **恢复条件**：实现 `input_audio` 编码（openai 形状 + Alibaba data-uri 变体，
  经 provider policy 分派），并补 audio_input synthetic 与 conformance 覆盖。

## 3. 输出媒体仅接受 inline 来源

- **现状**：模型事件流中的媒体部件只接受 inline base64
  （`_media_block_from_part`）；返回 URL 的 provider（任务型视频等）必须由
  client 下载转内联。大体积媒体会整体进入事件与历史。
- **恢复条件**：媒体部件 url 来源契约（涉及 wire codec、事件与脱敏投影）。

## 4. 任务型视频取消不止损

- **现状**：`ark_video_generation` 的远端任务在本地取消后继续运行并计费；
  Ark 返回 409（running 任务禁止删除）。
- **恢复条件**：厂商支持任务取消接口。

## 5. 目录中仍无内置 Adapter 的媒体协议

- **现状**：`dashscope_multimodal_generation`、`dashscope_video_generation`、
  `dashscope_speech_synthesis`、`dashscope_realtime`、`dashscope_speech_recognition`、
  `openai_audio_speech`、`openai_embeddings`、`openai_realtime` 仍为
  catalog-only（声明 + 探针），调用需应用自行装配 Adapter。
- **恢复条件**：逐协议实现 Client/Adapter 并官方实测（参照 `openai_images` 与
  `ark_video_generation` 的做法）。

## 6. 实测记录：qwen-omni-turbo AZ 路由 audio_output

- **现状**：2026-10-16 AZ conformance 实测，`qwen-omni-turbo` 的
  `audio_output` 场景返回 `invalid_request`（流式音频参数被 AZ 网关路由拒绝）。
  同日 `gpt-audio` 的 `audio_output` / `audio_input` 均通过。探针请求构造
  与 Pygent codec 无关，属网关路由侧问题。
- **恢复条件**：AZ 网关路由修复，或获得 DASHSCOPE 官方 key 直测后修正声明。
