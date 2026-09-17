# LLM 配置与执行

本页冻结下一版实现必须收敛的配置契约；当前实现状态以 [LLM SDK](SDK.md) 顶部说明为准。

工具结果中的图片和视频如何独立检查模型模态与协议 tool-message 能力，以及如何编码到
Provider wire，见[多模态工具结果 Proposal](../tool/MULTIMODAL_TOOL_RESULTS_PROPOSAL.md)。
当前 OpenAI Chat Completions compatible Adapter 的显式装配方式见 [LLM SDK](SDK.md#多模态工具消息)。

## 配置边界

Pygent 的公开配置固定分为 Connection、Model 和 ModelGroup 三层：

- `ConnectionConfig` 表示一份服务账号配置，包含开放的 Provider 标识、credential 引用、TLS/代理策略，以及按 protocol 保存的 endpoint；
- Model 表示一个已启用模型，以用户定义的 `model_key` 标识，引用 `connection_key` 和其中一个 protocol，保存服务端实际接受的 `model_id`、Provider 私有选项和完整 capabilities；
- `ModelGroup` 保存有序的已启用模型，顺序就是普通调用的 fallback 顺序。

解析后，Model 的 `connection` 引用与所选 protocol endpoint 形成部署资源投影；Connection 的 Provider、Model 的 model ID 与 protocol、Provider 私有选项和 capabilities 形成完整 `ModelSpec`。因此用户不在 Model 中重复填写 Provider，但 `ModelSpec.provider` 仍是完整模型语义的一部分。

`ModelEntry` 用本地配置名包装一个 `ModelSpec`。这个名称是 client 绑定、fallback、资源映射和诊断使用的稳定 `model_key`，不属于底层模型语义。`ModelGroup` 保存有序的 `ModelEntry`，顺序就是 fallback 顺序。

三层配置值分别提供严格的 `from_mapping()` 与可往返的 `to_mapping()`，应用负责把普通 Mapping 保存到自己的 YAML、JSON、数据库或配置中心。Pygent 不提供配置存储后端。`ModelConfig.from_mapping()` 解析三层 Mapping 后委托给 `ModelConfig.from_components()`；应用已经持有解析值时可以直接使用后者。两条入口产生相同的不可变 `connections`、`models` 和 `model_groups`，并在内部保存模型到 Connection 的部署关联。`config.connection_for(model_key)` 是 direct invoker 和 managed resolver 查询已选 protocol endpoint 的唯一公开方法；内部关联不是第四段用户配置。解析不读取环境变量、不创建 client，也不注册 Runtime。真实 credential 只在部署资源装配时解析，不能进入定义摘要、事件、持久化数据或 `repr`。

## 能力是用户声明的事实

每个 `ModelSpec` 必须携带完整的 `ModelCapabilities`：

- `modalities.input`、`modalities.output`；
- `streaming.output`；
- `tools.call`、`tools.choice`、`tools.parallel`；
- `structured_output.json_object`、`structured_output.json_schema`；
- `reasoning.supported`、`reasoning.controllable`；
- `limits.context_tokens`、`limits.max_output_tokens`。

Pygent 提供 `ProviderCatalog`、`ModelCapabilityCatalog` 和 `CapabilityPresetCatalog` 三份不可变目录。它们只提供配置数据，不持有 client，不参与调用路由，也不联网更新。UI 可以选择内置条目后把完整值写入用户配置；用户也可以逐项修改或通过 `from_mapping()` 加载外部目录。

Capabilities 不用于自动选模型。调用与声明不一致时，框架发出 `model.capability.warning`，但仍调用用户指定的模型。第一版检查文本输入/输出、工具调用、显式 `tool_choice`、JSON Schema 和输出 token 上限。匹配路径不构造警告事件；retry 不重复警告，fallback 只在实际进入对应模型时检查。

模态使用 `text`、`image`、`audio`、`video` 四个封闭值，`streaming.output` 必须是 `modalities.output` 的子集。`limits` 中无法由官方资料确认的值保存为 `null`。当前文本调用检查 `"text" in capabilities.streaming.output` 来决定 streaming 或 non-streaming transport，不存在第二份 transport capability 配置。

## Provider、protocol 与连接解耦

Provider 是 Connection 上的开放字符串。Provider preset 只提供 UI/配置默认值；一个 Connection 可以保存多个 protocol endpoint，Model 必须从其 Connection 已配置的 protocol 中选择一个。Adapter 按 `protocol` 注册；client 按 `(connection_key, protocol)` 创建和复用，再按 `ModelEntry.key` 绑定给 Invoker。多个 Provider 可以共享同一个 protocol Adapter。

内置协议使用 `BuiltinModelProtocol` 表达当前由 Pygent 实现的 wire contract：

- `OPENAI_CHAT_COMPLETIONS`：`openai_chat_completions`；
- `OPENAI_RESPONSES`：`openai_responses`；
- `ANTHROPIC_MESSAGES`：`anthropic_messages`；
- `GEMINI_GENERATE_CONTENT`：`gemini_generate_content`。

`ModelSpec.protocol` 仍保存开放字符串，第三方 Adapter 可以定义自己的 protocol。内置 Provider 目录按 protocol 提供 Connection 默认值，当前覆盖 DeepSeek、Anthropic、OpenAI、Google Gemini、Alibaba Cloud Model Studio、智谱、Moonshot、MiniMax、火山引擎和 xAI；Alibaba Cloud Token Plan 作为独立 Provider。一个 Provider 可以提供多个协议，例如 OpenAI 同时提供 Chat Completions 与 Responses，DeepSeek、Moonshot、MiniMax 和 Token Plan 同时提供 OpenAI Chat Completions 与 Anthropic Messages。UI 选择 Provider 时把需要的 protocol endpoint 填入 Connection；目录只提供 base URL、credential 环境变量名和表单 schema，不提供或读取真实 API key。

内置能力目录按 `(provider, model_id, protocol)` 区分同一模型的不同服务与协议。目录中的模型身份和能力来自对应官方 Provider 资料；网关别名和外部搜索服务不进入生产目录。OpenAI Chat Completions、OpenAI Responses、Anthropic Messages 与 Gemini Generate Content 由内置 Adapter 执行；仅进入目录但没有内置 Adapter 的媒体协议，需要应用自行装配对应 Adapter 后才能调用。

Provider 私有生成语义放在 `ModelSpec.provider_options`。连接、secret、认证头、代理、TLS、retry、deadline、stream 开关和框架保留请求字段不能放入其中。第三方 Adapter 只有实现 `ModelProviderSpecValidator` 才能接受非空选项。

Anthropic Messages 请求必须由 `GenerationConfig.max_output_tokens` 提供正整数，没有框架默认值。需要工具循环回传的 Provider 私有 thinking/reasoning 状态保存在 `AIMessage.continuation`；该值记录实际生产它的 `model_key`，并只交给 `model_key`、Provider、model ID 与 protocol 都匹配的后续请求。工具结果返回后，模型组优先续接原生产模型；如果该模型失败，后续 fallback 仍接收完整的 assistant tool call 与 tool result 历史，但不接收原模型的私有 continuation。其原始 `data` 不进入公开事件、请求摘要或 `repr`。Anthropic 官方 thinking block 必须携带 signature；其他 Anthropic-compatible Provider 可以返回无 signature 的 thinking block，Adapter 会保持原形回传。OpenAI Chat Completions 不维护 Provider 白名单：响应实际携带合法 `reasoning_content` 时才创建对应 continuation。

## Layer 与执行

`ModelCallLayer` 必须且只能接收 `model` 或 `model_group` 之一。单模型会在 Layer 内规范化为同名的单条目 `ModelGroup`，所以 direct 与 managed execution 使用相同身份和语义。

Direct 模式显式传入部署阶段构造的 `ModelInvoker`，调用方负责 client 生命周期。Managed 模式可省略 invoker，由 Runtime 使用现有 invoker 注册或 resource resolver。两种模式都把完整模型语义保存在 Layer；Runtime 不通过字符串重新查找模型配置。

Invoker 通常按 `ModelGroup.models` 顺序尝试模型，并在每个模型内执行 retry。工具 continuation 续接时，先尝试组内身份匹配的原生产模型，其余模型按原组顺序 fallback。公开 attempt、请求快照和事件统一使用 `model_key`。Provider 原始载荷、异常消息和 secret 不跨越公开边界。

`ModelCallLayer` 当前结果契约是文本与 ToolCall。能力目录可以描述图像、音频、视频和 embedding，但不代表文本协议 Adapter 能消费这些返回；内置文本 Adapter 遇到非文本响应部件会明确拒绝，专用媒体协议必须由对应 Adapter 执行。

Managed 模型并发只由 Binding 的 `model_capacity` 控制。Layer 调用无参数 `model_permit()`；`ModelSpec` 和 `ModelGroup` 不声明容量。

## 动态模型组

`ModelGroup.deferred(name=...)` 声明 managed 部署需求。应用通过 Binding 上的模型组句柄发布 profile：每个 profile 接收有序 `ModelEntry`、可选 resident invoker 或可重建资源，然后生成不可变 snapshot。单独的 fallback 参数不存在。

Profile snapshot、admission、Worker codec、effect 和 SQLite 持久化均保存新模型投影。旧 profile 数据不做双格式读取，遇到旧字段会明确拒绝。资源 bundle 以 `model_key` 映射资源，同时保留 `ModelResourceRef` 的 ownership 和 domain 语义。

Runtime 继续负责资源租约、取消、deadline、durability 和 Worker 执行，不解释 capabilities，也不改变 fallback 顺序。

## 目录发布

PyPI wheel/sdist 携带三份 JSON 快照。GitHub Release 附加相同 JSON 和 `SHA256SUMS`。框架默认只读取包内数据，不在模型调用时访问 GitHub。应用若下载新目录，必须自行校验后显式调用对应的 `from_mapping()`；新目录不会自动修改已保存的 `ModelSpec`。

本契约是新的唯一模型配置方式，不提供旧公开类型别名、字符串简写或兼容解析路径。
