# LLM 第一原则

## 配置边界

Pygent 把模型信息分成两个不可混合的投影：

- `ModelSpec` 是纯模型语义，包含 `provider`、`model_id`、`protocol`、`provider_options` 和完整 `capabilities`；
- `ModelConnection` 是部署资源配置，包含 endpoint、credential 引用和 TLS 策略。

`ModelEntry` 用本地配置名包装一个 `ModelSpec`。这个名称是 client 绑定、fallback、资源映射和诊断使用的稳定 `model_key`，不属于底层模型语义。`ModelGroup` 保存有序的 `ModelEntry`，顺序就是 fallback 顺序。

`ModelConfig.from_mapping()` 一次产生两个不可变投影：`models`/`model_groups` 与 `connections`。解析不读取环境变量、不创建 client，也不注册 Runtime。真实 credential 只在部署资源装配时解析，不能进入定义摘要、事件、持久化数据或 `repr`。

## 能力是用户声明的事实

每个 `ModelSpec` 必须携带完整的 `ModelCapabilities`：

- `modalities.input`、`modalities.output`；
- `streaming.text`；
- `tools.call`、`tools.choice`、`tools.parallel`；
- `structured_output.json_object`、`structured_output.json_schema`；
- `reasoning.supported`、`reasoning.controllable`；
- `limits.context_tokens`、`limits.max_output_tokens`。

Pygent 提供 `ProviderCatalog`、`ModelCapabilityCatalog` 和 `CapabilityPresetCatalog` 三份不可变目录。它们只提供配置数据，不持有 client，不参与调用路由，也不联网更新。UI 可以选择内置条目后把完整值写入用户配置；用户也可以逐项修改或通过 `from_mapping()` 加载外部目录。

Capabilities 不用于自动选模型。调用与声明不一致时，框架发出 `model.capability.warning`，但仍调用用户指定的模型。第一版检查文本输入/输出、工具调用、显式 `tool_choice`、JSON Schema 和输出 token 上限。匹配路径不构造警告事件；retry 不重复警告，fallback 只在实际进入对应模型时检查。

`capabilities.streaming.text` 直接决定使用 streaming 或 non-streaming transport，不存在第二份 transport capability 配置。

## Provider、protocol 与连接解耦

Provider 是开放字符串。Provider preset 只提供 UI/配置默认值；Adapter 按 `protocol` 注册；client 按 `ModelEntry.name` 绑定。多个 Provider 可以共享同一个 protocol Adapter。

第一版内置目录只包含 DeepSeek 官方：

- protocol：`openai_chat_completions`；
- base URL：`https://api.deepseek.com`；
- credential 环境变量：`DEEPSEEK_API_KEY`；
- 模型：`deepseek-v4-flash`、`deepseek-v4-pro`。

Anthropic-compatible 仅是目录 schema 可扩展的 protocol 字符串；第一版不提供 preset 或 Adapter。

Provider 私有生成语义放在 `ModelSpec.provider_options`。连接、secret、认证头、代理、TLS、retry、deadline、stream 开关和框架保留请求字段不能放入其中。第三方 Adapter 只有实现 `ModelProviderSpecValidator` 才能接受非空选项。

## Layer 与执行

`ModelCallLayer` 必须且只能接收 `model` 或 `model_group` 之一。单模型会在 Layer 内规范化为同名的单条目 `ModelGroup`，所以 direct 与 managed execution 使用相同身份和语义。

Direct 模式显式传入部署阶段构造的 `ModelInvoker`，调用方负责 client 生命周期。Managed 模式可省略 invoker，由 Runtime 使用现有 invoker 注册或 resource resolver。两种模式都把完整模型语义保存在 Layer；Runtime 不通过字符串重新查找模型配置。

Invoker 按 `ModelGroup.models` 顺序尝试模型，并在每个模型内执行 retry。公开 attempt、请求快照和事件统一使用 `model_key`。Provider 原始载荷、异常消息和 secret 不跨越公开边界。

Managed 模型并发只由 Binding 的 `model_capacity` 控制。Layer 调用无参数 `model_permit()`；`ModelSpec` 和 `ModelGroup` 不声明容量。

## 动态模型组

`ModelGroup.deferred(name=...)` 声明 managed 部署需求。应用通过 Binding 上的模型组句柄发布 profile：每个 profile 接收有序 `ModelEntry`、可选 resident invoker 或可重建资源，然后生成不可变 snapshot。单独的 fallback 参数不存在。

Profile snapshot、admission、Worker codec、effect 和 SQLite 持久化均保存新模型投影。旧 profile 数据不做双格式读取，遇到旧字段会明确拒绝。资源 bundle 以 `model_key` 映射资源，同时保留 `ModelResourceRef` 的 ownership 和 domain 语义。

Runtime 继续负责资源租约、取消、deadline、durability 和 Worker 执行，不解释 capabilities，也不改变 fallback 顺序。

## 目录发布

PyPI wheel/sdist 携带三份 JSON 快照。GitHub Release 附加相同 JSON 和 `SHA256SUMS`。框架默认只读取包内数据，不在模型调用时访问 GitHub。应用若下载新目录，必须自行校验后显式调用对应的 `from_mapping()`；新目录不会自动修改已保存的 `ModelSpec`。

本契约是新的唯一模型配置方式，不提供旧公开类型别名、字符串简写或兼容解析路径。
