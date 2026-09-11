# Provider Options 规范

## 定义

`ModelSpec.provider_options` 保存 Provider 私有、稳定且可持久化的模型生成语义。它必须是严格 JSON object，构造时防御性复制并递归冻结，且不出现在 `repr` 中。

```python
ModelSpec(
    provider="deepseek",
    model_id="deepseek-v4-flash",
    protocol="openai_chat_completions",
    provider_options={"thinking": {"type": "disabled"}},
    capabilities=capabilities,
)
```

修改选项必须创建新的 `ModelSpec`，固定 Layer 会产生新的定义摘要，动态部署会产生新的 profile snapshot。

## 允许与禁止

允许的值必须同时满足：

- 是 Provider 生成语义；
- 能稳定序列化为 JSON；
- 不包含 secret；
- 不覆盖 Pygent 保留请求字段。

以下内容属于部署资源或执行策略，不能进入 `provider_options`：

- API key、认证头和其他 credential；
- base URL、endpoint、代理、TLS/证书配置；
- client、连接池或活跃资源；
- retry、fallback、deadline；
- streaming transport 开关；
- `model`、消息、工具和框架生成的请求字段。

连接信息使用 `ModelConnection`，streaming transport 使用 `ModelCapabilities.streaming.output`，retry 使用 Layer 的 `RetryPolicy`。

## 校验职责

`ModelSpec` 只校验 provider-neutral 事实。Adapter 可实现公开 SPI：

```python
class ModelProviderSpecValidator(Protocol):
    def validate_model(self, model: ModelSpec) -> None: ...
```

Invoker 和动态 profile 发布在 Provider I/O 前调用该校验。非空选项配合未实现 validator 的第三方 Adapter 时 fail closed。DeepSeek 与 Alibaba Cloud Token Plan 的私有 schema 由 OpenAI-compatible Adapter 根据 `ModelSpec.provider` 校验，而 Adapter 本身仍按 `protocol == "openai_chat_completions"` 分派。

## 投影与安全

Provider options 进入以下确定性投影：

- Module definition/config digest；
- effect request；
- profile snapshot 和 admission；
- Worker codec。

这些投影保存冻结的配置值，不包含连接和 credential。公开模型事件不携带 Provider options；prepared request 只包含经过 allowlist 的 provider-neutral 请求快照。

旧 profile 数据的字段结构不同，SQLite 和 Worker 解码器必须明确拒绝，不能省略选项或尝试兼容降级。

## OpenAI-compatible 投影

Adapter 先生成框架拥有的请求字段，再合并通过校验的 Provider options。冲突字段、非法 token limit、非 JSON 数值和嵌套 secret 名称在发送前拒绝。每个 fallback 模型只使用自己的 `ModelSpec.provider_options`。

Alibaba Cloud Token Plan 严格接受 `enable_thinking`、`preserve_thinking`、`reasoning_effort`、`thinking`、`thinking_budget` 和 `tool_stream`。布尔字段不接受整数替代；`reasoning_effort` 使用固定枚举；`thinking` 只能是 `adaptive` 或 `disabled` 的单字段对象；`thinking_budget` 是非负整数。未知字段在 I/O 前拒绝。

## Anthropic Messages 投影

`AnthropicMessagesAdapter` 只接受 `thinking`、`output_config`、`service_tier` 和 `stop_sequences`。`thinking` 支持 `disabled`、带显式 token budget 的 `enabled`，以及 `adaptive`；`output_config` 只接受 `effort`。未知字段、非法组合和越界值在 Provider I/O 前拒绝。

Anthropic Messages 的 `GenerationConfig.max_output_tokens` 是必填正整数。它是 Provider-neutral 生成参数，不放入 `provider_options`。结构化输出由 Pygent 的 response schema 投影到 `output_config.format`，用户不能通过 Provider options 覆盖该保留字段。

Anthropic 官方 thinking block 必须带非空 signature。使用相同 Messages wire contract 的其他 Provider 可以省略 signature；Adapter 在 continuation 中保留实际收到的形状，并只在 Provider 与 protocol 同时匹配时回传。兼容规则不允许放宽 Anthropic 官方响应校验。
