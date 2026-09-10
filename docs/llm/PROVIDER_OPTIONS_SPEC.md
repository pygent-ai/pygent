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

连接信息使用 `ModelConnection`，streaming transport 使用 `ModelCapabilities.streaming.text`，retry 使用 Layer 的 `RetryPolicy`。

## 校验职责

`ModelSpec` 只校验 provider-neutral 事实。Adapter 可实现公开 SPI：

```python
class ModelProviderSpecValidator(Protocol):
    def validate_model(self, model: ModelSpec) -> None: ...
```

Invoker 和动态 profile 发布在 Provider I/O 前调用该校验。非空选项配合未实现 validator 的第三方 Adapter 时 fail closed。DeepSeek 私有 schema 由 OpenAI-compatible Adapter 根据 `ModelSpec.provider == "deepseek"` 校验，而 Adapter 本身仍按 `protocol == "openai_chat_completions"` 分派。

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
