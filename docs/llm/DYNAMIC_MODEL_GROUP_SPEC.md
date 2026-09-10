# 动态模型组规范

## 目的

动态模型组允许 Agent 定义只声明一个稳定模型位置，部署方在 managed Runtime 中发布和选择具体模型 profile。它不改变 `ModelCallLayer.forward()`，也不引入字符串模型简写。

## 声明

```python
requirement = ModelGroup.deferred(name="assistant")
model_layer = ModelCallLayer(
    model_group=requirement,
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(),
)
```

Deferred `ModelGroup` 的 `models` 必须为空；concrete `ModelGroup` 的 `models` 必须非空且名称唯一。组本身不包含并发或容量字段。

Direct execution 不能执行 deferred 模型组。Managed Runtime 在 bind/compile 时把组名记录为部署需求，但完整具体模型只存在于 Layer 固定配置或已发布 profile 中。

## 发布 profile

```python
handle = bound.model_groups.get(requirement)
snapshot = await handle.ensure_profile(
    profile="quality",
    models=config.model_groups["assistant"].models,
    invoker=invoker,
    make_default=True,
    deadline=monotonic() + 5,
)
```

`models` 是有序 `ModelEntry` 集合，顺序就是 fallback。发布接口不接收额外 fallback 参数。

Profile 必须提供以下资源来源之一：

- resident `invoker`；
- `resource_ref`，由一个 resolver 共享映射到所有模型；
- 完整 `resource_bundle`，以 `model_key` 映射每个模型资源。

`ModelResourceRef` 的 resolver、resource revision、capacity owner 和 coordinator domain 字段保持现有语义。`ModelResourceBundle.model_resources` 必须恰好覆盖 profile 的全部模型键。

非空 `provider_options` 在计算 digest 前校验。Resident invoker 使用 `validate_model(ModelEntry)`；可重建资源使用 resolver 的 `validate(ModelGroup, ModelResourceBundle)`。

## 选择与 admission

调用未显式选择 profile 时使用当前 default。允许请求级选择和生成参数覆盖，仍由 `ModelCallPolicy` 的既有 allowlist 控制。

Runtime admission 固定 snapshot ID、内容 digest 和资源 bundle digest。执行期间 profile 更新或 default 变化不能改变已经 pin 的调用。取消、deadline、恢复和 purge 继续使用现有 admission 生命周期。

## 持久化格式

Snapshot 中的模型组只包含：

```json
{
  "name": "assistant",
  "models": [
    {
      "name": "deepseek_primary",
      "spec": {
        "provider": "deepseek",
        "model_id": "deepseek-v4-flash",
        "protocol": "openai_chat_completions",
        "provider_options": {},
        "capabilities": {}
      }
    }
  ],
  "resolution": "concrete"
}
```

示意中的 `capabilities` 在真实数据中必须是完整六段结构。资源 bundle 使用 `model_resources` 数组，每项包含 `model_key` 和 `resource`。

旧字段结构不做双格式读取。解码遇到旧模型组、旧资源映射或不完整 ModelSpec 时明确失败。

## Worker 与 resolver

Worker 从相同 snapshot 重建同一 `ModelGroup`，并按 `ModelSpec.protocol` 选择 Adapter、按 `ModelEntry.name` 绑定 client。含非空 Provider options 的调用要求 Worker 声明 `model.provider-options.v1`。

历史 `AIMessage` 携带的 Provider continuation 作为 Message 值随 Worker 和 durable effect 传输。只有 profile 中实际进入且 Provider 与 protocol 同时匹配的模型可以回传该状态；不匹配时 Adapter 必须忽略它。

Resolver 租约、resident invoker ownership、关闭、取消和 coordinator domain 校验保持既有行为。Runtime 不根据 capabilities 修改 profile 或 fallback。

## 容量

Managed 模型并发只使用 Binding 的 `model_capacity`。Layer 在实际调用周围获取一次无参数 `model_permit()`；fallback 和 retry 不创建新的组级容量所有者。Direct execution 的并发由调用方控制。
