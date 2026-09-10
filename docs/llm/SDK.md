# LLM SDK

本文是 LLM 的第二级契约，必须服从 [LLM 第一原则](FEATURES.md)。

## 从 Mapping 加载

Pygent 接收普通 Mapping，不绑定 YAML。YAML、JSON、数据库或 UI 表单由应用转换成 Mapping 后使用同一个入口：

```python
from pygent import ModelConfig

config = ModelConfig.from_mapping(user_config)

primary = config.models["deepseek_primary"]
assistant = config.model_groups["assistant"]
connection = config.connections["deepseek_primary"]
```

完整配置形状：

```yaml
models:
  deepseek_primary:
    provider: deepseek
    model_id: deepseek-v4-flash
    protocol: openai_compatible
    connection:
      base_url: https://api.deepseek.com
      credential:
        env: DEEPSEEK_API_KEY
      verify_ssl: true
    provider_options: {}
    capabilities:
      modalities: {input: [text], output: [text]}
      streaming: {text: true}
      tools:
        call: true
        choice: [none, auto, required, named]
        parallel: true
      structured_output: {json_object: true, json_schema: false}
      reasoning: {supported: true, controllable: true}
      limits: {context_tokens: 1000000, max_output_tokens: 384000}

model_groups:
  assistant:
    models: [deepseek_primary]
```

解析严格拒绝未知字段、空名称、重复组条目、未知模型引用、不完整 capabilities、非法 URL、URL 内嵌凭据，以及同时设置 `credential.env` 和 `credential.none`。解析阶段不会读取 `DEEPSEEK_API_KEY`。

## Direct：单模型

应用在部署边界读取 credential 并构造 client/invoker。Layer 接收完整 `ModelEntry`：

```python
from pygent import GenerationConfig, ModelCallLayer, RetryPolicy
from pygent.llm import (
    DefaultModelInvoker,
    OpenAICompatibleAdapter,
    OpenAICompatibleClient,
)

entry = config.models["deepseek_primary"]
connection = config.connections[entry.name]
api_key = connection.credential.resolve()  # 部署边界才读取环境变量

client = OpenAICompatibleClient(
    base_url=connection.base_url,
    api_key=api_key,
    verify_ssl=connection.verify_ssl,
)
invoker = DefaultModelInvoker(
    adapters={"openai_compatible": OpenAICompatibleAdapter()},
    clients={entry.name: client},
)

model_layer = ModelCallLayer(
    model=entry,
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
    invoker=invoker,
)
```

单模型会规范化成名称为 `entry.name` 的单条目模型组。调用方在结束时关闭 invoker；配置对象不持有 client。

## Direct：多模型 fallback

```python
model_layer = ModelCallLayer(
    model_group=config.model_groups["assistant"],
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
    invoker=invoker,
)
```

`ModelGroup.models` 的顺序就是 fallback 顺序。Invoker 的 `clients` 必须按每个 `ModelEntry.name` 绑定，`adapters` 必须按每个 `ModelSpec.protocol` 绑定。Provider 名称不用于选择 client 或 Adapter。

## Managed：固定单模型或模型组

固定 Layer 可以省略 `invoker`，由 Runtime 使用组名注册现有 invoker：

```python
model_layer = ModelCallLayer(
    model=config.models["deepseek_primary"],
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
)

runtime.register_model_invoker(model_layer.model_group.name, invoker)
bound = runtime.bind(model_layer)
```

多模型写法相同，只把 `model=` 换成 `model_group=`。Binding 的 `model_capacity` 是 managed 模式的整体模型并发控制；模型配置不再声明容量。

## 动态 profile

```python
from time import monotonic
from pygent import ModelCallLayer, ModelGroup

requirement = ModelGroup.deferred(name="assistant")
model_layer = ModelCallLayer(
    model_group=requirement,
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(),
)

bound = runtime.bind(model_layer)
handle = bound.model_groups.get(requirement)
await handle.ensure_profile(
    profile="default",
    models=config.model_groups["assistant"].models,
    invoker=invoker,
    make_default=True,
    deadline=monotonic() + 5,
)
```

`ensure_profile()` 接收有序模型集合。使用 `resource_ref` 或 `resource_bundle` 时，Runtime 继续通过 resolver 重建并租赁 invoker。已发布 snapshot 是不可变的；旧持久化结构不会被兼容读取。

## 使用内置目录和能力模板

UI 选择内置模型时，把目录中的完整能力复制到用户配置：

```python
from pygent import ModelCapabilityCatalog, ProviderCatalog

providers = ProviderCatalog.builtin()
capabilities = ModelCapabilityCatalog.builtin().models[
    ("deepseek", "deepseek-v4-flash", "openai_compatible")
]
```

自定义模型可以从模板展开；两个 limits 必须显式提供：

```python
from pygent import CapabilityPresetCatalog

capabilities = CapabilityPresetCatalog.builtin().presets["text_tools"].materialize(
    context_tokens=131_072,
    max_output_tokens=8_192,
)
```

保存时写入完整 capabilities，不保存“模板 + 覆盖”。外部目录先由应用下载并校验，再调用 `ProviderCatalog.from_mapping()`、`ModelCapabilityCatalog.from_mapping()` 或 `CapabilityPresetCatalog.from_mapping()`。

## Provider 私有选项

```python
from pygent import ModelEntry, ModelSpec

entry = ModelEntry(
    "deepseek_primary",
    ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        protocol="openai_compatible",
        provider_options={"thinking": {"type": "disabled"}},
        capabilities=capabilities,
    ),
)
```

`provider_options` 是冻结的模型语义。DeepSeek 校验依据 `ModelSpec.provider`，实际 Adapter 分派依据 `ModelSpec.protocol`。

## 能力警告与事件

能力不匹配事件固定为：

```python
{
    "model_key": "deepseek_primary",
    "provider": "deepseek",
    "model_id": "deepseek-v4-flash",
    "missing_capabilities": ("tools.call",),
}
```

所有 attempt、prepared request、usage、reset 和 completion 事件也使用 `model_key`。`model.output.reset` 到达时，消费者按 `(model_key, attempt)` 撤销暂存输出。
