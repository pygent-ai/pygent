# LLM SDK

本文是 LLM 的第二级契约，必须服从 [LLM 第一原则](FEATURES.md)。

Connection、Model、ModelGroup 使用统一的三层配置契约。`ModelConfig.from_mapping()` 解析普通 Mapping，`ModelConfig.connection_for()` 在部署装配边界返回模型已经选定 protocol 后的连接投影。

## 从 Mapping 加载

Pygent 接收普通 Mapping，不绑定 YAML。YAML、JSON、数据库或 UI 表单由应用转换成 Mapping 后使用同一个入口：

```python
from pygent import BuiltinModelProtocol, ModelConfig

config = ModelConfig.from_mapping(user_config)

primary = config.models["deepseek_primary"]
assistant = config.model_groups["assistant"]
connection = config.connection_for("deepseek_primary")

assert BuiltinModelProtocol.OPENAI_CHAT_COMPLETIONS == "openai_chat_completions"
assert BuiltinModelProtocol.OPENAI_RESPONSES == "openai_responses"
assert BuiltinModelProtocol.ANTHROPIC_MESSAGES == "anthropic_messages"
assert BuiltinModelProtocol.GEMINI_GENERATE_CONTENT == "gemini_generate_content"
```

完整配置形状：

```yaml
connections:
  company_gateway:
    provider: custom_gateway
    credential:
      env: MODEL_API_KEY
    verify_ssl: true
    protocols:
      openai_chat_completions:
        base_url: https://gateway.example.com/v1
      openai_responses:
        base_url: https://gateway.example.com/v1
      anthropic_messages:
        base_url: https://gateway.example.com/anthropic

models:
  deepseek_primary:
    connection: company_gateway
    model_id: deepseek-v4-flash
    protocol: openai_chat_completions
    provider_options: {}
    capabilities:
      modalities: {input: [text], output: [text]}
      streaming: {output: [text]}
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

`company_gateway` 是 `connection_key`，`deepseek_primary` 是用户定义的 `model_key`（UI 可显示为“模型别名”），`assistant` 是模型组名称。`model_id` 是服务商提供的真实模型 ID，与 `model_key` 含义不同。Connection 的 `provider` 是开放字符串；内置 Provider 与自定义 Provider 使用相同结构。Model 的 `protocol` 必须引用所选 Connection 的 `protocols` 条目。解析器用 Connection 的 Provider 与 Model 的 model ID、protocol、options 和 capabilities 构造完整 `ModelSpec`，因此 Model 不重复保存 Provider。

解析严格拒绝未知字段、空名称、重复组条目、未知 Connection/Model 引用、Connection 中不存在的 protocol、不完整 capabilities、非法 URL、URL 内嵌凭据，以及同时设置 `credential.env` 和 `credential.none`。模态只能是 `text`、`image`、`audio`、`video`，`streaming.output` 必须属于输出模态；无法确认的 token limit 显式写 `null`。解析阶段不会读取 `MODEL_API_KEY`。

## 逐级校验和保存

应用可以按三层分别校验并保存配置，不需要先填写完整 `ModelConfig`：

```python
from pygent import (
    ConnectionConfig,
    EnabledModelConfig,
    ModelConfig,
    ModelGroupConfig,
)

connection = ConnectionConfig.from_mapping(connection_value)
model = EnabledModelConfig.from_mapping(model_value)
model_group = ModelGroupConfig.from_mapping(model_group_value)

# 以下 Mapping 代表应用自己的数据库记录、JSON 文档或配置文件内容。
saved_connections = {"company_gateway": connection.to_mapping()}
saved_models = {"deepseek_primary": model.to_mapping()}
saved_model_groups = {"assistant": model_group.to_mapping()}

config = ModelConfig.from_mapping(
    {
        "connections": saved_connections,
        "models": saved_models,
        "model_groups": saved_model_groups,
    }
)
```

这三个值只校验各自层级的结构。`EnabledModelConfig` 保存 `connection_key`，但不要求解析时 Connection 已存在；`ModelGroupConfig` 保存有序 `model_keys`，但不要求解析时 Model 已存在。每个 `to_mapping()` 都返回与对应 `from_mapping()` 可往返的普通 Mapping，供应用写入 YAML、JSON、数据库或配置中心；Pygent 不提供 `save_connection()`，也不规定存储后端或草稿格式。

应用已经持有解析后的配置值时，可以跳过再次解析 Mapping：

```python
config = ModelConfig.from_components(
    connections={"company_gateway": connection},
    models={"deepseek_primary": model},
    model_groups={"assistant": model_group},
)
```

`ModelConfig.from_mapping()` 先解析三层 Mapping，再进入同一个 `from_components()` 组装路径。最终组装统一检查 Connection、protocol 和 Model 引用，并产生运行时 `ModelEntry`、`ModelGroup` 与连接投影。

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
connection = config.connection_for(entry.key)
api_key = connection.credential.resolve()  # 部署边界才读取环境变量

client = OpenAICompatibleClient(
    base_url=connection.base_url,
    api_key=api_key,
    verify_ssl=connection.verify_ssl,
)
invoker = DefaultModelInvoker(
    adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
    clients={entry.key: client},
)

model_layer = ModelCallLayer(
    model=entry,
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
    invoker=invoker,
)
```

`connection_for()` 返回该模型已经选定 protocol 后的不可变 `ResolvedModelConnection`，包含 `connection_key`、Provider、protocol、base URL、credential 引用、TLS 和代理策略。单模型会规范化成名称为 `entry.key` 的单条目模型组。调用方在结束时关闭 invoker；配置对象不持有 client。

多个模型引用同一个 Connection 和 protocol 时，应用只创建一个 client，再按 `model_key` 绑定给 Invoker：

```python
entries = (config.models["fast"], config.models["reasoner"])
connection = config.connection_for(entries[0].key)
assert all(config.connection_for(entry.key) == connection for entry in entries)

shared_client = OpenAICompatibleClient(
    base_url=connection.base_url,
    api_key=connection.credential.resolve(),
    verify_ssl=connection.verify_ssl,
)
invoker = DefaultModelInvoker(
    adapters={"openai_chat_completions": OpenAICompatibleAdapter()},
    clients={entry.key: shared_client for entry in entries},
)
```

共享单位是 `(connection_key, protocol)`。不同 protocol 使用各自 Adapter 和 endpoint，即使它们属于同一个 Connection。Invoker 关闭时对相同 client 实例去重。

## 多模态工具消息

OpenAI Chat Completions compatible endpoint 只有经过应用明确声明后，才会接收
ToolResult 中的结构化内容块。模型本身还必须在 `ModelCapabilities.modalities.input`
中声明对应的 `image` 或 `video` 输入能力。这两个条件独立检查：

```python
from pygent.llm import (
    OpenAICompatibleAdapter,
    MediaTransportCapabilities,
)


class AppMediaResolver:
    def resolve(self, source):
        # 在部署边界按稳定 URI 读取有界 bytes，并实施权限与生命周期策略。
        return media_store.read(source.uri)


adapter = OpenAICompatibleAdapter(
    media_transport=MediaTransportCapabilities(
        enabled=True,
        modalities=("image", "video"),
        source_kinds=("url", "inline"),
        max_media_bytes=20 * 1024 * 1024,
        image_mime_types=("image/jpeg", "image/png", "image/webp"),
        video_mime_types=("video/mp4",),
    ),
)
resolver = AppMediaResolver()
invoker = DefaultModelInvoker(
    adapters={"openai_chat_completions": adapter},
    clients={entry.key: client},
    media_resolver=resolver,
)
```

默认 `MediaTransportCapabilities()` 为禁用状态，框架不会因为协议名、Provider 名或模型
名推断 endpoint 支持该扩展。`ModelCapabilities.media_input` 进一步声明模型接受的 MIME、
大小、图片尺寸/像素、视频时长/FPS/音轨等细节。Invoker 取模型能力和 endpoint 能力的
交集，为每个目标模型独立生成 request-local 投影；图片可缩放、转码或压缩，视频在安装
`video` extra 后可缩放、降帧、移除音轨并转码为 MP4。投影自己的摘要、大小与转换步骤进入
prepared-request trace，canonical `MediaBlock` 与 Context 保持不变。

视频能力还可以通过 `media_input.video.delivery_modes` 按模型声明 `video_url` 或
`image_frames`。该字段是模型级明细，不按 Provider 整族推断。OpenAI Chat Completions
投影会保留原 `tool` 结果的文本和调用关联，并把图片或视频放入紧随其后的 `user` 媒体
消息，因为该协议的媒体内容不是 `tool` 角色内容；这只影响请求 wire，不修改 Context 中的
canonical ToolResult。

`media_resolver` 由 Invoker 使用，负责在需要转换或 inline 传输时解析稳定 resource/URL。
inline Base64 按解码后的字节校验和计算 SHA-256；resource 按声明的摘要和大小复核。Adapter
只把最终投影编码成目标协议原生字段。Anthropic Messages、Gemini generateContent、OpenAI
Responses 和 OpenAI Chat Completions compatible Adapter 都使用同一投影契约；各 Adapter
只声明自身实际支持的模态、来源与 MIME，不支持的视频不会被伪装成图片或文本。

若模型组没有候选能够查看历史媒体，Invoker 为首个候选生成一次 `status="not_viewed"` 的
请求投影，说明当前模型没有查看该媒体的能力，并保留 `media_ref`；这不表示文件不存在，也
不会修改原工具结果。普通超时、限流或响应错误继续走现有 retry/fallback，失败尝试的
Assistant 不进入 Context。fallback 到另一个兼容模型时，新投影始终从 canonical media
重新生成，不从上一个模型的派生结果继续转换。

## 多模态用户消息

`UserMessage` 除了文本 `content`，还支持 `media` 媒体块元组；块类型为 `MediaBlock`
（工具结果内容块与用户消息共用同一类型），来源为 `MediaSource`。文本之后按声明顺序
投递媒体；构造时的校验
（MIME、SHA-256、尺寸描述）、路由门控、request-local 投影、token 估算和 `not_viewed`
降级都与工具结果媒体共用同一套管线，canonical Context 不变。Adapter 把用户媒体编码为
目标协议的原生 user 消息部分：OpenAI Responses 使用 `input_image`，Chat Completions
使用 `image_url`/`video_url` content parts，Anthropic Messages 使用 `image` source 块，
Gemini generateContent 使用 `inlineData` part。Wire codec 会序列化 `media` 字段，读取
不带该字段的旧 payload 时按无媒体处理。

## Direct：Anthropic Messages

Anthropic Messages 使用独立的 client 和 Adapter。该协议要求每次请求显式设置正整数 `max_output_tokens`：

```python
from pygent.llm import AnthropicMessagesAdapter, AnthropicMessagesClient

entry = config.models["anthropic_primary"]
connection = config.connection_for(entry.key)
client = AnthropicMessagesClient(
    base_url=connection.base_url,
    api_key=connection.credential.resolve(),
    verify_ssl=connection.verify_ssl,
)
invoker = DefaultModelInvoker(
    adapters={"anthropic_messages": AnthropicMessagesAdapter()},
    clients={entry.key: client},
)
model_layer = ModelCallLayer(
    model=entry,
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
    invoker=invoker,
)
```

DeepSeek 官方和 Alibaba Cloud Token Plan 都可以使用同一 Adapter。用户先在 Connection 中启用对应 Provider preset 的 `anthropic_messages` endpoint，再由 Model 显式选择它；使用 OpenAI Chat Completions 时选择同一 Connection 中的 `openai_chat_completions` endpoint。Pygent 不自动探测或切换协议。

## Direct：多模型 fallback

```python
model_layer = ModelCallLayer(
    model_group=config.model_groups["assistant"],
    retry_policy=RetryPolicy(),
    generation=GenerationConfig(max_output_tokens=256),
    invoker=invoker,
)
```

`ModelGroup.models` 的顺序就是 fallback 顺序。Invoker 的 `clients` 必须按每个 `ModelEntry.key` 绑定，`adapters` 必须按每个 `ModelSpec.protocol` 绑定。Provider 名称不用于选择 client 或 Adapter。

待发送的 ToolResult 含图片或视频时，Invoker 在发出 I/O 前按这个顺序检查模型输入模态、
endpoint 的结构化 tool-result 能力、source kind 与已知大小限制。不兼容候选产生
`model.route.skipped`，但不产生 attempt、prepared request 或费用；剩余兼容候选仍按原
顺序 retry/fallback。若没有任何兼容候选，首个候选收到请求级不可用投影：assistant 的
ToolCall 与 `call_id` 不变，ToolResult 中无法发送的媒体成为结构化文字说明。这个投影只
属于本次 `ModelProviderRequest`，不会覆盖 Context 中的真实媒体。

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
    ("deepseek", "deepseek-v4-flash", "openai_chat_completions")
]

token_plan = providers.providers["aliyun_token_plan"]
token_plan_capabilities = ModelCapabilityCatalog.builtin().models[
    ("aliyun_token_plan", "qwen3.8-max", "openai_chat_completions")
]
```

Token Plan 的 OpenAI preset 使用 `ALIYUN_TOKEN_PLAN_OPENAI_API_KEY` 和 `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1`；Anthropic preset 使用 `ALIYUN_TOKEN_PLAN_ANTHROPIC_API_KEY` 和 `https://token-plan.cn-beijing.maas.aliyuncs.com/apps/anthropic`。应用把所选 Provider preset 的协议入口填入 Connection，把完整 capability 记录填入引用该 Connection 的 Model。目录同时展示图像、视频和音频模型；其五个 `dashscope_*` protocol 第一版没有内置 Adapter，应用应只在自行装配对应 Adapter 后标记为可执行。

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
    key="deepseek_primary",
    spec=ModelSpec(
        provider="deepseek",
        model_id="deepseek-v4-flash",
        protocol="openai_chat_completions",
        provider_options={"thinking": {"type": "disabled"}},
        capabilities=capabilities,
    ),
)
```

`provider_options` 是冻结的模型语义。通过 Mapping 加载时，`ModelSpec.provider` 来自 Model 引用的 Connection，`ModelSpec.protocol` 来自 Model 选择的 protocol；直接构造 `ModelSpec` 时仍显式提供完整语义。DeepSeek 校验依据 `ModelSpec.provider`，实际 Adapter 分派依据 `ModelSpec.protocol`。

Alibaba Cloud Token Plan 的 OpenAI 扩展字段为 `enable_thinking`、`preserve_thinking`、`reasoning_effort`、`thinking`、`thinking_budget` 和 `tool_stream`。Adapter 对字段、类型和枚举做闭集校验；例如：

```python
provider_options={
    "enable_thinking": True,
    "thinking_budget": 4096,
}
```

Anthropic Messages 的私有选项为 `thinking`、`output_config`、`service_tier` 和 `stop_sequences`，字段和值由 Adapter 严格校验。例如：

```python
provider_options={
    "thinking": {"type": "enabled", "budget_tokens": 2048},
    "output_config": {"effort": "high"},
}
```

`thinking.type=enabled` 时 budget 必须小于本次 `max_output_tokens`；启用 thinking 时显式 temperature 只能为 `1`。

## Provider continuation

任意 OpenAI Chat Completions Provider 的响应实际返回合法 `reasoning_content` 时，以及 Anthropic Messages 返回 thinking block 时，Adapter 会将其规范化为 `AIMessage.continuation`，并记录实际生产它的 `model_key`。Anthropic 官方 block 必须有 signature；Anthropic-compatible Provider 的无 signature block 会保持无 signature。ReAct 工具循环会把它原样回传给 `model_key`、Provider、model ID 与 protocol 都匹配的后续请求。工具结果返回后，模型组先尝试原生产模型；fallback 保留 assistant tool call 和 tool result，只移除不属于它的私有 continuation。Continuation 会随 Message 经过 Worker、effect 与 SQLite 持久化；其原始 `data` 不会出现在 `repr`、公开模型事件或 prepared-request snapshot 中，snapshot 只保留摘要。应用通常不需要读取或修改它。

当前 `ModelCallLayer` 返回文本与 ToolCall。内置文本 Adapter 遇到图片、音频、视频或 embedding 返回部件会明确拒绝；能力目录中的这些输出类型用于配置与专用 Adapter，不会被文本 Adapter 静默转换为空文本。

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

所有 attempt、prepared request、usage、reset 和 completion 事件也使用 `model_key`。`model.output.reset` 到达时，消费者按 `(model_key, attempt)` 撤销暂存输出。`model.route.skipped` 包含候选模型身份和稳定 `missing_capabilities`；因为没有真实 Provider attempt，它不会配对 `model.attempt.started`。

## 可选 AZ 一致性验证

AZ 只作为外部测试网关，不是 Pygent 内置 Provider。验证清单固定为 `tests/live/az_conformance/manifest.json` 中的 211 个 route；runner 会先比较实时 `/v1/models` 与冻结 SHA-256，发生漂移时在任何付费模型调用前退出。

凭据只通过进程环境传入：`AZ_BASE_URL` 必须是无内嵌凭据的 HTTPS URL，`AZ_API_KEY` 必须非空。结果目录必须放在仓库外；结果键包含快照 digest、Git source revision、route、protocol 和 scenario，因此代码或清单变化后不会复用旧通过记录。命令可能产生费用，输出与 ledger 不保存 API key、Provider 原始正文或下载媒体。

```powershell
$azResultDir = 'C:\Users\Administrator\.codex\artifacts\pygent-az-211-7398801a'
uv run --extra az-conformance python -m tests.live.az_conformance.cli validate --manifest tests/live/az_conformance/manifest.json
uv run --extra az-conformance python -m tests.live.az_conformance.cli inventory --manifest tests/live/az_conformance/manifest.json
uv run --extra az-conformance python -m tests.live.az_conformance.cli run --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir
uv run --extra az-conformance python -m tests.live.az_conformance.cli report --manifest tests/live/az_conformance/manifest.json --output-dir $azResultDir
```

只有同一 source revision 下 211 个 route 的所有 required scenario 都有 `passed` 记录时，report 才返回 `complete=true`；权限、限流、网关或上游故障仍是失败，不会被记成 skipped。
