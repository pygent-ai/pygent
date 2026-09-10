# Pygent 多模型配置重构 Proposal

## 1. 三方相互独立

Pygent 面向三类相互独立的参与者：

- 模型服务商提供模型，不同服务商使用不同的 Model ID，并且实际提供的模型能力可能不同；
- Agent 开发者开发不绑定具体模型的通用 Agent；
- Agent 使用者自己决定使用官方服务、第三方模型平台或自部署模型。

模型服务商、Agent 开发者和 Agent 使用者之间不需要直接沟通。Pygent 在三方之间提供统一的模型配置契约。

## 2. 用户配置一个真实可调用的模型

用户输入中的每个模型条目把以下信息放在一起：

- 实际 Provider；
- 该 Provider 使用的 Model ID；
- 调用该模型使用的 protocol；
- 连接配置；
- Provider 私有选项；
- 该模型在这个 Provider 上的完整 capabilities。

即使底层是同一个基础模型，只要 Provider 不同，就分别配置模型条目，因为 Model ID、接口和实际能力可能不同。

`ModelConfig.from_mapping()` 从这份用户配置产生两个投影：

- 模型语义投影包含 Provider、Model ID、protocol、Provider 私有选项和 capabilities；使用模型组时还包含 `ModelSpec` 的排列顺序；
- 部署资源投影包含 endpoint、credential 引用、TLS、代理以及构造 client 和 invoker 所需的配置。

模型语义对象命名为 `ModelSpec`。它固定包含 `provider`、`model_id`、`protocol`、`provider_options` 和 `capabilities`。`ModelEntry` 把用户配置中的本地名称与一个 `ModelSpec` 组合起来；名称用于 fallback、资源绑定和诊断，不进入模型语义。`config.models[...]` 返回完整、不可变的 `ModelEntry`，`config.connections[...]` 返回同名的静态连接配置。模型语义进入 `ModelCallLayer`，部署资源配置交给现有 direct invoker 构造或 managed resource resolver。连接、凭据和活跃 client 不进入 `ModelSpec`。

第一版 credential 只使用一种对象形式，其中二选一：`credential: {env: DEEPSEEK_API_KEY}` 引用环境变量，`credential: {none: true}` 表示无需认证。解析后的 credential 属于部署资源投影，不进入 `ModelSpec`，Pygent 不把环境变量中的真实值写回配置、定义、事件或持久化数据。

Provider 使用开放的稳定字符串标识，不使用封闭枚举。Pygent 维护一份内置 Provider preset 列表，每个 preset 提供显示名称、默认 protocol、官方 base URL、鉴权类型、建议的 API Key 环境变量名称和 Provider 私有选项 schema。用户选择内置 Provider 时，UI 把这些值填入配置；未收录的 Provider 仍可使用自定义标识，并由用户填写 protocol、base URL 和鉴权配置。真实 API Key 不由 Pygent 提供，也不直接保存在模型配置中，配置只保存 credential 引用。

Provider preset、模型能力目录和协议 Adapter 相互独立：

- Provider preset 按 Provider 标识提供连接默认值和配置表单信息；
- 模型能力目录按 `(provider, model_id, protocol)` 提供完整 capabilities；
- 协议 Adapter 按 `protocol` 提供实际请求、响应和错误归一实现。

多个 Provider 可以使用同一个 protocol 和 Adapter。模型组、retry 和 fallback 不解释 Provider 协议；`ModelInvoker` 选中 `ModelSpec` 后，由对应 protocol 的 Adapter 完成实际调用。

第一版只提供 DeepSeek 官方 Provider preset，并只开放其 OpenAI-compatible protocol。Provider preset 和配置 schema 允许同一 Provider 在后续增加其他 protocol；DeepSeek 的 Anthropic-compatible 接口在第一版只保留扩展位置，不作为可选择的 protocol，也不提供对应 Adapter。

Provider preset 和模型能力目录使用两级发布：

- PyPI 发布的 `pygent` 包内置一份与该框架版本一起验证过的目录快照，作为默认和离线数据源；
- GitHub Releases 独立发布最新的版本化目录文件及其完整性信息，目录更新不要求重新发布 `pygent`。

Pygent 默认读取包内快照，不在模型调用过程中访问 GitHub。应用或 UI 可以在模型配置阶段显式检查 GitHub Releases、验证并下载最新目录，然后缓存到本地；也可以通过 `ProviderCatalog.from_mapping()` 加载应用或用户提供的目录。目录更新只影响之后新建或重新编辑的模型条目，不自动修改已经保存或正在使用的 `ModelSpec`，base URL 等连接信息只有在用户确认后才写入配置。

## 3. 用户明确知道模型能力

Pygent 不猜测模型能力，也不自动探测后替用户决定。

Pygent 提供：

- 一份官方模型基础表；
- 几种基础能力组合；
- 一张统一、完整的 capabilities 表单。

用户在 UI 中选择官方模型后，UI 直接把官方表中的完整能力填入配置。如果是新模型，用户从基础能力组合开始填写。用户可以逐项修改，最终保存的是展开后的完整能力，不是“模板 + override”。

第一版 capabilities 固定包含：

```yaml
capabilities:
  modalities:
    input: [text]
    output: [text]
  streaming:
    text: true
  tools:
    call: true
    choice: [none, auto, required, named]
    parallel: true
  structured_output:
    json_object: true
    json_schema: true
  reasoning:
    supported: true
    controllable: true
  limits:
    context_tokens: 131072
    max_output_tokens: 8192
```

## 4. 多模型只组合 ModelSpec

用户可以把一个或多个 `ModelSpec` 配成一个模型组：

```yaml
model_groups:
  assistant:
    models:
      - deepseek_primary
      - qwen_backup
```

`ModelSpec` 的排列顺序就是主模型和 fallback 顺序。

模型服务商负责服务端配额和限流。Pygent 提供整体模型并发控制：managed 模式使用 Binding 的模型容量，direct 模式由调用方控制并发；`ModelSpec` 和 `ModelGroup` 都不单独配置容量。

Capabilities 不参与自动选模型，也不改变 fallback。当请求与能力配置不一致时，Pygent 提供结构化警告，最终仍调用用户配置的模型。

Capabilities 在配置解析时预先转换为不可变能力标记。实际调用只比较本次请求与当前 `ModelSpec` 的能力标记；匹配时不创建警告对象或事件。同一个逻辑模型调用对同一个 `ModelSpec` 的相同能力缺失只警告一次，retry 不重复警告；fallback 真正使用另一个 `ModelSpec` 时才检查该 `ModelSpec`。检查不探测网络，也不重新解析配置。

## 5. Agent 开发者不处理 Provider 差异

Agent 开发者只需要声明或使用 Agent 中的模型位置，例如 `assistant`。

应用加载用户配置：

```python
config = ModelConfig.from_mapping(user_config)
```

然后把解析完成的模型配置交给 `ModelCallLayer`。

单模型：

```python
model_layer = ModelCallLayer(
    model=config.models["deepseek_primary"],
    retry_policy=retry_policy,
    generation=generation,
    invoker=invoker,
)
```

单模型条目在 Layer 内规范化为同名且只包含该条目的 `ModelGroup`，因此 direct 和 managed execution 使用相同模型身份；managed Runtime 以这个名称完成 invoker 或资源解析。

多模型和 fallback：

```python
model_layer = ModelCallLayer(
    model_group=config.model_groups["assistant"],
    retry_policy=retry_policy,
    generation=generation,
    invoker=invoker,
)
```

传给 `ModelCallLayer` 的是完整、不可变的模型语义配置，不是连接、凭据、活跃 client，也不是等待 Runtime 查找的模型地址引用。两种写法的区别只是使用单模型还是多模型组。

上述示例使用现有 direct 模式，因此显式传入部署阶段准备的 `invoker`，并由调用方负责其生命周期。managed 模式可以省略 `invoker`，继续使用现有 Runtime 注册或 resource resolver。`invoker` 是部署资源 SPI，不属于用户模型配置。

这一约定用于新的具体模型配置方式，不在本次重构中删除或替换现有延迟模型组和 Runtime profile 机制。

## 6. 配置只传一次

- `ModelConfig.from_mapping()` 只负责把一份 Mapping 转换为模型语义和部署资源两个投影；
- `ModelConfig.from_mapping()` 不会自动把配置交给 Runtime；
- Binding 不重复接收同一份模型配置；
- 完整、不可变的模型语义配置进入 `ModelCallLayer`；
- 连接、凭据和活跃 client 继续由现有部署资源机制管理；
- 不提供 `model_group="assistant"` 这样的字符串简写。

## 7. 本次重构的核心

本次重构重新整理 Pygent 的多模型抽象和用户配置方式，并为每个具体模型增加标准 capabilities。

新的具体模型配置是唯一的公开方式，不保留两套并行接口：公开的 `ModelRoute` 和 `routes` 由 `ModelSpec` 和有序 `models` 替代，`ModelGroupConfig` 上的 `max_concurrency` 和 `capacity_key` 由 managed Binding 的整体模型并发配置替代。现有延迟模型组和 Runtime profile 机制继续存在，但其中的具体模型同样使用 `ModelSpec`，不保留旧类型别名或字符串简写。

本次重构不同时重新设计：

- Provider client 生命周期；
- Provider 注册机制；
- Runtime 和 Worker；
- 整套容量系统；
- 所有模型协议；
- 新的 Provider Adapter。
