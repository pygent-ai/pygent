# Provider 路由选项规范

状态：Implemented
目标版本：Pygent 0.2.x 兼容升级

本文从属于 [Pygent 第一原则](../FEATURES.md)、[LLM 第一原则](FEATURES.md) 与 [Runtime 第一原则](../runtime/FEATURES.md)，并服从现有 [LLM SDK](SDK.md)、[Runtime SDK](../runtime/SDK.md) 及 [动态模型组规范](DYNAMIC_MODEL_GROUP_SPEC.md)。如有冲突，以上级契约为准。

## 1. 背景

`GenerationConfig` 只描述 Pygent 已标准化的生成行为。不同 Provider 或兼容端点仍可能要求额外的请求语义，例如 DeepSeek Chat Completions 的：

```json
{
  "thinking": {
    "type": "disabled"
  }
}
```

当前应用只能继承 `OpenAICompatibleAdapter` 并覆盖 `build_request()` 注入此类字段。这会把简单的稳定配置变成自定义 Python 行为，也无法由框架统一保证其不可变性、profile digest、admission pin、effect identity 和恢复兼容性。

本规范为这类无法或尚未进入 `GenerationConfig` 的 Provider 私有请求语义定义一个受控扩展缝。

## 2. 目标

本升级必须：

1. 允许每条 `ModelRoute` 声明严格 JSON 形式的 Provider 私有选项；
2. 保持 `GenerationConfig` provider-neutral；
3. 让 fallback 中不同 Provider 的选项彼此隔离；
4. 让选项进入 Module 定义、profile digest、admission pin 和 model effect identity；
5. 由对应 `ModelProviderAdapter` 验证和解释选项，Runtime 不解释 Provider 语义；
6. 保持 direct、fixed managed、deferred managed、持久恢复和远程 placement 的配置身份一致；
7. 保持没有 Provider 选项的现有调用、snapshot 和持久记录兼容；
8. 禁止借此传递 secret、连接、client、回调、重试、deadline 或 stream 控制。

## 3. 非目标

本升级不：

- 给 `forward()` 或 `ExecutionOptions` 增加任意 Provider 参数字典；
- 允许一次调用临时修改 route、credential、client、retry、fallback 或 Provider 选项；
- 让 Runtime 构造或解释 OpenAI、DeepSeek、Anthropic 等 Provider 请求；
- 规定所有 adapter 都把选项原样合并进 HTTP JSON body；
- 允许选项覆盖 Pygent 已拥有的通用请求字段；
- 自动把常见私有选项提升为新的 `GenerationConfig` 字段；
- 为 Provider 配置提供 secret 存储或部署资源解析能力；
- 改变现有 `ModelProviderClient` 传输协议或连接生命周期。

## 4. 核心语义

### 4.1 Route 级稳定声明

`ModelRoute` 增加一个仅关键字使用的可选字段：

```python
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelRoute:
    route_id: str
    provider: str
    model: str
    provider_options: JsonObjectInput = field(default=(), kw_only=True)
```

示例：

```python
route = ModelRoute(
    route_id="primary",
    provider="deepseek",
    model="deepseek-v4-pro",
    provider_options={
        "thinking": {
            "type": "disabled",
        },
    },
)
```

`provider_options` 是该 route 生成语义的一部分。修改它必须创建新的 `ModelRoute`，进而创建新的 Layer 定义或发布新的 profile snapshot；不得原地修改已冻结的 route、Layer 或已发布 snapshot。

省略该字段等价于空对象。已有三个位置参数 `ModelRoute(route_id, provider, model)` 的调用形状保持有效；新代码应使用关键字传递 `provider_options`。

### 4.2 严格 JSON 与不可变性

构造 `ModelRoute` 时必须使用与其他公开 JSON 值相同的边界完成：

- 防御性复制；
- 严格 JSON 校验；
- 递归冻结；
- 拒绝 NaN、Infinity、bytes、callable 和任意活 Python 对象；
- 对外暴露 `FrozenJsonObject`，不得暴露调用方原始 dict。

调用方在构造后修改原字典，不得改变 route、definition identity 或后续 Provider 请求。

### 4.3 与 GenerationConfig 的边界

`GenerationConfig` 继续承载 Pygent 已标准化且跨 route 共享的生成策略。`provider_options` 只承载对应 adapter 定义的私有扩展。

两者冲突时不得用 merge 顺序决定结果。Provider adapter 必须拒绝让 `provider_options` 覆盖框架拥有的字段。

若某个私有选项以后形成稳定、可验证的跨 Provider 语义，未来版本可以将其提升为类型化的 `GenerationConfig` 字段。提升必须有独立规范、兼容策略和 adapter capability 规则；本规范不自动执行该提升。

### 4.4 Route 隔离

选项只属于声明它的 route：

```python
ModelGroupConfig(
    name="assistant",
    routes=(
        ModelRoute(
            "primary",
            "deepseek",
            "deepseek-v4-pro",
            provider_options={"thinking": {"type": "disabled"}},
        ),
        ModelRoute("fallback", "openai", "gpt-5"),
    ),
    fallback=FallbackPolicy(("primary", "fallback")),
)
```

`primary` 的选项不得传播、继承或合并到 `fallback`。Invoker 选择 route 后，只有该 route 的选项进入对应 adapter 的请求转换。

## 5. 权限与验证边界

### 5.1 通用构造验证

`ModelRoute` 只验证 provider-neutral 事实：字段非空、选项为严格 JSON object、值可冻结。它不判断某个 Provider 是否支持具体选项。

### 5.2 Provider 配置验证

Provider-specific validation 属于 LLM/application preparation，不属于 Runtime 语义。实现应提供独立的可选 SPI：

```python
@runtime_checkable
class ModelProviderRouteValidator(Protocol):
    def validate_route(self, route: ModelRoute) -> None: ...
```

规则如下：

- 内置 adapter 对非空 `provider_options` 必须提供验证；
- 第三方 adapter 没有实现该协议时，空选项继续兼容；
- 第三方 adapter 没有实现该协议却收到非空选项时，部署准备必须 fail closed；
- validator 不执行网络 I/O，不读取 credential，不创建 client，不修改 route；
- validator 的成功只证明结构和 adapter 支持，不证明 Provider 当前可用；
- `build_request()` 必须重复执行等价的安全检查，防止绕过部署准备的 direct 或自定义调用路径。

动态 profile 发布顺序必须是：

```text
构造并冻结 ModelRoute
    -> LLM/application preparation 验证 adapter 支持和 Provider 选项
    -> 生成完整 canonical profile digest
    -> Runtime 验证 provider-neutral 发布事实
    -> 原子发布 snapshot
```

Runtime 可以调用一个通用准备接口并接收验证结果，但不得按字段名解释 `provider_options`，也不得硬编码 Provider 规则。

### 5.3 错误分类

配置或 profile 发布前发现的非法选项属于 `ModelGroupConfigurationError` 或等价的配置错误，不属于 Provider attempt 失败，且不得产生 `model.started`。

direct execution 或绕过准备边界后在 `build_request()` 发现非法选项时，adapter 必须在 Provider I/O 前抛出：

```python
ModelProviderError(ModelErrorKind.INVALID_REQUEST, <sanitized message>)
```

错误消息可以包含安全的字段名，不得包含 secret、完整请求体或任意敏感值。

## 6. OpenAI-compatible 投影

### 6.1 投影规则

`OpenAICompatibleAdapter` 将经过验证的 `provider_options` 解释为 Chat Completions JSON body 的额外顶层字段。

投影采用顶层浅合并。不得递归合并嵌套对象，也不得依赖字典覆盖顺序处理冲突。

对 DeepSeek 示例，最终请求包含：

```json
{
  "model": "deepseek-v4-pro",
  "messages": [],
  "thinking": {
    "type": "disabled"
  }
}
```

`extra_body` 是 OpenAI SDK 的调用概念，不成为 Pygent 公共字段名；Pygent 的公共声明仍称为 `provider_options`。

### 6.2 框架保留字段

OpenAI-compatible adapter 至少保留以下字段：

```text
model
messages
temperature
max_tokens
response_format
tools
tool_choice
stream
```

`provider_options` 包含任一保留字段时必须拒绝，即使该字段当前没有由 `GenerationConfig` 或可见工具实际发出。

保留规则保证：

- route 不能通过私有选项替换实际模型；
- adapter 仍唯一拥有 Message 与 ToolDefinition 投影；
- stream 仍由统一的 invoke/stream 执行路径控制；
- 通用生成字段不会被私有配置静默覆盖；
- payload 在 direct、managed、流式和非流式路径保持同源。

### 6.3 Provider 子 schema

内置 adapter 可以按 `route.provider` 对已知 Provider 增加更严格的结构验证。例如 DeepSeek 的 `thinking`：

```text
thinking 必须是 object
thinking.type 必须是 "enabled" 或 "disabled"
thinking 不接受未知字段
```

未知 OpenAI-compatible Provider 默认接受不与保留字段或敏感类别冲突的严格 JSON 选项。这只是结构透传，不证明目标 Provider 或 endpoint 支持相应语义；应用仍需通过部署验证、集成测试或 Provider 文档确认能力。严格部署可以选择要求更窄的显式 Provider validator。

## 7. 定义身份、snapshot 与 digest

### 7.1 Canonical route 投影

所有 definition、profile、manifest、effect 和 wire codec 必须复用同一个 canonical route 投影语义：

```json
{
  "route_id": "primary",
  "provider": "deepseek",
  "model": "deepseek-v4-pro",
  "provider_options": {
    "thinking": {
      "type": "disabled"
    }
  }
}
```

空 `provider_options` 在 0.2.x 兼容编码中必须省略。解码缺失字段时恢复为空对象。这样现有无选项 route 的 canonical payload 和 digest 保持不变。

任何非空选项及其嵌套值变化都必须改变：

- fixed Module definition identity；
- dynamic profile digest；
- admission snapshot/pin digest；
- model effect request identity。

JSON object key 顺序不得影响 digest。

### 7.2 Profile snapshot

动态 profile 的 canonical group value 必须包含每条 route 的非空 `provider_options`。SQLite 与其他 `ModelDeploymentStore` 实现必须无损保存和恢复该字段。

两个 profile 具有相同 routes、fallback 和资源，但 `provider_options` 不同时，是不同配置 digest。`ensure_profile()` 不得把它们折叠为同一发布。

### 7.3 Admission pin

一次 admission 固定的 snapshot 必须包含原始 route 选项。profile 后续重新发布不得改变已经 admission 的 Execution。

Runtime 在执行中不得读取 adapter 的“当前默认选项”替代 pin 中的选项，也不得丢弃选项后重新查询 profile current。

## 8. Effect 与恢复

### 8.1 Model effect request

model effect request 中每条 route 的 canonical 投影必须包含非空 `provider_options`。因此相同 Message、Context 和 GenerationConfig 在不同 Provider 选项下不是同一个 effect 请求。

已提交 effect 的重放直接使用保存的结果，不需要 live adapter 或 Provider client。

### 8.2 未提交工作恢复

恢复需要新 Provider 工作时，必须取得原 admission pin 指向的精确资源 revision，并使用 pin 中的原始 route 选项。以下任一情况必须 fail closed：

- 原 snapshot 无法解码；
- route 选项被丢失或被当前 profile 替换；
- 精确资源 revision 不可用；
- 当前 adapter 不再支持 pin 中的选项；
- Worker 无法验证或无损承载扩展后的 route schema。

恢复不得静默删除未知选项，也不得用空对象降级继续请求 Provider。

### 8.3 Worker 与远程 placement

所有承载 profile snapshot、admission manifest 或 model effect request 的 Worker codec 必须无损往返 `provider_options`。不认识扩展 route schema 的 Worker 必须在 placement/admission 阶段拒绝，不得在 Provider I/O 后失败。

如果远程能力协商使用显式 capability，支持本规范的实现应声明新的稳定 capability，例如：

```text
model.route.provider-options.v1
```

是否新增 capability 由现有 Worker schema/version 协商机制决定，但不得仅凭 Python 包版本推断兼容。

## 9. 安全边界

`provider_options` 是 portable semantic configuration，不是资源容器。禁止包含：

- API key、token、cookie、Authorization header；
- endpoint、代理认证、TLS 私钥、CA 或证书校验策略（包括 `verify_ssl`）；
- client、session、连接池、锁、Task、协程或 callback；
- retry 次数、backoff、attempt timeout 或总 deadline；
- stream 开关；
- Runtime、Binding、Execution 或 resource resolver 对象；
- Provider 原始响应或内部异常。

Credential、endpoint、headers、TLS/CA 策略和连接仍通过精确 `ModelResourceRef`、resolver、client 与 invoker 部署边界提供。配置界面、日志、异常、事件和诊断输出不得打印完整 `provider_options`；只允许经过策略批准的字段名或摘要。

框架无法仅凭字符串内容可靠判断 secret，因此发布方仍承担不把 secret 放入 portable metadata 的责任。受信准备层可以实施额外 denylist、schema 和大小限制，但不能把扫描成功描述为 secret-free 证明。

## 10. Direct 与 managed 行为

### 10.1 Direct execution

固定 `ModelCallLayer` 使用其不可变 `ModelGroupConfig.routes`。Invoker 选择 route，adapter 在 Provider I/O 前验证并投影该 route 的选项。

Direct execution 不获得 profile publication、跨 Root 容量治理或 durable recovery；但 route 不可变性、保留字段和 adapter 验证规则与 managed execution 相同。

### 10.2 Fixed managed execution

固定 managed 注册继续使用 Layer 中的 concrete group。编译后的定义身份和 model effect request 必须包含非空 route 选项。Runtime 仍只解析已注册 invoker，不解释选项。

### 10.3 Deferred managed execution

deferred group 的选项由发布的 profile route 提供，不由 Agent 中的空 requirement、`ExecutionOptions` 或当前 adapter 默认值提供。Admission 固定完整 snapshot，后续执行只使用该 pin。

## 11. 兼容性

### 11.1 Python SDK

现有调用保持有效：

```python
ModelRoute("primary", "openai", "gpt-5")
```

其 `provider_options` 为冻结的空对象。现有 adapter 在空选项下不需要实现新的 validator 协议，现有请求 payload 不增加字段。

### 11.2 持久数据

解码器必须接受旧 route 对象缺少 `provider_options`，并恢复为空对象。编码器对空对象省略字段，以保持现有 canonical digest。

对非空选项的支持是新的持久语义。所有保存或转发 route 的 codec 必须在同一升级中完成；不能发布只支持写入、不支持恢复或远程往返的部分实现。

### 11.3 第三方 adapter

第三方 adapter 的现有空选项路径保持兼容。若应用为该 adapter 配置非空选项，则 adapter 必须显式实现验证和投影；框架不得假设任意 adapter 支持透传。

## 12. 实现边界

推荐实现顺序：

```text
扩展并冻结 ModelRoute.provider_options
    -> 建立唯一 canonical route projection helper
    -> 更新 profile/snapshot/store codec 与 digest
    -> 更新 admission manifest 与 model effect request
    -> 更新 Worker/remote codec 和能力协商
    -> 增加 Provider route validator SPI
    -> 实现 OpenAI-compatible 保留字段与浅合并
    -> 实现 DeepSeek thinking 子 schema
    -> 更新 SDK、README、示例和兼容说明
    -> 运行 direct/managed/durable/remote 验收矩阵
```

不得先发布 adapter 透传，再在后续版本补 digest 或恢复支持。

## 13. 验收标准

### 13.1 值与 API

- `ModelRoute` 接受省略或非空 `provider_options`；
- 原始 dict 在构造后修改不影响 route；
- 嵌套 list/object 被递归冻结；
- 非 JSON、非 object 和非有限数值被拒绝；
- 三位置参数构造保持兼容；
- `repr`、异常和公共事件不泄漏完整敏感配置值。

### 13.2 Adapter

- DeepSeek route 可生成 `thinking.type=disabled`；
- 同一 adapter 的无选项 route 请求保持原样；
- 保留字段覆盖在 Provider I/O 前失败；
- 输入或返回 payload 的外部修改不污染后续请求；
- fallback route 只收到自身选项；
- stream 与非 stream 使用同一基础请求投影；
- 第三方 adapter 对非空未知选项 fail closed，除非显式支持。

### 13.3 Identity 与 publication

- 空选项不改变已有 canonical route payload 和 digest；
- 非空选项或嵌套值变化会改变 definition、profile 和 effect identity；
- JSON key 顺序不改变 identity；
- 同 profile 名、不同选项产生不同 publication digest；
- 已 admission 的 Execution 在 profile 更新后继续使用原选项；
- Runtime validation 不包含 Provider 字段名分支。

### 13.4 持久化与恢复

- SQLite profile snapshot 无损往返非空嵌套选项；
- 旧 snapshot 缺失字段时解码为空对象；
- 已完成 effect 重放不获取 adapter/client；
- 未完成工作只使用原 admission pin 的选项与资源 revision；
- adapter 不再支持原选项时恢复 fail closed；
- Worker codec 无损往返，旧 Worker 在 placement/admission 前拒绝。

### 13.5 回归

- 现有 LLM、Runtime、durability、Worker 和 public API 测试通过；
- 现有 OpenAI-compatible 请求在空选项下字节语义不变；
- route、retry、fallback、deadline、取消和 client ownership 行为不变；
- 不新增 Runtime Provider 逻辑、隐式 retry 或第二套 stream 路径。

## 14. 文档完成条件

实现发布前必须同步：

- [LLM SDK](SDK.md) 的 route 配置示例和限制；
- [LLM 详细契约](README.md) 的配置与 adapter 边界；
- [动态模型组规范](DYNAMIC_MODEL_GROUP_SPEC.md) 的 snapshot/digest/validation 描述；
- Provider adapter SPI 文档；
- 持久化与 Worker 兼容说明；
- DeepSeek 或通用 OpenAI-compatible 示例。

本规范不修改第一原则；它把第一原则已经要求的不可变声明、Provider adapter 边界、精确 deployment pin 和可验证恢复应用到 Provider 私有 route 配置。
