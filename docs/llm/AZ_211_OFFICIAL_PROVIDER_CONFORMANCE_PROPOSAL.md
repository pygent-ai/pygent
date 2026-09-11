# Pygent 官方 Provider 与 AZ 211 模型一致性验证 Proposal

## 1. 目标

Pygent 以模型官方来源定义 Provider、Model ID、协议和 capabilities，同时使用
`az.gptplus5.com` 作为独立的真实连接渠道，验证 Pygent 对不同官方模型和协议的兼容性。

本次交付必须同时满足：

- `az.gptplus5.com` 不进入 Pygent 内置 `ProviderCatalog`；
- Pygent 内置目录只发布能由模型官方资料确认的 Provider、Model ID 和能力；
- AZ 当前 Token 返回的 211 个 route ID 全部进入测试清单并逐项真实调用；
- 每个 route ID 的所有必需测试场景都通过，才算该 route 通过；
- 平台路由别名、官方模型和非模型搜索服务保持明确区分；
- 现有 DeepSeek、Anthropic、Alibaba Cloud Token Plan 和 Runtime 行为不回归。

这里的“全部测试通过”不是指成功读取 `/v1/models`，也不是只对所有模型发送同一种
聊天请求，而是每个 route 根据其可确认能力执行对应测试。

## 2. 三种身份彼此分离

### 2.1 官方模型身份

官方模型身份由以下三元组确定：

```text
(official_provider, official_model_id, protocol)
```

它用于 Pygent 的 `ProviderCatalog` 和 `ModelCapabilityCatalog`。其中：

- `official_provider` 是模型或官方托管服务的稳定标识；
- `official_model_id` 必须能在官方资料中确认；
- `protocol` 是 Pygent 实际实现的 wire contract，而不是 SDK 品牌名称。

Provider preset 保存官方 base URL、官方鉴权方式和官方推荐环境变量。AZ 地址和凭据不能
写进这些 preset。

### 2.2 AZ 路由身份

AZ `/v1/models` 返回的是可调用 route ID。route ID 可能是：

- 与官方 Model ID 完全一致的模型；
- 带 `urg`、`high`、`low`、`medium`、`az`、`ds` 等服务侧路由语义的别名；
- 带 `thinking`、`thinking-temp` 等服务侧行为选择的别名；
- `serpapi-*` 这类非 LLM 服务。

每个 route ID 在测试清单中单独存在，不因映射到相同官方模型而合并。测试调用始终使用
原始 route ID。

### 2.3 测试连接身份

AZ 只是一份连接投影：

```yaml
connection:
  base_url:
    env: AZ_BASE_URL
  credential:
    env: AZ_API_KEY
```

测试装配使用官方模型语义和 Pygent 协议实现，再以 AZ 连接替换官方连接。这个替换只存在于
显式 live conformance runner 中，不修改 `ModelSpec.provider`，不进入用户默认配置，也不改变
官方 Provider preset。

## 3. 冻结测试范围

2026-09-11 使用本地 `AZ_API_KEY` 请求当前 `AZ_BASE_URL` 的 `/models`，得到 211 个唯一
route ID。冻结规则为：

1. 按 route ID 做 ordinal 排序；
2. 每行写一个 UTF-8 route ID；
3. 文件末尾保留一个换行；
4. 对完整字节串计算 SHA-256。

本次快照标识为：

```text
count: 211
sha256: 7398801a0d480abc1b45d64d87e9c8eac508404f53dffc37d202c222f264330e
```

服务端声明的 endpoint 组合为：

| endpoint 组合 | route 数量 |
|---|---:|
| `openai` | 130 |
| `anthropic` + `openai` | 46 |
| `gemini` + `openai` | 28 |
| `openai` + `serpapi` | 7 |

实现阶段把 211 个 route ID 原样保存到版本化测试 manifest。实时 `/v1/models` 与 manifest
不一致时，runner 在执行任何付费调用前失败并输出新增、删除的 ID；它不会自动重写 manifest，
也不会把新 ID 偷渡进本次完成范围。

## 4. 官方目录准入规则

每个 AZ route 必须经过以下分类之一。

### 4.1 `official_model`

route ID 与官方 Model ID 一致，并且存在可引用的官方模型或 API 资料。该模型可以进入内置
官方目录。目录记录必须包含来源、确认日期、协议和完整 capabilities。

### 4.2 `gateway_alias`

route ID 是 AZ 的服务等级、渠道或行为别名。测试 manifest 保存：

```yaml
route_id: claude-opus-4-8-urg
kind: gateway_alias
canonical_provider: anthropic
canonical_model_id: claude-opus-4-8
protocols: [openai_chat_completions, anthropic_messages]
catalog_eligible: false
```

它引用 canonical model 的基础能力，再由测试场景记录别名实际暴露的协议差异。别名不进入
`ModelCapabilityCatalog`，也不伪装成 Anthropic 官方 Model ID。

### 4.3 `external_service`

`serpapi-*` 属于外部搜索服务，不进入 LLM Provider 或模型能力目录。它们仍保留在 211 项
manifest 中，并通过对应搜索入口逐项验证。

### 4.4 无法确认的名称

无法在官方资料确认、也无法可靠映射到 canonical model 的 route 不能猜测能力。必须通过
AZ 文档和最小真实请求收集证据，最终归入 `gateway_alias` 或 `external_service`。在分类完成前
不能进入官方目录，分类未完成也不能宣告 211 项验证完成。

## 5. Provider 与协议实现

### 5.1 Provider 批次

官方模型按以下批次建立或补全 Provider preset 与能力目录：

1. OpenAI；
2. Anthropic；
3. Google Gemini；
4. Alibaba Cloud（Qwen）；
5. DeepSeek；
6. Zhipu GLM；
7. Moonshot Kimi；
8. MiniMax；
9. ByteDance Volcano Engine（Doubao）；
10. xAI Grok；
11. 其余能确认官方归属的模型。

已有 Provider 和模型记录只按官方证据补充或修正，不复制一份 AZ 版本。

### 5.2 协议优先于 SDK 品牌

Adapter 按 wire protocol 注册。Provider 可以共享协议 Adapter，连接渠道也不决定 Adapter。

现有协议继续使用：

- `openai_chat_completions`；
- `anthropic_messages`。

只有当 211 项的真实能力无法由现有协议表达时，才新增协议。预期需要评估并按官方 contract
分别实现：

- OpenAI Responses；
- Gemini `generateContent` / streaming；
- embeddings；
- image generation/editing；
- speech synthesis/transcription；
- realtime WebSocket；
- video generation 的异步任务协议。

协议名称必须指向精确 contract。不能用 `openai`、`gemini` 这类模糊总称，也不能因为 AZ
声明所有 route 都有 `openai` endpoint，就假定所有能力都走 `/chat/completions`。

### 5.3 Pygent Adapter 与 raw conformance probe 的边界

当 Pygent 的公开请求/响应类型已经能够完整表达某项能力时，live runner 必须通过正式 Adapter
执行，以验证真实框架调用链。

当能力需要尚未属于 Pygent Agent 模型调用契约的产物类型，例如异步视频文件或独立搜索结果，
测试使用隔离的 raw protocol probe。raw probe 只验证该 route 和协议，不因此把临时 transport
逻辑暴露成公共 Adapter。以后若 Pygent 正式接纳该输出契约，再由独立 proposal 将其升级为
Adapter。

## 6. 能力资料与测试 profile

### 6.1 能力来源

能力证据按以下优先级使用：

1. 模型厂商官方模型目录或 API reference；
2. 官方发布、弃用或迁移文档；
3. 官方托管平台对该模型的协议说明；
4. AZ endpoint 元数据只用于决定测试入口，不作为官方能力真相；
5. live probe 只证明当前 route 的实际行为，不反向篡改官方能力定义。

官方来源 URL 和核对日期保存在测试侧的来源索引中，以
`(official_provider, official_model_id, protocol)` 为键。生产发布的模型能力 JSON 仍只保存完整
展开后的 capabilities，不扩张 `ModelCapabilityCatalog` 的运行时 schema，也不在运行时访问来源
URL。

### 6.2 场景分配

每个 route 有一个或多个必需 scenario，scenario 来自 canonical capabilities 和 route 暴露的
协议交集：

| 能力 | 必需验证 |
|---|---|
| 文本输入/输出 | 非流式文本生成 |
| text streaming | SSE 或协议原生流，至少收到增量和正常终止 |
| tools.call | 模型产生指定工具调用，并能接收工具结果继续生成 |
| tool choice | 分别验证该模型声明支持的可强制选择方式 |
| structured JSON | 返回值通过本地 JSON 解析；声明 schema 时同时通过 schema 校验 |
| reasoning | 响应包含协议定义的 reasoning/thinking 证据；可控时验证显式开关 |
| image input | 使用仓库内固定小图并得到与图像内容一致的文本结果 |
| image output/edit | 返回可下载且能解码的图片；编辑场景验证输入与输出均有效 |
| video output | 异步任务完成并返回可探测的视频资源 |
| audio output | 返回可解码、非空的音频 |
| audio input/ASR | 使用固定或刚生成的音频并获得非空识别文本 |
| realtime | 完成 WebSocket 握手、发送最小会话事件并收到合法服务端事件 |
| embedding | 返回非空、有限数值向量，维度在同一模型重复请求中稳定 |
| search | 返回合法搜索结果结构并至少包含一个结果项 |

一个 route 支持多个 endpoint 时：

- 官方 Provider 原生协议是必测协议；
- AZ 明确宣告的额外兼容协议也必测；
- 同一协议的 retry 不计为独立通过；
- 不适用于该 route 的场景不进入它的 required scenario 集合，而不是记作成功。

### 6.3 全部通过的精确定义

route 通过需满足：

```text
manifest 分类完成
AND required_scenarios 非空
AND 每个 required scenario 有本次 source revision 的 passed 记录
AND 没有更新、更直接或更真实的 active failed 记录
```

总体验收要求：

```text
通过 route ID 集合 == 冻结的 211 route ID 集合
```

不允许 `skipped`、`expected_failure`、只测 `/models` 或把多个别名折叠成一次调用来满足此集合。

## 7. Live conformance runner

### 7.1 文件职责

实现阶段按职责拆分：

```text
tests/live/az_conformance/
  manifest.json            # 冻结的 211 route、canonical 映射、协议和场景
  sources.json             # 官方模型/协议来源 URL 与核对日期
  schemas.py               # manifest/result 的严格不可变解析
  inventory.py             # 读取 /models、比较 count 与 digest
  runner.py                # 调度、断点续跑、限流、结果聚合
  openai_probes.py          # Chat/Responses/Embedding/Image/Audio probes
  anthropic_probes.py       # Messages probes
  gemini_probes.py          # generateContent probes
  media_probes.py           # 视频及其他异步媒体 probes
  search_probes.py          # SerpAPI probes
  cli.py                    # 唯一命令行入口
```

生产目录和 Adapter 留在 `src/pygent/llm/`，测试渠道代码不能被生产包导入。

### 7.2 断点续跑

runner 把结果写入用户显式指定的输出目录。结果键为：

```text
(snapshot_sha256, source_revision, route_id, protocol, scenario)
```

仅 `passed` 且键完全匹配的记录可在续跑时复用。代码 revision、manifest digest、协议或 scenario
变化后必须重测。失败记录保留用于诊断，但不会阻止后续重试。

### 7.3 并发和成本

- 文本、embedding 和搜索使用小输出上限与受控低并发；
- 图片、视频、音频、realtime 默认串行；
- 服务端 `Retry-After` 优先于本地 backoff；
- 只对 timeout、429 和可恢复 5xx 做有上限的自动 retry；
- 认证、参数、能力不匹配和内容校验失败不盲目 retry；
- 每次运行先打印待执行场景数量，不打印预计金额或无法由服务端确认的费用；
- 中断后已通过场景可安全续跑。

### 7.4 脱敏结果

公开结果只包含：

```yaml
snapshot_sha256: "..."
source_revision: "..."
route_id: "..."
canonical_provider: "..."
canonical_model_id: "..."
protocol: "..."
scenario: "..."
status: passed
attempts: 1
error_kind: null
```

允许的失败分类至少包括：

- `configuration`；
- `authentication`；
- `permission`；
- `rate_limit`；
- `invalid_request`；
- `protocol_mismatch`；
- `capability_mismatch`；
- `invalid_response`；
- `timeout`；
- `gateway_unavailable`；
- `upstream_unavailable`；
- `content_rejected`；
- `dependency_failed`；
- `unknown`。

私有诊断可在进程内包含异常类型、HTTP 状态和经过清洗的 provider error code，但不能包含：

- API Key 或请求鉴权头；
- `.env` 内容；
- 用户数据；
- 完整 prompt、响应正文、thinking/reasoning 正文；
- data URL、音视频字节或临时下载 URL。

## 8. 配置和安全

live runner 只接受环境变量引用：

```text
AZ_BASE_URL
AZ_API_KEY
```

它不修改 `.env`，不把真实值写进 manifest、日志、pytest snapshot、Git、wheel 或 sdist。base URL
必须是 HTTPS，不能带内嵌凭据。错误输出在序列化前统一脱敏，未知异常默认只保留类型名。

官方 Provider 的真实冒烟测试继续使用各自独立的官方凭据；AZ 测试通过不能替代官方连接测试，
只能证明同一模型语义和协议在该连接渠道上的兼容行为。

## 9. 测试与发布门禁

### 9.1 离线测试

离线测试覆盖：

- manifest 严格 schema、不可变性、重复 route 和未知引用；
- 211 count、排序和 SHA-256；
- 每个 route 都有分类、canonical 映射和非空 required scenarios；
- 官方目录准入规则，保证 alias 与 external service 不进入生产 catalog；
- protocol 到 probe 的全覆盖映射；
- 结果脱敏、错误分类、retry 和断点续跑；
- manifest drift 在任何付费请求前失败；
- required scenario 集合与 capabilities 一致；
- 汇总器必须精确证明 211/211，不能把 skipped 当 passed；
- 新增 Provider、Adapter、codec 和请求/响应 contract；
- wheel/sdist 包含官方目录，但不包含 live 凭据和运行结果。

### 9.2 真实验证

真实验证分批执行并最终合并为同一 revision、同一 manifest digest 的完整结果：

1. 文本与 streaming；
2. tools 与 structured output；
3. reasoning/thinking；
4. image/vision；
5. video；
6. audio/realtime；
7. embedding；
8. search；
9. 所有额外兼容协议。

最终报告必须列出 211 个 route 的通过状态和每类场景数量。任何 active 失败都按代码缺陷、配置
错误、AZ 网关限制或上游故障分类；只有对应场景获得更新的通过证据后，失败才可被 supersede。

### 9.3 仓库门禁

211 项全部通过后执行：

```text
uv run pytest -q
uv run ruff check src tests examples benchmarks
uv run mypy src benchmarks
uv build
uvx twine check dist/*
```

随后逐字节比较源码、wheel 和 sdist 中的内置目录 JSON，并扫描：

- AZ 地址或环境变量是否进入官方 preset；
- gateway alias 或 `serpapi-*` 是否进入官方模型目录；
- API Key、响应正文、媒体 URL 或 live 结果是否误入 Git；
- 新旧协议名称是否存在双格式兼容路径；
- 既有模型、Runtime、Worker、持久化和资源生命周期测试是否回归。

## 10. 分阶段交付

该目标拆成能独立验证但共同指向 211/211 的阶段：

1. 冻结 manifest、schema、drift 检查和脱敏结果契约；
2. 建立官方来源索引并完成 211 route 分类；
3. 按 Provider 批次扩充官方 preset 和模型能力目录；
4. 按协议批次补齐必要 Adapter 或隔离 raw probe；
5. 按能力批次执行 live conformance，持续修复真实失败；
6. 汇总同一 revision 的 211/211 证据；
7. 执行完整仓库与发布包验证。

每个阶段独立提交。任何阶段都不能以缩小 manifest、删除失败 route、降低 required scenarios 或
把平台元数据当官方能力资料的方式获得绿色结果。

## 11. 官方资料基线

第一轮分类至少使用以下官方入口，并在实现时为每条目录记录保存更精确的模型页：

- OpenAI Models：<https://developers.openai.com/api/docs/models>
- Anthropic model deprecations/status：<https://docs.anthropic.com/en/docs/about-claude/model-deprecations>
- Gemini models：<https://ai.google.dev/gemini-api/docs/models>
- Alibaba Cloud 文本模型：<https://help.aliyun.com/zh/model-studio/text-generation-model>
- DeepSeek API Docs：<https://api-docs.deepseek.com/>
- MiniMax API overview：<https://platform.minimax.io/docs/api-reference/api-overview>

官方资料会随时间变化，因此目录记录的“官方能力”以核对日期的证据为准；本次 AZ 验收范围则始终
以第 3 节冻结的 211 项快照为准。
