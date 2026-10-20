# 消息投影统一重构：规格与验收（WP0 规格）

状态：执行中。本文是本次重构的唯一验收依据；工人与指挥以本文为准。

## 1. 目标与铁律

目标：把 Message 树的四份手写投影收敛为一个核心投影器，**行为零变化**。

铁律（违反即验收不通过）：

1. **字节级等价**：每个消费点对黄金语料中任意样例的输出，与
   `tests/support/message_projection_golden.json` 中录制的黄金输出
   `json.dumps(sort_keys=True, ensure_ascii=False)` 后逐字节相等。
2. **文件所有权**：每个工作包只允许修改自己的文件清单（见各 WP）。
3. **禁止顺手改进**：发现任何"想修的东西"（bug、风格、疑似缺口）记录上报，
   不得在本次重构中处理。
4. 不新增/删除/改名任何公开 API；不新增任何功能；不改变任何错误行为。

## 2. 选项矩阵（行为规格，以黄金语料为权威）

四个消费点的现状差异收敛为 `MessageFieldOptions`：

| 维度 | codec(wire) | layer(effect) | snapshot(digest) | agent(估算) |
|---|---|---|---|---|
| base 含 data/metadata | ✅ | ❌ | ❌ | ❌ |
| usage | ✅ | ❌ | ❌ | ❌ |
| tool_call 身份 | full | full | full | minimal(call_id/name/arguments) |
| tool_result 元数据 | full | full | minimal | minimal |
| media(UserMessage) | full | ❌(预存缺口，保持) | identity_drop_key | ❌(omit) |
| media(AIMessage) | full | full | identity_drop_key | ❌(omit) |
| media(ToolMessage content) | full | full | identity_drop_key | identity_null(base64_data=None) |
| continuation | full | full | digest | ❌(omit) |

术语：
- **full**：完整字段；MediaBlock.source 含 base64_data。
- **identity_drop_key**：MediaBlock 值 dict 的 source **不含 base64_data 键**（snapshot 现状）。
- **identity_null**：保留 base64_data 键、值为 None（agent 现状 `_estimated_content_value`）。
- **digest**：continuation 投影为 `continuation_digest`（实现从现有
  `llm/_request_snapshot.py::_continuation_digest` 原样移植，黄金语料验证等价）。
- 已知预存行为（**保持原样，不在本次修复**）：layer effect 对 UserMessage 的
  media 不编码（黄金语料已固化该行为）。

## 3. 核心模块契约（WP-core 交付物）

`pygent/core/_message_fields.py`：

```python
@dataclass(frozen=True, slots=True)
class MessageFieldOptions:
    include_data_metadata: bool
    include_usage: bool
    tool_call_identity: Literal["full", "minimal"]
    tool_result_metadata: Literal["full", "minimal"]
    media_payload: Literal["full", "identity_drop_key", "identity_null", "omit"]
    continuation: Literal["full", "digest", "omit"]

# 预设（与四个消费点一一对应）
CODEC_OPTIONS / LAYER_EFFECT_OPTIONS / SNAPSHOT_OPTIONS / AGENT_ESTIMATE_OPTIONS

def project_message_fields(message, options) -> dict[str, object]
def message_fields_expected_keys(role, options, message_type) -> frozenset[str]
def message_from_fields(value, options) -> Message   # 严格 decode：未知字段显式报错
```

要求：
- 覆盖全部 Message 子类型：Message / UserMessage / AIMessage / ToolMessage /
  ToolAuthorizationRequest / ToolAuthorizationDecision。
  ToolAuthorizationRequest/Decision 仅在 codec 档全量编码；其余档投为基础字段
  （与现状一致）。
- decode 侧严格字段集从选项派生，且必须与现有两组字面集合逐一相等：
  - codec assistant：common + {tool_calls, usage, continuation, media}
  - layer effect：`_EFFECT_MESSAGE_FIELDS`
- 叶子编解码复用 `core/_tool_values.py` 既有 helpers
  （`_tool_result_content_to_value/_from_value`、`_tool_call_*`）。
- `pygent/core/__init__.py` 导出新符号。

### WP-core 验收
1. 反射完整性测试：`dataclasses.fields()` 的每个字段在各档投影键中出现
   （omit 档除外，逐字段声明）；常驻 `tests/core/test_message_fields.py`。
2. 黄金测试绿（消费点未迁移时天然绿）。
3. 全量测试绿；ruff/pyright 触碰文件干净。

## 4. 黄金语料（WP0 交付，已冻结）

- `tests/support/message_projection_corpus.py`：12 个规范样例
  （全部子类型 × media 三来源 × tool_call 全身份 × 失败/取消/-detached 结果 ×
  continuation × authorization）。
- `tests/support/message_projection_golden.json`：四个消费点对全部样例的
  现状输出（`json.dumps(sort_keys=True, ensure_ascii=False)` 字符串），
  由重构前的实现录制，仅此一次。
- `tests/llm/test_message_projection_golden.py`：常驻黄金测试——每个消费点
  对每个样例断言与黄金输出逐字节相等；codec/layer 附 decode 往返断言。

黄金测试在 WP0 提交时必须绿（它钉住的是"现状"）。此后任何迁移若改变输出，
黄金测试红 = 验收不通过。

## 5. 工作包、文件所有权与验收

| WP | 文件所有权 | 交付 | 验收（除铁律 1 外） |
|---|---|---|---|
| WP-core | `pygent/core/_message_fields.py`（新）、`pygent/core/__init__.py`、`tests/core/test_message_fields.py`（新） | §3 全部 | §3 验收三条 |
| WP-codec | `src/pygent/runtime/codec.py` | `_message_value/_message_from_dict` 内部改调核心；对外签名不变 | 黄金测试绿；全量绿 |
| WP-layer | `src/pygent/llm/layer.py` | `_message_effect_value/_message_from_effect` 内部改调核心 | 同上 |
| WP-snapshot | `src/pygent/llm/_request_snapshot.py` | `_message_projection` 内部改调核心 | 同上 |
| WP-agent | `src/pygent/agent/pygent_agent.py` | `_message_projection` 内部改调核心 | 同上 |

依赖：全部 WP 依赖 WP-core 合入 main。四个消费 WP 互相独立，可并行
（文件不相交）。

## 6. 门禁（每 WP 合入 main 前逐条过）

1. 黄金测试逐字节相等（铁律 1）。
2. 全量测试绿（`uv run pytest tests`，排除三个预存环境失败）。
3. ruff / pyright 触碰文件干净。
4. 指挥 diff 评审：确认无越界改动、无顺手改进。

验收不通过：退回该 WP 工人，附具体失败证据。

## 7. 完成定义（Done）

1. 四份手写 walker 删除，全部消费点走 `core/_message_fields.py`。
2. 反射测试与黄金测试常驻。
3. 全量测试绿、黄金全等价、无任何行为变化（含 digest/pinned 值零改动）。
