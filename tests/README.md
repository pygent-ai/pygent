# Pygent 0.2.x 测试目录

本目录只维护 Pygent 0.2.x 契约。测试覆盖公开 Core/Context/Runtime、LLM、Tool/MCP、Agent/ReAct、分布式恢复、示例和发布边界。

| 0.2 文档模块 | UT 目录 |
| --- | --- |
| `agent` | `tests/agent/` |
| `module` / `context` | `tests/core/` |
| `llm` | `tests/llm/` |
| `mcp` | `tests/mcp/` |
| `runtime` | `tests/runtime/` |
| `tool` | `tests/tool/` |

## 其他目录

- `tests/integration/`：跨两个或以上模块的公开 API、端到端和组合测试。
- `tests/support/`：测试服务器、契约加载器和共享 helper；不得包含可收集测试。
- `tests/conftest.py`：全局 0.2.x pytest 治理规则。

## 约定

- 不建立版本目录或旧版兼容目录；所有测试均属于当前 0.2.x 契约。
- 单模块行为测试放到拥有该公开契约的模块目录。
- facade 与底层机制分属不同模块时，以 facade 所有者放置公开 API 测试，
  底层模块只测试其拥有的调度、状态机或资源语义。
- 跨模块测试不得反向冻结其他模块的私有类型。
- 新增测试文件后，`pytest --collect-only` 的收集数量和
  `contract_02` marker 覆盖必须通过治理检查。
- `tests/integration/test_test_layout.py` 自动检查精简后的文档集合、根目录无平铺
  测试，以及 `support/` 不收集测试。

## 契约执行

- `uv run pytest -q` 执行完整 0.2.x 行为契约；任何失败都表示门禁未闭环，不能使用 xfail 隐藏。
- 源码树 subprocess 测试只证明导入与构造的零 I/O Unit seam，不构成
  `RM-SDK-01` 的安装后发布证据。发布门禁必须 build wheel、在隔离环境安装，
  并从工作区外执行发布版 Quickstart/SDK 示例。
