# Bash 有限等待与后台任务设计

状态：已实现并完成本地验证。本文保留已确认的核心需求、返回规则和正确性约束。

本文服从 [总第一原则](../FEATURES.md)、[Tool 第一原则](FEATURES.md)、[Runtime 第一原则](../runtime/FEATURES.md) 和 [Execution 契约](../EXECUTION.md)。Tool 第一原则仅扩展第 11 条，允许应用显式选择独立任务的有限等待。

## 1. 本次需求

- 为工具提供统一的有限等待与后台执行机制，暂时只在 Bash 中落地。
- 与现有 detach 联动：等待期间完成则返回最终结果；到期仍未完成则返回后台任务引用和 snapshot，原命令继续执行。
- 提供查询结果、停止任务的方法，其他 Agent 也能通过公共接口访问。
- 尽量复用 ToolTask、ToolResult、ToolTaskManager 和现有 Bash 进程管理逻辑。

已确认：本次 Bash 采用单一超时配置，默认 10 分钟，含义为前台等待时限；到期转为后台返回，不因这个超时取消或杀死命令。不会另设一个 Bash 执行硬超时与它竞争，也不能通过取消后重新提交命令实现转后台。

### 已确认的部署与重启范围

- 同一存活 Runtime 内，其他 Agent 可以通过公共任务接口查询和停止原管理器持有的任务；这是共享控制，不转移执行 owner。
- Runtime 重启后，只要求按任务引用查询已持久化的状态、输出快照和最终结果，不要求接管原 Bash 进程或恢复其控制连接，也不自动重新执行命令。
- 已提交的终态结果在重启后保持可查询。原 owner 已丢失且没有确认终态的任务按现有恢复契约记为 unknown，保留已持久化输出；unknown 不证明原进程已停止。
- 正常关闭 Runtime 仍履行已有取消、清理与结果保存责任。异常退出后只能查询已经持久化的记录，不承诺取回尚未保存的输出。
- 本次不要求新增跨进程实时控制或进程接管服务。持久化记录查询与执行恢复是不同能力。

## 2. 与 detach 联动

采用已经讨论的“独立 admission + 有限等待”结构：应用允许独立执行，任务管理器接纳并持有任务，前台观察同一个任务。

| 情况 | 返回行为 |
|---|---|
| 原有 sync | 保留同步调用语义 |
| 原有 detach | 保留默认立即返回任务引用的语义 |
| 应用选择有限等待，期间完成 | 返回最终 ToolResult |
| 应用选择有限等待，到期未完成 | 返回 detached ToolResult、ToolTask 快照及当时输出 |

已确认：受管 Bash 的 `is_background=True` 也通过同一套 detach admission 创建受管任务，立即返回任务引用和输出快照，不再只返回无人管理的进程 PID。普通调用等待超时后返回同一种 detached 结果。两种入口共享 task_id、执行 owner、结果保存、查询与停止机制，区别仅在是否先等待。

`is_background=True` 表达立即返回的请求，不自行授予独立执行权限；模型调用仍须通过应用或授权 Module 的生命周期授权。该参数不构成第二个超时时长配置。此处明确改变旧的外部进程行为，SDK 和测试需同步；direct 模式也使用同一任务机制，返回形式见第 3 节。

模型参数不能自行授予独立生命周期，仍由应用或授权 Module 决定。

已确认：前台等待时限由应用在工具装配时按工具配置，Bash 可以单独配置，默认时长为 10 分钟（600 秒）；本次不增加逐次授权的等待字段。该配置只决定已获独立执行授权的任务等待多久，不代替 lifecycle 授权。具体配置字段名和 SDK 签名需据此设计。

这次明确改变旧 Bash timeout 的“到期终止命令”语义：统一为“到期返回后台快照”。实现时需消除旧的 30 秒默认杀进程定时器和 10 分钟命令时限对该受管路径的竞争，也要核对 Bash ToolSpec 当前 timeout=610 的外层限制，不能在转后台后又被遗留的 Bash 超时配置终止。不会增加第二个面向使用者的 Bash 硬超时配置；其他工具的 ToolSpec.timeout 语义不因此全局改变。

框架已有且适用于任务的执行预算和清理规则仍须遵守，不能因前台等待到期暗中延长。Parent 的有效 deadline 限制 Parent 的等待，不自动成为已独立 admission 任务的命令计时器。后台任务可以自然完成，或通过显式停止、Runtime 关闭及适用的治理终止进入清理。

独立任务从 admission 起具有自己的身份与 owner。等待到期不重新执行命令、不更换 task_id，也不重置执行预算。观察者退出不终结独立任务；显式停止通过任务管理器请求取消。

## 3. 执行和结果归属

| 组件 | 职责 |
|---|---|
| 应用授权 Module / 受信 adapter | 决定是否允许独立执行，沿用现有授权边界 |
| ToolCallLayer | 校验、授权、admission 编排、有限等待及批量结果排序 |
| Runtime / ToolTaskManager | 持有执行任务，保存状态与结果，提供查询、取消和回收 |
| ToolRunner / ToolExecution | 保留唯一工具操作执行责任，处理超时、取消和最终结果归一化 |
| Bash executor / _BashProcess | 保管进程、管道和捕获资源，执行进程树清理 |

Bash 不新增独立任务注册表，Agent 不保存活进程或 manager。管理器持有执行 Task；前台返回时，仍在运行的命令和输出资源不能因调用栈退出而被关闭。

现有 InMemoryToolTaskManager 已有任务快照、执行 Task 和最终 ToolResult 存储，但仅靠内存不能满足重启查询。此次补充运行中输出快照访问及记录持久化，优先复用现有任务历史设施，不另建任务状态机。

任务身份、状态、输出快照与最终结果的持久化应复用现有任务历史设施；输出继续复用现有捕获逻辑。不能将需要重启查询的记录仅放在内存或随 Runtime 关闭删除的临时文件中。记录持久化不启用命令重放或进程接管，也不绕过现有沙箱与恢复能力校验。本文不规定新的日志分页协议、存储配额系统或存储后端。

### 已确认的 direct 装配与返回规则

开发者传入任务设施时复用该设施；未传入时，由 direct 工具装配对象自动创建并持有本地任务设施，不要求 Runtime，不创建隐藏的全局管理器。实际执行仍由任务设施持有，不能只依赖调用方保存返回的 handle。

| Python 直接调用 Bash 的情况 | 返回值 |
|---|---|
| 等待期限内完成 | 正常命令结果 |
| 等待到期仍在运行 | 活 handle，命令继续运行 |
| is_background=True | 立即返回活 handle |

活 handle 携带稳定 task_id，提供同一任务的控制引用，不创建第二个执行 owner。开发者仍可通过 tool_task_get(task_id) 和 tool_task_stop(task_id) 查询或停止任务；这两个工具必须使用与 Bash 相同的任务设施。装配对象持有自动创建的设施并负责其关闭，调用方负责关闭装配对象；外部传入设施的生命周期由原所有者负责。

Python 直接调用可以返回活 handle，但模型工具调用中的 ToolResult 仍是严格 JSON。工具适配边界必须在普通输出 schema 校验和 JSON 序列化之前识别该后台返回，将其转换为 detached ToolResult、task_id 和 snapshot；不能把 handle 放入 output、metadata、Context 或 wire，也不能放宽普通工具的 JSON 契约。需要同步调整 Bash 的 Python 返回类型及相应工具适配，不把所有工具的任意返回对象都视为句柄。

direct 和 managed 复用同一套执行、快照和控制逻辑；区别在部署资源归属和 Python/模型返回形式。自动创建的内存设施提供当前进程内管理，不因返回活 handle 就具有重启查询能力。第 1 节已确认的 Runtime 重启查询通过持久化任务记录实现；direct 若使用持久设施则查询该设施已保存的记录，不额外承诺自动内存设施在重启后恢复。

## 4. Snapshot、查询和停止

复用现有公开值：

- ToolTask 保存稳定 task_id、调用身份和当前任务状态。
- ToolResult.task 携带该快照；后台返回使用 status="detached"。
- ToolResult.output 携带当时捕获的输出，不冒充最终结果。
- 最终结果由同一工具执行产生，保留现有成功、失败、unknown 和副作用信息。
- 快照是不可变 JSON 值，不包含 Runtime、文件句柄或进程对象。

查询和停止优先复用现有入口：

```python
task = await runtime.get_tool_task(task_id)
result = await runtime.get_tool_result(task_id, wait=False)
cancelled = await runtime.cancel_tool_task(task_id)
```

运行中输出快照需要补齐读取能力，底层优先复用现有任务查询接口，不预设额外字段或版本协议。

已确认模型侧接口：

```text
tool_task_get(task_id)
tool_task_stop(task_id)
```

tool_task_get 立即返回当前任务快照和已保存输出，或已有最终结果，不等待任务完成。tool_task_stop 通过现有取消接口请求停止，不另建进程停止路径；停止未确认时不得报告已停止。两个工具薄封装同一任务管理接口，沿用应用授权，并在后台返回内容中提供调用提示。SDK 应补充显式装配示例，不另建生命周期逻辑。

其他 Agent 通过显式传入的任务引用访问公共接口，不依赖创建命令的 Bash 实例。访问沿用应用已有授权；业务会话如何保存和传递引用仍由应用负责。

## 5. 必须保持的正确性约束

以下来自现有第一原则及本次执行语义，不是额外产品功能：

- 命令只执行一次，task_id 和执行 owner 唯一；admission 与取消竞争不能遗失已启动任务。
- 有限等待不能把等待超时传播成任务取消。现有 get_result(wait=True) 路径须核查取消隔离后才能复用。
- managed 等待遵守 runnable lease 的释放与恢复。实际 Bash 仍占用应有工具资源；查询和停止不能因等待同一饱和资源而无法执行。
- 遵守框架已有有效 deadline 和有界清理规则；Bash 单一超时只控制前台等待，不保留与其竞争的旧 Bash 杀进程定时器，不新增第二套 Bash 预算配置。
- 显式停止由唯一 owner 触发 Bash 清理。清理未确认不能声称已停止，取消不代表撤销已有副作用。
- 输出继续遵守已有捕获上限、截断和管道排空规则；任务结束或所属资源关闭时履行回收责任。
- Parent 已结束的事件流不接收后续后台终态；返回 detached 不能把后台任务标记为执行成功。
- ToolCallLayer 保留 Context 和批量结果顺序，通过公开 Infrastructure SPI 接入治理，不新增私有 Runtime 访问路径。
- 现有 direct、managed、sandbox 和 durable 契约仍有效，不能把内存结果或本机 PID 解释为持久恢复能力。

## 6. 核心验收

1. 短命令返回最终结果；长命令到达等待时限后返回任务引用和 snapshot，命令没有重复启动。
2. 返回后台引用后仍能查询输出快照和最终结果，其他 Agent 能通过公共接口访问同一任务。
3. 等待到期或取消观察不停止独立任务；显式停止进入既有进程清理路径。
4. 默认等待 10 分钟后命令仍能后台继续，不被旧 Bash 30 秒、10 分钟或 ToolSpec 的遗留限制终止；适用的框架执行预算仍被遵守。完成与等待超时竞争不产生虚假成功或遗失结果。
5. 前台返回不会关闭后台仍在使用的输出资源，输出超限仍能排空管道。
6. 原有同步、默认立即 detach、批量排序和授权规则继续成立。
7. 正常关闭并重启后，仍能按原 task_id 查询已保存的输出和最终结果。
8. 原 owner 异常退出后，未确认完成的任务可查询为 unknown，并保留已持久化输出；查询不会重新执行或接管原命令。
9. 受管 Bash 的 is_background=True 立即返回与等待超时相同的任务引用和快照，后续查询、停止与重启查询行为一致；模型参数不能绕过独立任务授权。
10. tool_task_get(task_id) 立即返回当前快照或最终结果，tool_task_stop(task_id) 复用现有取消路径；两个入口都不重复启动命令。
11. direct 未传任务设施时自动创建本地设施；传入时复用。期限内返回正常结果，等待到期或显式后台返回活 handle；按该 handle.task_id 通过工具查询和停止的是同一次执行。
12. 同样的 Bash 后台返回经模型工具适配后为严格 JSON task_id/snapshot，不序列化活 handle；普通输出校验继续有效。关闭 direct 装配对象会清理其自动创建的设施，不能错误关闭调用方共享的外部设施。

实现验证复用现有 Bash 和任务管理测试，使用真实子进程检查执行、输出和清理，并保留已有生命周期回归。

## 7. 已确认范围与后续文档同步

部署与重启范围已在第 1 节确定；单一 Bash 超时、应用按工具配置、默认 10 分钟、到期不杀命令，以及 is_background=True 统一走任务机制，均已在第 2 节确定。direct 自动设施与活 handle 返回规则已在第 3 节确定，模型侧查询与停止接口已在第 4 节确定。

实现时同步 Tool SDK 的配置与两种调用返回示例、Runtime SDK 的任务查询与持久化记录说明，以及 Execution 契约和上述验收内容。具体类型和配置字段命名应服务于已确认语义，不据此新增功能。现阶段不新增长轮询、增量日志分页、存储配额治理或权限模型。

实现 API 已同步 Tool SDK、Runtime SDK、Execution 和 Durability 契约；项目地图的 flow、symbol 与最终验证记录在集成阶段同步。


## 8. 实现 API 对照

- `BashTools(workspace_root=..., timeout=600, task_manager=None)`；`timeout` 为秒，有限等待到期继续执行。`standard.shell.bash@3.1.0` 声明 `wait_timeout=600`、`timeout=None`，模型可传 `timeout` 覆盖本次等待时长，省略或传 null 时沿用装配配置；两者单位均为秒。
- `bash(command, working_directory=None, description=None, is_background=False, timeout=None) -> str | ToolTaskHandle`。`ToolTaskHandle` 提供 `task_id`、`snapshot()`、`wait(timeout=None)`、`result()` 和 `cancel()`；模型边界只输出 JSON。
- `.toolkit` 包含 Bash 和 `tool_task_get`、`tool_task_stop`；`StandardTools(bash_timeout=600, task_manager=None, ...)` 共装配十二个工具。装配对象提供异步上下文、`aclose()` 与异步 `close()`，只关闭自己创建的设施。
- `@tool(wait_timeout=...)` 或 `ToolKit(..., wait_timeouts={"bash": seconds})` 提供装配等待策略；模型调用还须显式 detach 授权。同步授权保持同步执行，显式后台参数不能提升授权。
- 运行中输出通过 `ToolExecutionContext.publish_output` 写入任务设施；公共读取为 manager `get_output` 和 Runtime `get_tool_output`。控制工具是受信 `task_control` adapter，不争用被目标任务占用的工具 permit，仍受 Execution 容量和授权约束。
- `DurableToolTaskManager` 使用现有 SQLite history 持久化普通任务观察记录。重启后查询不会重放或接管 Bash；当前 owner 租约约 30 秒，失效后未确认终态的任务为 unknown，并保留已提交输出。durable Job 执行恢复仍走已有独立契约。

## 9. 实现验证（2026-09-15）

- 全仓库 `pytest -q`：1319 项通过；随后补充 handle 适配用例，组合测试 6 项通过。
- `mypy src/pygent --follow-imports=silent`：89 个源文件通过；源代码及新增、相关测试 Ruff 检查通过。
- 真实子进程覆盖有限等待、持续输出、停止、关闭清理和历史库重开查询；任务设施回归覆盖观察取消隔离、owner 租约、并发启动及输出 schema。
- 本地 Windows 验证；本次没有重新执行跨平台 CI。重启只查询已保存记录，自动内存设施不提供重启查询。

## 10. 调用级等待覆盖

已确认恢复 Bash 的可选 `timeout` 入参：调用参数优先于装配配置，未配置时默认 600 秒；统一控制前台等待，到期继续后台运行。零值或 `is_background=True` 立即返回；负数、非有限值和非数字在启动前拒绝。direct 与 managed 共用解析函数，覆盖值不修改下一次调用的默认配置。通过 `wait_timeout_parameter` 显式声明参数映射，不重解释其他工具的 timeout，也不改变 detach 授权和执行预算。

调用级覆盖验证：本地全量测试 1327 项通过；Ruff 与 mypy 通过。覆盖 direct/managed、默认回退、零值、非法参数及编解码；调用级覆盖纳入 0.3.15 发布。
