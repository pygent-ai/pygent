# Native Shell 与 Interactive Terminal 设计

> 状态：设计提案。阶段 0（Shell 解析与身份、运行环境描述）与阶段 1（共享 Shell 进程引擎、Native
> PowerShell 执行）已实现，其余阶段未实现。本文只冻结边界与后续实现范围，不改变任何已发布的
> 稳定行为。
>
> 本文服从 [Pygent 0.3 第一原则](../FEATURES.md)、[Tool 第一原则](FEATURES.md)、
> [Runtime 第一原则](../runtime/FEATURES.md) 与 [Execution contract](../EXECUTION.md)，
> 并保持 [Tool SDK](SDK.md) 和 [Bash 有限等待与后台任务设计](BASH_BACKGROUND_DESIGN.md)
> 已确认的契约。冲突时以第一原则与正式契约为准。本文默认不修改
> `standard.shell.bash@3.1.0` 的稳定行为。

## 1. 核心原则

> **Native Semantics, Explicit State, Existing Control Plane**

- **Native Semantics**：Pygent 不翻译 Shell Language，不维护 Bash → PowerShell 等转换层；
  Agent 直接使用当前执行环境真实存在的 Shell 语义。
- **Explicit State**：默认 Shell 调用保持独立；需要持久状态时必须显式创建、显式持有
  owner、显式治理并发、显式结束，不得隐藏 Module 或 Tool 实例内部。
- **Existing Control Plane**：不新增 `TerminalManager`、`terminal.open()/write()/close()`
  这类第二套 Tool API；所有操作仍经 `ToolSpec` → `ToolCallLayer` → `ToolRunner` →
  `ToolTask` / `task_control` / `ExecutionInput`。

## 2. 背景

平台原生 Shell 存在差异：

```text
Linux   → bash / zsh / fish
macOS   → zsh / bash
Windows → PowerShell / pwsh / cmd
```

路径规则、环境变量、管道、重定向、引号、command substitution、exit status、encoding 与
进程行为在 Bash 与 PowerShell 之间没有完全等价的映射。统一抽象成 Bash，或做
Bash → PowerShell 语法翻译，都会引入新的语义层、转义层和平台缺陷面。

因此 Pygent 不实现 Universal Shell Language。但 Native Shell 不等于可以引入隐藏的持久
Shell 状态：现有契约要求调用独立、状态显式、Runtime 拥有 Execution 生命周期，所以
Shell 的"原生语义"与 Terminal 的"持久状态"必须分成两条路径设计。

## 3. 目标与非目标

### 3.1 目标

1. 支持不同操作系统的原生 Shell；
2. Agent 能明确感知当前 OS、Shell 与工作目录；
3. 不做 Bash / PowerShell 语法翻译；
4. 保持现有 Tool 调用独立语义，不引入隐藏调用状态；
5. 保持现有 ToolRunner / Execution / ToolTask 控制面；
6. 为未来的 PTY / ConPTY Interactive Terminal 提供扩展路径；
7. 明确 Persistent Terminal 的 ownership、sandbox 与 recovery 边界。

### 3.2 非目标

```text
Universal Shell Language
Bash → PowerShell Translation
Shell AST Translation
Transparent Persistent Shell
Cross-Execution Shell State
Durable Terminal Recovery
```

第一阶段不承诺 Agent 与人类 Terminal（UI / PTY 行为）完全一致，只保证 Agent 使用与目标
操作系统一致的 Native Shell Language 与 Runtime Semantics。

## 4. 与现有原则的关系

本方案不新增业务能力域。逐条对应关系如下，实现时不得偏离。

| 本方案条目 | 依据 |
|---|---|
| Native Shell Exec 每次调用独立、无跨调用 cwd/变量/alias | 总第一原则第 3、4 条 |
| Shell 选择结果 `ShellIdentity` 属于定义配置 | 总第一原则第 3 条（Module 可以持有定义配置与显式部署资源） |
| 不新增平行 Tool API，ToolRunner 仍是唯一 Tool Operation Owner | Tool 第一原则第 6、8 条；[Execution contract](../EXECUTION.md) "Model and tool ownership" |
| Terminal 是显式 ToolTask，不是 Module 或 Agent Session 状态 | Tool 第一原则第 8、11 条；总第一原则第 11、12 条 |
| 外部交互输入进入已有 ExecutionInput 通道 | 总第一原则第 18 条；[Execution contract](../EXECUTION.md) "Managed Runtime inputs" |
| direct execution 不接受外部持续输入 | 总第一原则第 18 条 |
| workspace 校验继续由每次调用参数解析保证 | Tool 第一原则第 12 条；[Tool SDK](SDK.md) 标准工具段 |
| `sandbox_profile` 只是隔离需求声明，不构成已实施证明 | Tool 第一原则第 12 条 |
| 不承诺 durable process recovery | 总第一原则第 16 条；[Execution contract](../EXECUTION.md) 任务观察段 |
| System Prompt 不被框架动态改写 | Agent 第一原则第 11、13 条 |

## 5. 总体结构

```text
                        Agent
                          │
                          ▼
                       ToolSpec
                          │
                          ▼
                      ToolRunner
                          │
             ┌────────────┴─────────────┐
             ▼                          ▼
      Native Shell Exec         Interactive Shell Task
             │                          │
       Stateless Call                 ToolTask
             │                          │
             │                   Managed Execution
             │                          │
     ┌───────┴────────┐                 ▼
     ▼                ▼            PTY / ConPTY
   bash             pwsh                │
     │                │            Native Shell
     ▼                ▼                 │
  Process          Process        ExecutionInput（外部输入）
     │                │                 │
     ▼                ▼                 ▼
 ToolResult       ToolResult     input 工具调用 → PTY stdin
```

两条路径职责不同：Native Shell Exec 是默认的一次性执行；Interactive Terminal 是显式的
受管 ToolTask，只在需要交互或长驻进程时使用。

## 6. Native Shell Exec

### 6.1 调用语义

> **One Call → One Shell Process**

```text
Agent → ToolRunner → Native Shell → Process → ToolResult（进程退出）
```

调用结束即进程结束，因此不存在跨调用的 `cwd`、Shell Variable、Alias、Function 与 Shell
History。现有 `standard.shell.bash@3.1.0` 已经是这一形状（`src/pygent/tool/standard/_bash.py`
中每次调用以 `[<shell>, "-lc", command]` 启动独立进程），本方案不改变它。

该路径不需要 Command Boundary Protocol：命令边界就是进程退出，exit code 来自进程本身，
无需在输出流中插入内部标记。

### 6.2 Shell 选择

新增 `ShellResolver`，在 **Tool 装配阶段**确定 Shell。它属于定义配置，不是调用状态。
选择顺序：

```text
显式装配参数
      ↓
应用配置
      ↓
环境探测
      ↓
平台默认
```

解析结果的稳定形状如下（字段为建议，冻结前属于实现范围）：

| 字段 | 含义 |
|---|---|
| `platform` | `windows` / `linux` / `darwin` |
| `name` | `bash` / `zsh` / `powershell` / `cmd` |
| `version` | 探测到的 Shell 版本，探测失败时为 `null` |
| `executable` | 绝对路径 |
| `args` | 固定启动参数（例如 `-NoLogo`） |

公开装配方式沿用现有标准工具的 Python 装配风格，例如
`BashTools(workspace_root=..., bash_executable=...)` 与 `PYGENT_BASH_PATH` 环境变量覆盖
（`src/pygent/tool/standard/_bash.py`）；仓库没有配置文件加载器，YAML 只作为应用自己配置层
到这些参数的映射说明，不是框架 API。

现有实现已经包含装配期的功能性探测（`bash -lc` 探针），`ShellResolver` 沿用同一思路：
探测失败必须回退到平台默认或显式报错，不得静默选择一个不可用的 Shell。

### 6.3 Shell Identity 的放置

`ShellIdentity` 是定义配置：

- 它可以陈述在 Runtime Context 中（见 6.4），可以出现在装配日志与诊断中；
- 它**不得**被加入 `ToolSpec` / `ToolDefinition`。这两个公开值是携带封闭字段集的可移植
  wire 值（`src/pygent/runtime/codec.py` 对 `ToolSpec` 使用 `_only(...)` 校验），新增字段
  属于独立管理的 schema 变更；
- 它不包含 `cwd`、环境变量变更、Shell 变量、history 或进程状态。

### 6.4 运行环境感知

Agent 需要知道当前执行环境，但框架不能改写 Agent 定义。复用已有机制：

- `InjectionKind.RUNTIME_CONTEXT` 与 `format_context`（`src/pygent/agent/reminder.py`）；
- `Reminder` 是无 session 状态的普通 Module，由应用显式组合（`docs/agent/SDK.md`）；
- 无 Module 时可直接投递带该 kind 的 UserMessage。

框架只陈述事实（OS、Shell 名称与版本、可执行文件、工作目录），**不**注入
"Use PowerShell syntax" 这类行为指令；是否要求模型遵循 Native Shell 语法由应用自己的
Agent Definition / System Prompt 决定。模型层不得替换 System Prompt，这一点由 ReAct 现有
校验保证（`src/pygent/agent/react.py`）。

### 6.5 Command 透传

Agent 生成 `command: string`，执行链路保持：

```text
Agent Command → Shell Tool → ShellAdapter → Native Shell
```

不做 `parse → translate → normalize → reconstruct`，也不做 Bash → PowerShell 转换。
Adapter 只负责安全的进程调用，不改变命令语言本身。

### 6.6 ShellAdapter 职责边界

| 可以负责 | 禁止负责 |
|---|---|
| executable discovery | `translate_command()` |
| startup arguments | `bash_to_powershell()` |
| encoding 处理 | `powershell_to_bash()` |
| process invocation | `rewrite_shell_ast()` |
| exit status 提取 | 命令语义的二次解释 |
| interactive protocol 支持 | 模拟 Shell 行为 |

ShellAdapter 是 Execution Adapter，不是 Language Translator。

### 6.7 Working Directory 与 Workspace

每次调用继续执行现有边界校验：

```text
working_directory → Path Resolution → Workspace Boundary Validation → Native Shell Process
```

默认 Stateless Native Shell 不得弱化现有 `sandbox_profile="workspace"` 行为。现有路径归一化
（含 Windows 上的 MSYS 风格路径）与越界拒绝逻辑必须保留其语义；新增的非 Bash Shell 需要
各自说明路径与编码处理范围，不能假定 Bash 的归一化规则可直接复用。

### 6.8 Structured Exec 与 Shell Exec 分离

```text
Execution
   ├── Structured Exec   argv[]        → 直接进程调用
   └── Shell Exec        command 字符串 → Native Shell
```

框架已知的确定性调用（例如内部的 `git status`）优先使用 argv 形式，避免引号、转义、注入和
平台差异；只有需要 Shell Language Semantics 的 Agent 操作才经过 Native Shell。

## 7. Interactive Terminal

### 7.1 定位

需要 Python/Node REPL、debugger、ssh、interactive CLI、长驻 `npm run dev` 等场景时使用
Interactive Terminal。它不是 Module 的隐藏状态，必须建模为显式的受管 ToolTask：

```text
Managed Execution
      │
      ▼
   ToolTask
      ├── task_id
      ├── PTY / ConPTY
      └── Native Shell Process
```

> Persistent Terminal 是显式 ToolTask，不是 Agent Session Resource。

### 7.2 所有权与生命周期

```text
Execution E1  ──owns──▶  ToolTask T1  ──owns──▶  PTY/ConPTY  ──▶  Native Shell Process
```

- 每个 Terminal ToolTask 有唯一 owner Execution 与稳定 `task_id`；
- Terminal 不因为某个 Module 实例仍存在而隐式存活；
- 生命周期沿用 ToolTask / Execution 契约，Runtime 负责资源释放；
- 生命周期控制复用现有 `task_get` / `task_stop` 一族，不新增
  `terminal.list()/stop()/close()`；
- 多个 Terminal 由多个 ToolTask 表达（同一个 Terminal 工具的多次独立 admission），
  跨调用引用统一使用 `task_id`，不引入 `terminal_id` 业务参数；
- Module 不自行维护 Process Pool、Terminal Pool、Terminal Manager 或 Semaphore。

### 7.3 资源治理

Interactive Terminal 复用 `ToolSpec.resource_key = "shell"` 与 Runtime 的 per-Binding 共享
物理工具门禁（[Runtime SDK](../runtime/SDK.md) 关于 `CapacityPolicy.capacity_key` 与
`ToolSpec.resource_key` 的说明；`src/pygent/runtime/local.py`）。

需要明确：该共享门禁只在 Binding 声明了带 `max_concurrency` 的 Tool 容量策略时叠加到
Binding 总门禁之外；默认配置下不存在本地 limiter。因此容量与并发治理是应用显式声明的部署
决策，不能假定默认生效。

### 7.4 输入路径

不新增 `terminal.write()` 这类平行接口。写入运行中 Terminal 的 stdin 只有一条落点：受信的
input 工具调用。它属于与 `task_control` 同族的受信适配器（`src/pygent/tool/control.py`），
沿用既有可见性、授权、deadline 与结果校验，并按 `task_id` 解析目标进程。

两种输入来源分别走各自既有通道：

| 输入来源 | 通道 |
|---|---|
| 模型自发输入（例如向 REPL 发送一行代码） | 直接的 input 工具调用 |
| 外部/人类输入 | `send_input()` → Execution inbox → 唯一消费该 `kind` 的 Module → 同一个 input 工具调用 |

外部输入必须进入已有 ExecutionInput 模型（有界、有序、幂等、单消费者、managed-only）。
`send_input()` 不得被解释为直接写进程；目标 kind 只能由一个 Module path 消费，因此交互输入
的 `kind` 必须与 Agent 的普通 prompt 输入区分开。direct execution 拒绝 `send_input()`。

### 7.5 Direct Execution 限制

> Interactive Terminal 第一阶段只支持 Managed Execution。

Direct Execution 继续只使用 Native Shell Exec，不支持外部交互输入、持久 PTY 交互与
`send_input`。不为支持 Terminal 而修改 Direct Execution 的既有语义。这是范围决策而非机制
限制：direct 缺少的是外部输入通道与 Runtime 资源治理，模型自发的工具调用本身在 direct 中
仍然可用（参见 Bash 在 direct 中由装配对象持有本地任务设施的先例）。

### 7.6 Timeout 与 Interrupt

严格区分四个概念，不得混用：

| 概念 | 含义 | 归属 |
|---|---|---|
| Shell foreground timeout | 已有 Bash 语义：前台等待到期转后台，不杀进程 | `ToolSpec.wait_timeout` / 调用级覆盖 |
| Execution timeout | Execution 的 `ToolSpec.timeout` 执行截止 | ToolRunner |
| ToolTask stop | 通过任务控制请求停止 | 任务管理器 / Runtime |
| Interactive interrupt | 向交互进程发送中断（例如 Ctrl+C） | 任务控制路径 |

本方案不修改 `standard.shell.bash@3.1.0` 的 timeout 契约，也不引入"timeout → Ctrl+C"这第三种
语义。Interactive Terminal 的 interrupt 与 stop 必须通过任务控制明确表达。

### 7.7 公开值约束

- Terminal 的 owner Execution 身份、`resource_key` 等执行事实记录在 `ToolTask.metadata`
  或 durable Job 记录中；`ToolTask` 的公开快照字段是封闭集合（`task_id`、`call_id`、
  `tool_id`、`version`、`state`、`job_id`、`metadata`），不新增必填公开字段；
- 进程丢失等未确认终态复用已有 `ToolTaskState.UNKNOWN`，不新增状态值；新增状态属于公开
  枚举与 wire 变更；
- Terminal 的输入输出是流，不是按命令切分的结果。实现不得把流式输出伪装成一次性
  `ToolResult`，也不得把内部标记写入模型可见输出。

## 8. Working Directory Validation 与进程树隔离

持久 Terminal 带来额外隔离问题：初始 `cwd` 位于 workspace 并不能阻止随后的 `cd ..`。

> Working Directory Validation ≠ Persistent Terminal Filesystem Confinement

如果 Interactive Terminal 声明 `sandbox_profile="workspace"`，必须真正存在针对整个 Shell
进程树的 filesystem confinement（OS sandbox、container、namespace、受限令牌、文件系统虚拟化
等）。在该能力落地之前：

- Interactive Terminal 不得声明与 Stateless Bash 相同的 workspace confinement；
- managed 部署不会自动为它派生 `tool.sandbox.<profile>` capability，也无法据此声称隔离；
- ToolSpec 的 profile 字段是隔离需求声明而非实施证明（Tool 第一原则第 12 条）。

## 9. Capability Matrix

| Capability | Native Shell Exec | Interactive Terminal |
|---|---|---|
| Native Bash / PowerShell / Zsh | Yes | Yes |
| Call Independence | Yes | No（显式 ToolTask 状态） |
| Hidden Module State | No | No |
| Working Directory Validation | Yes | 仅初始 |
| Workspace Filesystem Confinement | 现有语义 | 需要专门的进程树隔离实现 |
| Persistent Shell State | No | Yes |
| Interactive Input | No | 仅 managed |
| PTY / ConPTY | No | Yes |
| ToolTask Control | 现有后台任务模型 | 是 |
| Durable Process Recovery | N/A | No |

## 10. Durable Recovery

> Interactive Terminal 明确不支持 Durable Process Recovery。

Runtime 崩溃后可以恢复 Execution 元数据、ToolTask 元数据与已提交结果；不能恢复 Shell 进程、
`cwd` 变更、环境变量变更、Shell 变量、REPL 状态或子进程状态。恢复后按现有语义进入
终止/丢失状态，未确认终态记为 `unknown`，已提交输出保持可查询，且查询不会重放命令或接管
原进程。框架不承诺，也不得伪造 Terminal Recovery。

## 11. 实现范围与文档同步

实现时的落地顺序与归档要求（沿用 [Bash 有限等待与后台任务设计](BASH_BACKGROUND_DESIGN.md)
的既有做法）：

1. 新 Shell 与 Terminal 能力使用新的 `tool_id` + `version` 身份，`standard.shell.bash@3.1.0`
   的语义与版本不变；具体工具命名与参数在实现前冻结。
2. 同步 [Tool SDK](SDK.md)：新工具的装配参数、返回形状、等待与停止用法。注意该文档中部分
   句子被 `tests/integration/test_tool_sandbox_documentation_contract.py` 逐句断言，新增段落
   不得改写这些句子。
3. 同步 [Runtime SDK](../runtime/SDK.md)：仅当需要补充任务查询、持久化记录或 ExecutionInput
   的使用示例时。
4. [Execution contract](../EXECUTION.md)：只有当 Interactive Input 被提升为公共投递 SPI，
   或任务控制家族新增需要澄清的成员时才需要修改；若输入只是"Module 消费 ExecutionInput 后
   调用普通 input 工具"，则不需要改动执行契约。
5. 总第一原则与 [Tool 第一原则](FEATURES.md) 不需要修改：本方案没有新增或推翻原则条目。
6. 验收覆盖：ShellResolver 探测与回退、无论哪种 Shell 的 workspace 越界拒绝、调用独立性
   （连续两次调用不共享 `cwd`/变量）、`wait_timeout` 与后台任务语义不变、任务查询与停止不
   重复启动命令、`sandbox_profile` 未实现时不产生虚假 capability。

## 12. 建议的工具身份与装配形状（未冻结）

现有身份不变：`standard.shell.bash@3.1.0` 声明 `wait_timeout=600`、无执行硬超时、
`resource_key="shell"`、`sandbox_profile="workspace"` 与 `required_permissions=("shell:execute",)`。

建议的新增身份如下，命名与参数在实现前冻结，本文不作为已发布契约：

| 能力 | 建议身份 | 说明 |
|---|---|---|
| Native Shell Exec（非 Bash） | `standard.shell.<shell>`（按 Shell 家族独立，例如 `standard.shell.powershell`、`standard.shell.zsh`） | Shell Language、`ToolDefinition` 描述、可用参数与编码处理各不相同；不合并成带 `shell` 参数的通用工具，避免把 Shell 选择变成调用状态 |
| Interactive Terminal | `standard.shell.terminal` | detach 生命周期，由应用或授权 Module 决定；不作为同步调用 |
| Terminal 输入 | `standard.shell.terminal_input` | 受信适配器，与 `task_control` 同族；按 `task_id` 写入目标 stdin |
| 任务查询与停止 | 复用 `standard.shell.task_get` / `standard.shell.task_stop` | 不新增第二套控制入口 |

装配沿用标准工具的 Python 形状（示意，非冻结）：

```python
tools = NativeShellTools(workspace_root=..., shell=resolved_shell_identity)
tools = TerminalTools(workspace_root=..., shell=..., task_manager=None)
```

新工具必须各自声明 `wait_timeout` / `timeout` / `required_permissions`，不得隐式继承 Bash
的取值；`ToolSpec` 的 `sandbox_profile` 只在对应隔离真实存在时声明。

## 13. 分阶段实现与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| 0 | `ShellResolver` + 运行环境描述（`describe_shell_environment`，由应用的 `Reminder` / `InjectionKind.RUNTIME_CONTEXT` 注入）。已实现 | 探测与回退可用；注入不改变 System Prompt 与消息权限 |
| 1 | Native Shell Exec 支持目标 Shell。已实现 PowerShell | 调用独立（连续两次调用不共享 `cwd`、变量、alias）；workspace 越界仍被拒绝；输出投影上限、截断与管道排空规则不变 |
| 2 | Interactive Terminal（Pipe 后端、仅 managed、受信 input 工具） | 查询、停止与输入指向同一 ToolTask，不重复启动命令；等待到期不改变生命周期语义；direct 明确拒绝外部交互输入 |
| 3 | PTY / ConPTY 后端 | 交互与回显能力边界在 SDK 中显式声明，不宣称与人类终端一致 |
| 4 | 进程树级隔离与 profile 声明 | 未实现前不得声明 `"workspace"`；capability 校验与实际隔离一致 |

跨阶段不变量：命令只执行一次；任务身份与 owner 唯一；输出与进程资源回收遵守既有上限与清理
预算；不承诺 durable process recovery。

阶段 0 实现记录（2026-09-19）：`src/pygent/tool/standard/_shell.py` 新增 `ShellIdentity`、
`ShellResolver` 与 `describe_shell_environment`；`_bash.py` 的候选探测与回退改由 resolver 执行，
优先级为显式 `bash_executable`、`PYGENT_BASH_PATH`、平台候选、平台回退，行为与
`standard.shell.bash@3.1.0` 保持一致；`BashTools.shell_identity` 暴露解析结果，`bash_executable`
仍是同一路径；`ShellIdentity` 与 `describe_shell_environment` 经 `pygent.tool` 与
`pygent.tool.standard` 导出。`tests/tool/standard/test_shell_resolver.py` 覆盖优先级、探测回退、
平台名、版本探测缓存与事实文本；`tests/tool` 与相关 integration 测试 350 项通过。

阶段 1 实现记录（2026-09-19）：新增 `src/pygent/tool/standard/_process.py`，承载与 Shell 无关的
进程与输出引擎（有界捕获、清理预算、进程树终止、结果投影、UTF-16/代码页解码）；`_bash.py` 改为
委托该引擎，模块级常量、`pygent-bash-*` 任务名与既有测试断言的属性保持不变；新增
`src/pygent/tool/standard/_powershell.py`（`standard.shell.powershell@1.0.0`、`PowerShellTools`），
每次调用启动独立 `-NoLogo -NoProfile -NonInteractive -Command` 进程，共用 workspace 边界校验、
`resource_key="shell"` 与前台等待语义，且不进入 `StandardTools` 默认可见集合；工作目录校验与候选去重
分别抽到 `_paths.resolve_workspace_directory` 与 `_shell.append_unique_path` 单一实现。
`tests/tool/standard/test_powershell_tool.py` 20 项通过（真实 PowerShell 进程覆盖状态不跨调用、
原生命令退出码、UTF-16 中文输出、workspace 越界拒绝、前台等待转后台、取消清理与共享任务设施）；
`tests/tool/standard` 全量 259 项通过。

## 14. 未决问题

1. **Interactive Input 的落点**：按本文采用"受信 input 工具调用"（现有控制面内，不改执行
   契约）；若改为公共投递 SPI，需要先修改 [Execution contract](../EXECUTION.md)。
2. **非 Bash Shell 的路径与编码范围**：现有 MSYS 路径归一化与 UTF-16 解码逻辑是 Bash 专属，
   新 Shell 需要各自的可验收范围。
3. **Interactive Terminal 是否开放 direct 支持**：本文按保守范围只支持 managed。
4. **隔离 profile 的命名**：在进程树级隔离实现前，Interactive Terminal 使用什么 profile 名
   （或暂不声明）需要与 Runtime 的 capability 校验一起确定。
5. **PTY 后端的能力边界文案**：Pipe 与 PTY/ConPTY 下的 prompt、回显、编码与信号行为差异，
   必须在 SDK 中显式声明，而不是让使用者假定与人类终端一致。
6. **工具身份粒度**：第 12 节的建议是"按 Shell 家族各自独立 `tool_id`"；若改为参数化单个
   Native Shell 工具，需要同时说明 Shell 选择为何不属于调用状态。
7. **Interactive Terminal 的默认授权**：输入与交互入口默认可见还是需要应用显式授权
   （模型不得自行提升生命周期，这一点由现有 detach 授权规则决定，不新增授权种类）。
