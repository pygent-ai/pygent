# Runtime SDK

本文是 Runtime 的第二级契约，必须服从 [Runtime 第一原则](FEATURES.md)。只有需要托管并发、资源、placement、结构化取消或恢复时才使用本 SDK；普通本地调用从 [Module SDK 的直接执行](../module/SDK.md#直接执行) 开始。本文的语义、使用方式和示例调用形状都是实现必须收敛的二级契约；尚未实现的 API 不得由实现自行改变语义或静默换名，需要调整时必须先更新本契约。本文中的 RESUME 只表示活跃 coroutine 重新获得 execution lease；故障重试和持久化恢复见 [持久化与恢复边界](DURABILITY.md)。

## 创建部署治理域

推荐先在 Runtime 中创建服务级 Binding，再把 Root Agent/Module 接入该治理域：

```python
# DEPLOYMENT means one shared budget across Runtime instances.  The in-memory
# coordinator is the LocalRuntime reference implementation for one process.
coordinator = InMemoryCapacityCoordinator()
runtime = LocalRuntime(capacity_coordinator=coordinator)

binding = runtime.create_binding(
    name="interactive-service",
    execution_capacity=ExecutionCapacityPolicy(
        scope=CapacityScope.DEPLOYMENT,
        max_live_executions=128,
        max_runnable_executions=8,
        max_queue_size=64,
        max_waiters=128,
        max_child_depth=8,
        max_children_per_execution=32,
    ),
    model_capacity=CapacityPolicy.passthrough(
        capacity_key="openai-account-a",
    ),
    tool_capacity=CapacityPolicy.limited(
        scope=CapacityScope.DEPLOYMENT,
        max_concurrency=32,
        max_queue_size=128,
        capacity_key="weather-account-a",
    ),
)

bound = binding.bind(agent)
```

这里的 Binding 属于整个 `interactive-service`，而不是只属于 `agent`。如果服务有多个 Root 入口，可以显式接入同一个治理域：

```python
chat = binding.bind(chat_agent)
admin = binding.bind(admin_agent)
```

`chat`、`admin` 及其各自的原始结构化 Child 共享该 Binding 的 Execution 容量。原始子 Agent 不调用 `bind()`；它在 `forward()` 中被调用时自动继承 Parent 的 Binding、lineage、deadline、取消 scope 和容量预算。显式预绑定 Child 与 placement policy 的隔离形式见下文。

`Binding` 是不可变部署声明和治理域身份，不包含 semaphore、queue、Task 或 dispatcher；这些活状态全部由 Runtime 按 Binding 身份持有。只有创建另一个 Binding 才形成隔离的 Execution 调度命名空间。

`CapacityPolicy.capacity_key` 把不同 Binding 的 Model/Tool 门禁映射到同一个稳定物理资源容量所有者；相同 key 的策略必须一致，否则 bind 明确拒绝。`runtime_instance` 的 key 只在当前 Runtime 内共享；`deployment` 的 key 通过 coordinator 跨 Runtime 共享；`external_resource` 表示 permit 由 Provider、数据库或其他外部所有者管理，LocalRuntime 不创建本地 limiter。Tool 调用还会把 `ToolSpec.resource_key` 接入共享物理工具门禁，且仍同时受 Binding 总门禁和 Layer 单 Execution fan-out 限制。

本节不适用于 `module.invoke()`/`module.stream()` 的 direct execution。direct execution 没有隐式默认 Binding，也不能通过“字段全部采用默认值”被描述成托管 Runtime；它使用的是更小的内部 direct scope，治理能力为零且必须如实报告。

`CapacityScope.DEPLOYMENT` 表示所有共享该 `binding_id` 的 Worker 共同遵守一个总量；Runtime 必须拥有集中 admission、分布式 permit 或等价协调能力。`LocalRuntime` 在没有 `capacity_coordinator` 时会拒绝这种 Binding，绝不退化为实例内计数。`CapacityCoordinator` 是可注入协议；`InMemoryCapacityCoordinator` 只用于同一进程，`SQLiteCapacityCoordinator` 用独立的 `pygent_capacity_*` 表、FIFO ticket、带 fencing token 的可续期 lease，在共享同一 SQLite 文件的多个 Worker 进程间协调并在 crash 后回收 permit。多主机部署必须注入能兑现相同协议和租约语义的外部协调器，不能把 SQLite 文件放到不保证锁与持久性的网络文件系统上冒充全局 owner。若希望副本数增加时总容量随之增加，应显式使用 `CapacityScope.RUNTIME_INSTANCE`。不支持请求作用域的 Runtime 必须在 bind/compile 阶段拒绝，不能静默把 deployment 上限解释成每 Worker 上限。

### Binding 与动态模型部署权限

Runtime 为每个实际 Binding 治理域签发稳定的内部 `deployment_scope_id`。该身份不是 Binding 显示名称、tenant 字符串或 Python 对象地址。动态模型配置必须通过 `runtime.create_binding()` 返回的 RuntimeBinding 取得受控句柄；单独的 raw Binding 策略值不能作为配置权限。

```python
assistant = binding.model_groups.get(assistant_group)

snapshot = await assistant.ensure_profile(
    profile="balanced",
    routes=balanced_routes,
    fallback=balanced_fallback,
    invoker=balanced_invoker,
    resource_ref=balanced_resource_ref,
    make_default=True,
    deadline=configuration_deadline,
)
```

`ModelGroupHandle` 属于控制面，不能放入 Module/Agent。`ensure_profile()` 对规范化配置计算稳定 digest；相同 scope、profile 与 digest 的并发调用 single-flight 并返回同一不可变 snapshot。`make_default=True` 把 profile current pointer 与 default pointer 在一个事务中发布。控制面验证、资源解析、store 打开与发布全都受本次显式 `deadline` 和 cancellation 控制；失败不暴露半发布状态。目录查询不会自动发布配置。

Runtime 只通过 `runtime.create_binding()` 创建治理域和签发配置权限。原始结构化 Child 继承 Parent scope，预绑定 Child 使用自己的 scope。固定模型的 `register_model_invoker()` 路径保持独立，不因同名 profile 配置而改变。

延迟模型组在 `start()` admission 时一次解析当前有效 Binding 边界内的全部声明需求与 profile 选择。缺少默认或显式选择的 profile、选择不被 Layer policy 允许、或者生成参数越界时，都在该边界的 `forward()`、模型事件和 effect 开始前失败。独立 Binding Child 在每次 Child execution 开始时建立独立 admission；它不会覆盖 Parent 或另一次 Child admission 已经 pin 的快照。详细契约见 [延迟与动态模型组规范](../llm/DYNAMIC_MODEL_GROUP_SPEC.md)。

## Child 的三种放置方式

以下示例冻结 0.2.x 的三种方式；它们都不改变 `forward()`：

```python
class MainAgent(Agent[UserMessage, AIMessage]):
    def __init__(self, reviewer: ModuleDependency):
        super().__init__()
        self.reviewer = reviewer

    async def forward(self, message: UserMessage, context: Context):
        return await self.reviewer(message, context)
```

### 1. Inherit：原始 Child 继承 Parent Runtime/Binding

```python
reviewer = ReviewerAgent()
main = MainAgent(reviewer=reviewer).bind(
    main_runtime,
    binding=binding_main,
)
```

`reviewer` 在 `main_runtime` 中作为受管 Child 执行，并继承 `binding_main`。

### 2. Pinned：预绑定 Child 固定到另一 Runtime/Binding

本例中 reviewer 因独立容量、权限与服务生命周期边界而使用 `binding_reviewer`。如果只需改变物理 Worker 而不形成独立治理边界，应使用当前 Binding 的 placement policy，不应新建 Binding。

```python
reviewer = ReviewerAgent().bind(
    reviewer_runtime,
    binding=binding_reviewer,
)

main = MainAgent(reviewer=reviewer).bind(
    main_runtime,
    binding=binding_main,
)
```

`MainAgent.forward()` 仍然执行 `await self.reviewer(...)`。在已有 ExecutionScope 内调用预绑定 BoundModule 创建的是 Main Execution 的 Child，而不是 reviewer Runtime 上的新 Root；Child 使用 `binding_reviewer` 的 admission、queue、live/runnable 和资源容量，同时继承 Main Execution 的 root/parent lineage、有效 deadline 与取消 scope。`binding_main` 仍计算该调用的 child depth、fan-out、waiter 和结构化 live ownership，不能通过跨 Runtime 调用绕过 Parent 执行树预算。

预绑定 BoundModule 在执行树外创建 Root 时必须使用：

```python
message, context = await reviewer.invoke(message, context, execution=execution)
```

不要在 `forward()` 中调用 `self.reviewer.invoke(...)`。

### 3. Adaptive/Remote：稳定 Binding 引用选择健康目标

```python
registry.publish(
    "reviewer-service",
    (
        WorkerTarget("reviewer-a", "https://reviewer-a.internal", ("model:reasoning",)),
        WorkerTarget("reviewer-b", "https://reviewer-b.internal", ("model:reasoning",)),
    ),
)
client = HTTPWorkerClient(registry)
verified_plan = reviewer_bound.plan  # 由受信部署控制面发布
placement = PlacementPolicy.adaptive()
main_runtime.register_remote_target(
    "reviewer-service",
    HTTPRemoteModuleTarget(
        client,
        binding_ref="reviewer-service",
        plan_id=verified_plan.plan_id,
        graph_hash=verified_plan.graph_hash,
        required_capabilities=("model:reasoning",),
        placement=placement,
    ),
)

reviewer = RemoteModule[UserMessage, AIMessage](
    binding_ref="reviewer-service",
    plan_id=verified_plan.plan_id,
    graph_hash=verified_plan.graph_hash,
    required_capabilities=("model:reasoning",),
    placement=placement,
)
main = MainAgent(reviewer=reviewer).bind(
    main_runtime,
    binding=binding_main,
)
```

每个 HTTP Worker 进程必须在创建自己的 `LocalRuntime` 时注入指向同一容量数据库的独立 coordinator；共享的是数据库中的 owner/lease，而不是 Python 对象：

```python
capacity = SQLiteCapacityCoordinator("/var/lib/pygent/deployment-capacity.sqlite3")
worker_runtime = LocalRuntime(
    history=worker_history,
    capacity_coordinator=capacity,
    code_artifact=CodeArtifactSpec(
        package="support-agent",
        version="1.4.0",
        digest="sha256:0123456789abcdef",
        entrypoint="support.agents:build_reviewer_worker",
    ),
    input_schema="schema://pygent/message-context-input@0.2",
    output_schema="schema://pygent/message-context-output@0.2",
    serializer="pygent-json-v1",
)
worker_bound = worker_runtime.create_binding(
    name="reviewer-service",
    execution_capacity=deployment_run_capacity,
    model_capacity=deployment_model_capacity,
    tool_capacity=deployment_tool_capacity,
).bind(reviewer_agent)
worker_app = HTTPWorkerApp(
    bound_module_worker_handler(
        {"reviewer-service": worker_bound},
        artifact_resolver=deployment_artifact_resolver,
    ),
    capabilities=("model:reasoning",),
)
```

HTTP Worker 只接纳完整可移植计划：`CodeArtifactSpec` 的 package/version/digest/entrypoint、输入 schema、输出 schema 和 serializer 都会进入 `graph_hash`。`artifact_resolver` 必须返回 `WorkerDeploymentManifest`，把已验证 digest、实际加载 callable 的 canonical entrypoint 与实际 wire schema/serializer 绑定到该计划；缺 resolver、digest/entrypoint/schema 不匹配或缺少任一字段都 fail closed。不能以 pickle、Python 类型名、未验证的本地 import 或 Worker 默认值补齐。

进程优雅退出时应 `await capacity.close()` 主动释放自身 lease；进程崩溃时 heartbeat 停止，其他 coordinator 只能在 lease TTL 到期并获得更高 fencing token 后取得 coordinator permit。`max_waiters` 与 live/runnable/Model/Tool permit 一样由 deployment coordinator 全局计数，不能由每个 Worker 各维护一份整数。若 Worker 无法访问同一可靠 owner，`CapacityScope.DEPLOYMENT` 继续在 Binding 创建阶段拒绝，不能降级为进程内上限。

`CapacityPermit` 会通过 `async with current_infrastructure().tool_permit(...) as permit` 和 `current_capacity_permit()` 暴露给受保护操作；公开 `owner_key` 使用稳定的 `<kind>:<capacity_key>` 形式，例如 `tool:application.database`，不暴露 coordinator 的表名或内部命名空间。`fencing_token` 本身只防止旧 owner 再次取得 Pygent permit；只有数据库、对象存储或外部服务在最终提交时原子校验并拒绝过期 token，才能形成强副作用 fencing。外部资源不校验 token 时，已经发出的 I/O 或提交仍可能在 lease 失效后完成，应用不得据此宣称 exactly-once 或强 fencing。

`RemoteModule` 表示稳定逻辑部署引用，不保存 URL、连接池或 Worker 地址，但必须携带由受信部署控制面验证过的 `plan_id`、`graph_hash`、capability 与 placement。ExecutionPlan 在 bind 阶段保持稳定；`WorkerRegistry` 先按声明信息筛选，但客户端在每次启动前仍调用物理 endpoint 的 `/health` 并以实时 capability 复核，随后把 required capabilities 放入启动请求，由 Worker 再次校验自己的实时能力，陈旧 Registry 不能替 Worker 背书。注册的 `HTTPRemoteModuleTarget.placement` 必须与编译后的 `RemoteModule.placement` 完全一致；adaptive 不能覆盖 pinned，pinned 的 target 也不能被替换。远程 Child 等待与本地 Child 一样执行 Parent lease handoff，并以稳定 child/root/parent identity 发出同一事件树。启动请求尚未被接纳时可以尝试其他健康 target；请求已接纳后的 poll 故障默认进入 outcome unknown 并 fail closed，只有稳定 idempotency key、共享 durable Worker Job capability和原子 owner claim 同时成立时才允许故障接管，不能把普通 placement 偷偷升级成副作用重放。

远程延迟 ModelGroup 还必须声明 `model.deferred.exact-pin.v1`。Worker health 会返回模型 deployment store 的稳定 namespace；客户端把首次接纳的 namespace 固定到 `RemoteExecutionHandle`，故障转移只尝试相同 namespace 的 Worker。多个 Worker 应注入共享的 `ModelDeploymentStore`；SQLite 参考实现可显式传入 `namespace_id=`，跨主机的自定义 Store 也必须提供稳定 namespace。Worker 会校验稳定 admission ref、scope、snapshot/resource digest、resolver 与 capacity domain，不会改读本机 current。固定模型请求不携带这些扩展字段。

`binding_ref` 是进入 ExecutionPlan 的逻辑依赖，不是一次调用时随意查询 Agent 的字符串。部署控制面可以维护该引用对应的动态实例集合；以下是控制面伪代码，不是已经冻结的 Pygent SDK：

```python
deployment_registry.publish(
    binding_ref="reviewer-service",
    targets=["worker-a", "worker-b", "worker-c"],
)
```

扩容、缩容、健康摘除和地域路由只改变 `targets`，不改变 `reviewer-service` 的 schema、授权、容量归属、durability 要求或 Child identity。业务代码不提供 `registry.resolve(user_supplied_name)` 形式的开放调用入口。要引入一个此前未声明的逻辑 Agent，应重新 bind/compile 得到新的 ExecutionPlan，或从执行树外创建独立 Root Execution/Job。

## 可移植 ExecutionPlan

具体 Runtime 在绑定阶段编译 Module 图。以下示例展示远程 Worker 可验证和加载的计划形态；应用通常不手写它：

编译器不能通过“看不懂就忽略”生成稳定 identity。direct freeze 与 compiler 共用 Core validator：除已声明依赖、callable 和 Module 在 `trusted_live_resource_attributes` 中逐项声明的 opaque 部署资源外，public/private stored state 都必须是严格不可变可移植值，或者由 `execution_plan_config()` 显式投影为严格 JSON snapshot。mutable dict/list、未知对象和不安全 frozen dataclass 字段在 direct 或 bind 前失败；裸 mutable container 也不能伪装成 live resource。hook snapshot、声明 capability、恢复/effect safety 和 Binding 策略都参与最终 identity。Runtime 在每个 Execution 启动前保留 drift 检查，防止绕过 Module 冻结后以旧 `graph_hash` 执行新语义。opaque adapter 是部署信任边界，其内部生命周期状态不会进入 identity，也不得承载请求或业务状态。

```python
from pygent.runtime.plan import (
    CodeArtifactSpec,
    ExecutionPlan,
    ModuleSpec,
)

plan = ExecutionPlan(
    root="agent",
    runtime_api_version="0.2",
    artifact=CodeArtifactSpec(
        package="support-agent",
        version="1.4.0",
        digest="sha256:0123456789abcdef",
        entrypoint="support.agents:build_agent",
    ),
    modules=(
        ModuleSpec(
            path="agent",
            type_name="SupportAgent",
            definition_id="support.SupportAgent",
            children=("agent.model",),
            input_schema="schema://message/user@1",
            output_schema="schema://message/ai@1",
            serializer="json",
        ),
        ModuleSpec(
            path="agent.model",
            type_name="ModelCallLayer",
            definition_id="pygent.llm.ModelCallLayer",
            resource_keys=("model:support",),
            input_schema="schema://message/user@1",
            output_schema="schema://message/ai@1",
            serializer="json",
            required_capabilities=("llm.openai-compatible",),
            placement_constraints=("pool=networked",),
            retry_policy_ref="policy://model/default@1",
            checkpoint_policy_ref="policy://checkpoint/module-boundary@1",
        ),
    ),
)

assert plan.is_portable
wire_value = plan.to_dict()
restored = ExecutionPlan.from_dict(wire_value)
assert restored.graph_hash == plan.graph_hash
```

`type_name` 只是诊断与可读 metadata，不参与 Module 解析、远程加载或权威身份判断。远程 Worker 必须依据已验证的 `definition_id`、代码制品、entrypoint、schema 与策略引用重建 Module，不得仅凭 Python 类型名或 pickle。

`graph_hash` 覆盖会影响执行的图、代码制品和契约引用，但不覆盖描述性 metadata。`from_dict()` 拒绝未知 schema version、非法图和哈希不匹配的内容。Runtime 必须在调度远程执行前另外验证代码制品签名、Worker capability 和策略引用。

## Root 托管执行

历史 `manager.submit(func, ...)` 对应当前 BoundModule 的 Root 入口：

```python
message, context = await bound.invoke(
    UserMessage(content="请分析这份材料"),
    context,
    execution=ExecutionOptions(
        request_id="http-attempt-002",
        idempotency_key="analyze-document-017",
        deadline=finite_deadline,
    ),
)
```

如果 Agent 声明的 `ModelCallPolicy` 允许，本次 Root 可以选择 profile 和少量运行可变生成参数：

```python
message, context = await bound.invoke(
    message,
    context,
    execution=ExecutionOptions(
        deadline=finite_deadline,
        model_calls={
            "assistant": ModelCallOptions(
                profile="quality",
                temperature=0.1,
                max_output_tokens=4096,
            ),
        },
    ),
)
```

`model_calls` 的 key 必须对应 ExecutionPlan 中声明的延迟模型组。它不能引入新的模型组，也不能覆盖 route、credential、client、retry 或 fallback。Runtime 在 admission 时把 profile 名称解析为精确快照，并把选择与实际生成参数纳入 effect identity；执行和 retry 期间不再查询当前默认值。应用若要会话粘性，应自行在会话状态中保存 profile 名称，并在各次 Root 调用中重复传入，而不是修改 Agent 或创建会话私有 ModelGroup。

流式观察不创建另一套调度语义：

```python
async with bound.stream(message, context, execution=options) as stream:
    async for event in stream:
        print(event.kind, event.data)

    message, context = await stream.final_result()
```

`invoke()` 和 `stream()` 都创建 Root Execution，并竞争同一个 Binding 的 execution lease。

普通托管调用可以完全省略 `execution`；Runtime 生成 `request_id` 并使用 Binding 默认 deadline、identity 与瞬时执行策略。显式 `ExecutionOptions` 只覆盖本次调用，不是使用 Runtime 的入场券。

`request_id` 标识一次传输或 SDK 尝试；`idempotency_key` 标识逻辑业务操作。客户端重试同一操作时应生成新的 `request_id`，但复用原 `idempotency_key`。普通非 durable 调用可以省略 idempotency key；要求去重、恢复或最终提交协调时必须提供。

`stream()` 是拥有型便捷入口：它创建 Execution、订阅事件，并在调用方提前退出时取消该 Execution。需要后台继续、独立观察、断线重连或取消控制时直接使用 Execution Handle：

```python
handle = await bound.start(message, context, execution=options)

snapshot = await handle.snapshot()
assert snapshot.phase in {ExecutionPhase.SUBMITTING, ExecutionPhase.PREPARING}

async with handle.subscribe(after=cursor) as events:
    async for event in events:
        print(event.kind, event.data)

# 关闭 events 只取消订阅，不取消执行。
message, context = await handle.result()

# 只有显式控制调用才取消 Execution。
await handle.cancel()
```

`start()` 在同步验证输入并创建逻辑 Execution 与 owner Task 后立即返回，不等待 history 打开、profile pin、容量 admission 或 `forward()`。这些阶段通过 `ExecutionSnapshot.phase` 和 `execution.submitted`/`execution.admitted` 事件可观察，并全部受 `ExecutionOptions.deadline` 与 `cancel()` 控制。任何准入失败都由 `handle.result()` 抛出结构化错误，同时 snapshot/outcome 保持可查询。`invoke()` 与 `stream()` 只是该控制面的便捷组合，不改变 `(message, context)` 最终结果契约。

已有 Execution 只能附着，不得隐式恢复：

```python
handle = await runtime.get_execution_handle(execution_id)
snapshot = await handle.snapshot()

# 只有恢复控制器才调用；它会验证资格、获取 owner lease 并创建新 attempt。
recovered = await runtime.recover(execution_id, compatible_bound_module)
```

不要在 `forward()` 中通过另一个 BoundModule 的 `invoke()` 调用子 Agent。那会创建新的 Root 并切断当前父子树；原始、预绑定和远程 Child 都应直接 `await self.child(...)`，由当前 ExecutionScope 建立结构化关系并应用对应 placement。

## Blocking Child：历史 `call()` 示例

历史规范使用 `manager.call(child, ...)` 表达同步 Child。当前 Module 调用已经可以直接表达相同业务关系：

```python
class ReviewPipeline(Module[UserMessage, AIMessage]):
    def __init__(self, writer: Module, reviewer: Module):
        super().__init__()
        self.writer = writer
        self.reviewer = reviewer

    async def forward(self, message: UserMessage, context: Context):
        draft, context = await self.writer(message, context)
        answer, context = await self.reviewer(draft, context)
        return answer, context
```

Runtime 语义：

```text
parent RUNNING
  -> await child
  -> parent WAITING_CHILD，释放 lease
  -> child 获得 lease并执行
  -> child terminal
  -> parent WAITING_RESUME
  -> 原 parent coroutine 重新获得 lease并继续
```

这段示例不依赖 `Agent` 类型；writer 和 reviewer 可以是任意 Module。该 RESUME 要求 parent 的 owner Task 和 Python 调用栈仍然存在，不表示 Worker 故障后可以从 `await child` 之后恢复。

Model/Tool 跨容量平面等待必须遵守更严格的顺序，防止 Execution lease 与资源 permit 形成 hold-and-wait：

```text
release Execution lease
  -> wait/acquire Model or Tool permit
  -> execute adapter I/O
  -> release Model or Tool permit
  -> enqueue Execution RESUME
  -> reacquire Execution lease
```

Tool 容量同时包含两层：`ToolCallLayer.max_concurrency` 限制单 Execution fan-out，Binding `tool_capacity` 限制该 Binding 的跨 Execution 总并发，并可继续按稳定 `resource_key` 映射到物理资源容量。各层独立计数且同时生效，不能互相替代。

Model/Tool Layer 不通过私有 Runtime hook 获得上述能力。Runtime 向 `pygent.core.Infrastructure` 公共 SPI 提供 effective deadline、Model/Tool permit、部署资源 resolver、稳定 Tool 幂等 key 与 managed effect；用户开发的 Infrastructure Module 使用完全相同的入口。普通业务 Module 不接触低层 ExecutionScope。direct scope 的 permit 是 passthrough、effect 不持久重放、部署 resolver/Execution 幂等身份不可用；managed scope 才执行 lease handoff、共享容量和 durable replay。Runtime 只治理并解析注入资源，不实现 Provider、executor 或业务 adapter 逻辑。

## Parallel Child：结构化 `gather()`

需要并行 Child 时，Module 使用继承的结构化 `gather()`；用户仍不接触 ExecutionScope：

```python
class ParallelPipeline(Module[UserMessage, AIMessage]):
    def __init__(self, researcher: Module, reviewer: Module):
        super().__init__()
        self.researcher = researcher
        self.reviewer = reviewer

    async def forward(self, message: UserMessage, context: Context):
        research_result, review_result = await self.gather(
            self.researcher(message, context),
            self.reviewer(message, context),
        )
        return merge(research_result, review_result)
```

其语义是：

- 每个 Child 独立 admission 并等待 runnable lease；
- Parent 进入 `gather()` 时释放 lease；
- `gather()` 只执行一次 Parent yield 和一次调度 RESUME，结果保持输入顺序；
- Parent scope 退出前必须 cancel/join 未完成 Child；
- 结构化 Child 不提供 `detached=True`；
- 普通工具或 Agent-backed Tool 的 detach 通过独立 ToolTask admission 完成，需要 durable recovery 时由独立 Job 承载该 ToolTask；它不是 Child 选项，Parent 只获得稳定引用，新任务仍受 Binding 与资源治理。

Tool detach 的稳定语义如下：生命周期由应用或自定义授权 Module 选择，模型不能自行提升；Runtime 独立 admission 后，ToolCallLayer 立即返回带不可变 ToolTask 快照的 detached ToolResult。Parent 只保留该快照，并使用稳定 `task_id` 查询、取消或取得终态：

```python
task = detached_result.task
snapshot = await runtime.get_tool_task(task.task_id)
await runtime.cancel_tool_task(task.task_id)
final_result = await runtime.get_tool_result(task.task_id)
```

普通工具与 Agent-backed Tool 共享同一公开契约。Agent-backed Tool 在 detach admission 成功后不再是结构化 Child。需要故障后重新获得时，Runtime 必须由独立 Job 承载 ToolTask；不满足必需 durable task capability 时必须拒绝 detach。

参考 Runtime 的 durable Tool Job 公开面为：

```python
job: JobSnapshot | None = await runtime.get_job(job_id)
recovered: tuple[JobSnapshot, ...] = await runtime.recover_tool_jobs(bound_module)
```

`JobSnapshot` 包含稳定 `job_id`、所承载的 `task_id`、稳定 `logical_key`、`JobState`、`binding_id`、`plan_id`、`resource_key`、required capabilities 与 attempt。logical key 由 run/Root/Module path、确定性 Module occurrence、call/idempotency identity 派生；SQLite admission 按该 key 原子 get-or-create，同时提交 Job 身份和所承载 ToolTask 的稳定身份/请求。Parent 在 Job commit 与 Execution commit 的夹缝崩溃时，恢复重建相同 occurrence 并只能取回原 Job；不同 occurrence 即使复用 `call_id` 也必须产生独立 Job。记录中不得出现 callback、handler、registry 或连接。恢复不是 `DurableToolTaskManager.recover()` 后直接执行 registry：调用方必须重建并提供兼容 BoundModule，Runtime 校验所有治理身份后注入受管 execution path，新 attempt 才能经过原 Binding 的 Tool gates 执行。

## `max_concurrency=1` 的父子阻塞

历史规范专门要求单 lease 下不能出现父等子死锁：

```python
binding = Binding(
    name="serial",
    execution_capacity=ExecutionCapacityPolicy(
        scope=CapacityScope.RUNTIME_INSTANCE,
        max_live_executions=32,
        max_runnable_executions=1,
        max_queue_size=16,
        max_waiters=32,
        max_child_depth=8,
        max_children_per_execution=16,
    ),
)

bound = ReviewPipeline(writer, reviewer).bind(runtime, binding=binding)
message, context = await bound.invoke(message, context, execution=execution)
```

虽然并发上限为 1，Parent 在等待 writer 或 reviewer 时会释放 lease，因此 Child 可以执行；Child 完成后原 Parent coroutine 通过 RESUME 队列重新获得 lease。

## Parallel Child 等队列容量

Parent 不能占着最后一个 lease 等待 Child 入队。`gather()` 中任一 Child 在有界 admission 失败时，Runtime 取消并 join 同组未完成 Child，再向 Parent 传播 `ExecutionAdmissionError`；业务可以捕获后调用 fallback：

```python
try:
    results = await self.gather(self.worker(message, context))
except ExecutionAdmissionError:
    # max_queue_size=0，或有界 capacity waiter 已满时 fail fast
    return await self.fallback(message, context)
```

如果策略允许等待且 START queue 暂时已满，Runtime 会先释放 Parent lease，再等待 admission；错误必须发生在对应 Child `forward()` 或业务 Task 启动之前。

## `wait_external()`：短时外部反馈

用户可以在自定义 Module 中声明一个明确的外部等待点：

```python
class ApprovalModule(Module[ActionMessage, ApprovalMessage]):
    async def forward(self, message: ActionMessage, context: Context):
        value = await self.wait_external(
            kind="approval",
            key=message.approval_id,
            request={"action": message.action},
            timeout=60.0,
        )

        decision = ApprovalMessage(
            approval_id=message.approval_id,
            approved=value["approved"],
            comment=value.get("comment", ""),
        )
        return decision, context + decision
```

服务收到 HTTP、WebSocket 或消息系统中的反馈后，通过 Runtime 或共享信号适配器完成对应 waiter：

```python
result = await runtime.deliver_external(
    kind="approval",
    key=request.approval_id,
    value={
        "approved": request.approved,
        "comment": request.comment,
    },
)
```

`wait_external()` 会暂停当前 `forward()` 以及同步等待它的 Parent 调用链。Runtime 应让该执行流进入 `WAITING_EXTERNAL` 并释放 runnable lease，因此不会阻塞线程、event loop 或其他独立 Execution；但它仍保留 live execution、owner Task、调用栈、局部变量和 Context 引用。Module 定义本身不会被全局锁住，前提是 Module 不保存请求级可变状态。effective deadline 取 Execution deadline、局部 `timeout` 与 `ExecutionCapacityPolicy.max_external_wait_seconds` 三者中的最早值；即使调用方给出很远的 deadline，挂起 Task 也不能越过部署策略硬上限。deadline、取消或关闭后 waiter 必须原子注销，允许相同 `(kind, key)` 在后续 Execution 中安全复用。

调用方必须把它当作有成本的短等待：每个等待都要有 deadline，Runtime 必须限制 waiter 和 live execution 数；进程退出会丢失 coroutine continuation。无法合理限定为秒级或分钟级的审批应立即返回 `ApprovalRequiredMessage` 并结束当前 Execution，用户反馈到达后再创建新 Execution。

## Deadline 与取消

Root deadline 从 `start()` 被调用时开始，覆盖提交、history/store 初始化、Binding 与计划准备、模型 profile admission、资源与容量排队、`forward()`、清理和 finalization。Runtime 内部每一个 await 都必须使用剩余预算并同时响应 Execution cancellation 与 Runtime shutdown。终态提交可以使用有硬上限的 cleanup grace 保证记录完整，但不得继续业务执行。

配置与 profile 发布不是 Execution 的隐藏前置步骤；它们使用 `ensure_profile(..., deadline=...)` 的独立控制面预算。业务请求只读取并 pin 已发布快照。

Child deadline 只能收紧：

```python
effective_child_deadline = min_non_none(
    parent_deadline,
    requested_child_deadline,
)
```

Child 不能放宽 Parent deadline。Parent deadline 到达时，正在排队、运行或等待调度恢复的结构化 Child 都必须进入取消和清理流程。

## RESUME、RETRY 与 CHECKPOINT RESTORE

三者不得混用：

- `RESUME`：原 owner Task 仍存活，只是在受管等待期间释放并重新获得 execution lease；
- `RETRY`：旧 attempt 已失败或失去执行连续性，从声明的 Root 或 Module 边界创建新 attempt 并重新执行；
- `CHECKPOINT RESTORE`：从持久化状态重建执行边界，恢复单位、可用状态和后续重放范围由 checkpoint policy 决定。

普通 `forward()` 的局部变量、任意第三方 `await` 和 Python coroutine continuation 不属于默认 checkpoint。Durable Runtime 不得在没有显式状态与重放契约时承诺从任意源码位置继续。详细要求见 [持久化与恢复边界](DURABILITY.md)。

0.2.x 参考 Runtime 在受管调用边界自动记录 durable history，恢复时重新执行 `forward()` 并重放已提交结果；用户不调用 `step()` 或 checkpoint API。自动记录点、确定性限制和动态策略见 [透明恢复与确定性重放](REPLAY.md)。

## Durable eligibility

需要 durable 保证的 Module 通过通用 `ExecutionRequirements` 声明 capability；Runtime 在 bind/compile 阶段校验，不满足时拒绝绑定：

```python
class DurableAgent(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        requires_finite_deadline=True,
        required_capabilities=("durability.sqlite",),
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    async def forward(self, message, context):
        ...

history = await SQLiteHistoryStore("runs.sqlite3").open()
runtime = LocalRuntime(history=history)
bound = runtime.create_binding(
    ...,
    durability=DurabilityPolicy(mode=DurabilityMode.REQUIRED),
).bind(DurableAgent())
report = bound.durability
```

`DurabilityPolicy` 支持 `DISABLED`、`PREFERRED` 和 `REQUIRED`。`LocalRuntime(history=...)` 声明 `durability.sqlite`，但存储能力不等于 Module 恢复资格。`RecoverySafety` 与 `EffectSafety` 都是严格、不可变、可哈希的声明；普通 Module 默认两者均为 `UNDECLARED`。required 会逐个验证 ExecutionPlan 节点，缺 capability、未声明 boundary retry、存在未验证 effect 时都在 bind 阶段拒绝。preferred 可以降级，但必须由 `bound.durability` 报告 effective/missing capability、`recovery_undeclared_modules`、`effect_unverified_modules`、recovery/checkpoint/replay、事件重连、容量 scope 与降级原因。只有 SQLite 可用且整张图合格时才报告 `module_boundary_retry`；否则即使事件历史可重连，也只报告 `run_history_only`，且 `recover()` 拒绝。

报告始终明确 `arbitrary_coroutine_recovery=False`、`exactly_once_external_side_effects=False`。额外 Runtime/Worker capability 也使用稳定字符串进入 ExecutionPlan。`EFFECT_FREE` 表示本节点本身没有非确定性或外部 effect；`MANAGED_EFFECTS` 表示这些操作全部通过 Runtime 受管 effect 边界，不能把任意第三方 callback 包装成该声明。恢复仍只覆盖受管边界；未知外部副作用、任意第三方 `await`、snapshot compaction、跨版本 migration 和长期 suspend 继续服从 [持久化与恢复边界](DURABILITY.md)。

## SDK 边界

Runtime 只有一个逻辑 Execution 状态模型、一个 `ExecutionBackend` 控制协议和一个事件终结规则；Local、SQLite history、HTTP Worker 与 SSE 只是后端或 transport 实现。Priority 调度、自定义 snapshot compaction、长期 suspend、跨版本 migration 和开放式 Agent discovery 不属于基础 SDK；扩展实现不得改变现有 `forward()` 与 `(message, context)` 契约。
