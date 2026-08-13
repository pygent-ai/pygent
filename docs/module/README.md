# Module

阅读顺序：

1. [第一原则](FEATURES.md)
2. [SDK 使用](SDK.md)
3. 本文的详细契约

## 接口

```python
class Module(Generic[InputMessageT, OutputMessageT]):
    async def forward(
        self,
        message: InputMessageT,
        context: Context,
    ) -> tuple[OutputMessageT, Context]: ...
```

参数顺序固定为 `(message, context)`，对应 PyTorch/LSTM 的 `(x, h) -> (y, h')`。泛型只约束框架公开的 Message 类型，不允许各层发明另一套公开信封。Message 是 Module 间统一传递的类型化增量，不仅表示聊天文本；检索、计划、评估、审批或领域结果使用 `Message(kind=..., data=...)` 表达。`kind` 是稳定领域类型，`data` 是严格、有限且递归冻结的 JSON 对象，因此 direct、持久化与远程 Worker 使用同一公开值。任意 Python Message 子类可能引入 codec 未知字段或活对象，0.2 明确拒绝这种扩展方式。

## 调用与组合

Module 属性形成稳定定义图。一个 Module 实例可以被多个父级或多条属性路径引用。Module 通常通过原始 Module 实例声明计算依赖；需要独立部署边界时，也可以使用预绑定 BoundModule 或稳定 RemoteModule 引用。Placement 配置只决定一个已声明依赖在哪里执行，不凭空创建未声明的业务子节点。

未绑定 Module 的 `invoke()`/`stream()` 在当前进程建立 direct execution scope，不要求用户创建 Runtime。direct scope 只负责把 `forward()` 内的子 Module 直接调用识别为 Child、转发事件并维持本地取消边界；它不提供 Binding、容量门禁、远程 placement、持久恢复或跨 Root 调度。用户自行管理多个 Root 的并发与外部资源生命周期。

因此 Pygent 不提供让 `forward()` 按任意名称调用未声明 Agent 的开放式 Registry。服务发现可以为 ExecutionPlan 中已经声明的 RemoteModule 或 placement target 动态选择物理实例，但不能改变逻辑依赖图；新增逻辑 Agent 需要新的 Binding/ExecutionPlan，或由执行树外的独立 Root Execution/Job 发起。

`forward()` 中调用子 Module 时，通过当前 execution scope 转发。direct scope 直接执行 Child，不增加框架容量控制；managed scope 自动接入 Runtime 的调度、容量、取消和父子 Execution 管理。普通业务 Module 不直接操作 ExecutionScope 或调度器。需要实现 Provider、Tool、数据库适配层等基础设施 Module 时，可以使用 `pygent.core.Infrastructure` 和 `current_infrastructure()`，获得与内置 Model/Tool Layer 相同的受管基础能力，而不是导入私有 ContextVar。

### ExecutionScope 的具体边界

ExecutionScope 是框架执行协议，不是业务 API，也不要求用户注入到 `forward()`：

1. `module.invoke()`/`module.stream()` 在调用的动态作用域内安装 direct scope，并在成功、失败或取消后可靠恢复原 scope。
2. `bound.invoke()`/`bound.stream()` 安装 managed scope；同一个 `forward()` 不通过条件分支识别执行模式。
3. 活动 scope 内的 `await child(message, context)` 永远表示 Child；scope 外的 `module(message, context)` 不自动创建 Root，而应明确提示使用 `invoke()`/`stream()`。
4. direct scope 调用 Child 的 `forward()`、转发事件并传播原生取消，但不创建 Binding、lease、容量 permit、placement 或 durable checkpoint。

### 公共 Infrastructure SPI

`Infrastructure` 是面向基础设施 Module 作者的窄公共协议，不是普通业务状态容器。它公开 effective deadline、Model/Tool permit、部署资源 resolver、Tool 幂等 key、结构化 gather、detached ToolTask admission 和受管 effect 边界。Runtime 只提供治理、身份和重放，不把 Provider、executor、数据库或业务客户端逻辑吸收到 Runtime 中。

```python
from pygent.core import (
    EffectIdempotency,
    EffectRetryPolicy,
    EffectSideEffect,
    EffectSpec,
    current_infrastructure,
)

async def forward(self, message, context):
    infrastructure = current_infrastructure()
    async with infrastructure.tool_permit("customer-database"):
        result = await infrastructure.execute_effect(
            spec=EffectSpec(
                effect_type="customer.lookup",
                side_effect=EffectSideEffect.READ,
                idempotency=EffectIdempotency.INHERENT,
                retry_policy=EffectRetryPolicy.REPLAY_SAFE,
            ),
            request={"customer_id": message.data["customer_id"]},
            operation=self.lookup_customer,
        )
    return result_message(result), context
```

direct execution 同样提供这一协议，但边界明确降级：permit 是调用方自管的 passthrough；`execute_effect()` 只执行一次当前 operation，不承诺跨调用重放；Runtime 部署资源 resolver、稳定 Execution 幂等身份与 detached admission 不可用。需要这些能力的基础设施 Module 应使用 managed Binding。`active_infrastructure()` 只用于 adapter 判断当前是否位于执行中；独立任务 admission 可用 `independent_execution()` 防止继承 Parent scope。内置 `ModelCallLayer`、`ToolCallLayer` 与 `AgentToolExecutor` 也只使用这些公共入口，不享有私有 `_execution_scope` 特权。

managed effect 必须携带严格 `EffectSpec`，明确副作用类别、幂等策略、重试策略和需要时的稳定 key。Runtime 在 operation 前持久化 `started`；完成后才写入 result。恢复遇到未完成 `started` 时，只有只读、固有幂等或带稳定 key 且声明 `REPLAY_SAFE` 的 effect 可以重放；非幂等 write/external 进入 `unknown` 并 fail closed，不能把“没有 result”解释成“尚未执行”。
5. managed scope 把 Child 交给 Runtime，并建立受管 identity、lineage、deadline、取消、容量与 placement。
6. Python `ContextVar` 向任意 `asyncio.create_task()` 的自动复制不等于框架接纳了 detached 或 managed Child。direct execution 中任意后台 Task 由调用方负责，Root scope 结束后不得继续使用失效 scope；managed execution 必须使用 Runtime 声明的结构化并行 API，任意裸 Task 不获得 Execution、容量或恢复保证。

用户可以在隔离的纯函数单元测试中直接调用 `forward()`，但这会绕过 Child 转发、事件和执行生命周期，不属于生产 Root 执行协议。需要测试完整调用语义时使用 `invoke()`；需要测试托管语义时使用测试 Runtime。

managed Runtime 必须区分三种身份：Module 定义身份表示共享的配置节点；调用身份表示一次具体执行；资源身份表示连接和物理资源容量的所有者。共享定义的每次托管调用都有独立结果、事件、取消和 attempt；指向相同物理资源的调用共享物理资源容量，即使它们来自不同 Module 实例。Binding 拥有的逻辑执行容量是另一层约束，不与资源身份合并。direct execution 只保证每次调用的结果与事件隔离，不建立跨 Root 的资源身份或容量协调。

任意 Root Module 都可以直接执行，也可以按需接入 Binding。Binding 是服务或部署级资源治理域，不是 Module/Agent 身份；托管执行树中的原始子 Module 默认继承 Root 的 Binding。预绑定 BoundModule 保留自己的显式 Runtime/Binding，Parent Binding placement 可以为原始 Child 指定固定或自适应目标。同一 Module 定义可以接入多个隔离 Binding，多个 Root Module 也可以接入同一个 Binding。BoundModule 是“定义 + 部署治理域”的可执行句柄，不拥有会话状态；同一 Module 和 BoundModule 必须能够并发复用。若未绑定 Parent 以 direct 模式调用预绑定 Child，direct scope 会把该 Child 作为独立 managed Root 交给 Child 自己的 Runtime/Binding，等待相同的 `(message, context)` 结果，并在 Parent 取消时等待 Child 清理；direct Parent 本身仍不创建 Binding。

组合构造函数可以接受统一的 `ModuleDependency[InputMessageT, OutputMessageT]` 契约。原始 Module 与 BoundModule 保持不同身份，但都能在 `forward()` 内通过 `await dependency(message, context)` 创建结构化 Child；BoundModule 不应通过继承 Module 来混淆定义身份与部署身份。

首次 `invoke()`、`stream()` 或 `bind()` 会先用同一套 fail-closed validator 检查整张原始 Module 定义图，再原子冻结所有节点。从这一刻起，任何 Module 实例属性重绑定或删除都会立即失败。public/private instance state 以及用户 MRO 中生效的 class policy 都必须是严格不可变可移植值；mutable dict/list、callable class state、未知对象或伪装成 private 的请求容器会在 `forward()` 运行前拒绝。冻结时保存完整定义快照，后续 direct 执行与 managed 启动都会复验，因此 bind 后重绑用户类属性不能绕过 `graph_hash` drift guard。共享 client、连接池、测试同步探针等部署对象必须由 Module 通过 `trusted_live_resource_attributes` 显式列名，且必须是 opaque adapter/resource，不能把裸 list/dict 标成 live resource 绕过检查。

绑定阶段由 Runtime 把已经冻结的定义图编译成带版本和 `graph_hash` 的不可变 ExecutionPlan。BoundModule 使用该绑定快照；原始 Module 对象不是跨进程传输协议。远程部署通过代码制品、稳定 definition ID、schema、serializer 和策略引用重建并验证定义，不依赖 pickle 或仅凭 `type_name` 猜测实现。

ExecutionPlan identity 使用上述同一 validator：标量、tuple、`FrozenJsonObject`、Enum 和字段同样可移植的 frozen dataclass 可以直接参与配置摘要；可变 dict/list、未知对象、非有限浮点或包含不可移植字段的值会让 direct freeze 和编译都失败，不能被静默忽略。已声明 Module 依赖和 `trusted_live_resource_attributes` 中明确列出的部署 live resource 不进入 plan identity；private 名称本身不构成豁免。存入属性的 callable 也不会被静默忽略：它必须是显式 trusted 部署资源，或由 `execution_plan_config()` 的严格 snapshot 声明其稳定行为身份。若 Module 需要从复杂但不可变的内部结构导出稳定声明，应实现该 hook；hook declaration 与所有普通不可变 stored attributes 会同时进入 identity，任一部分变化都会改变 `graph_hash`。

Runtime 在启动 Execution 前仍会重新验证 ExecutionPlan identity，拒绝通过 `object.__setattr__` 等非公开手段绕过冻结造成的定义漂移。需要不同配置或依赖图时，应创建新的 Module 定义并重新绑定，获得新的 plan identity。

## 事件

Module 可以在 `forward()` 内调用继承的 `emit()`。当前 execution scope 将用户数据包装为有序 ExecutionEvent：direct scope 只保证单次调用内的顺序、背压、取消和关闭清理；managed scope 再由 Runtime 提供跨订阅者治理、保留与可恢复能力：

- direct 与 managed `stream()` 都向调用者暴露事件。
- direct 与 managed `invoke()` 都排空或转交事件，不得因无人观察而阻塞。
- 当前 `stream()` 是一次性执行与观察的便捷组合；提前退出该 stream 时，Runtime 取消其创建的 Execution 并等待执行清理。
- 支持 durable recovery 的 Runtime 可以另外提供稳定 Execution Handle 和可重连事件订阅；关闭订阅只释放观察者，不取消 Execution，取消执行必须通过 Execution Handle 显式完成。该高级控制面不改变普通 `invoke()/stream()` 的使用方式。
- direct/managed `invoke()`、Child 调用和 `stream().final_result()` 都返回同一 `(Message, Context)` 业务结果。
- `execution_id`、状态、usage 与订阅信息属于托管 Runtime 的独立 Execution Handle，不进入普通业务结果。

服务层可以把 ExecutionEvent 映射为 SSE 或 WebSocket；Module 不感知传输协议。框架保留自身事件命名空间，用户事件使用项目或领域前缀。

## 状态边界

Module 可以保存不可变配置、子 Module，以及显式 trusted 的 direct-capable 本地 adapter 稳定引用，但不得把 Context、当前 Message、运行结果、单次调用产生的连接/流、锁、队列或其他可变运行状态写入 `self`。validator 拒绝 raw mutable stored state，属性冻结再拒绝 `forward()` 对 `self` 的请求态赋值。框架不会深入检查 opaque trusted adapter 的内部对象图；因此该声明是部署信任边界，adapter 只能维护共享 client、连接池、telemetry 或测试同步等生命周期状态，不能成为请求/会话/业务状态旁路。违反该约束会使无状态、并发隔离和 durability 声明失效。可移植 ExecutionPlan 只记录 adapter 声明或资源引用，不序列化活连接。基础设施 Layer 默认原样返回 Context；拥有业务演进语义的组合层可以返回新 Context。

handoff、审批和其他领域控制流同样由用户 Module 组合：Module 可以返回带稳定 `kind` 和 JSON `data` 的领域 Message，由父 Module 或服务决定下一步。该规则不表示普通 `forward()` 可以在进程退出后保留 Python 调用栈；需要长时间等待时，应结束当前 Execution、外置保存待处理事实，并在反馈到达后通过新 Message 发起新 Execution。

Context 是 Module 显式传递的不可变 Agent 状态快照，可以包含模型历史投影、完整历史视图和其他 portable 领域状态，但不是完整 Execution checkpoint。`forward()` 的局部变量、普通 coroutine 等待状态、未提交事件和外部副作用不包含在 Context 中。普通 Module 可以由 Runtime 从声明的 Root 或 Module 边界重试，但不能仅凭 Message 与 Context 从任意 Python `await` 之后透明恢复；需要持久恢复时必须遵守 [Runtime durability 契约](../runtime/DURABILITY.md)。
