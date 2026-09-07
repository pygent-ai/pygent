# Module 第一原则

本文从属于 [Pygent 0.3 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **唯一计算抽象**：Agent、Layer 与用户组合都只是 Module。
2. **自由调用契约**：Module 的 `forward()` 可以接收零个、一个或多个位置参数与关键字参数，并返回自身声明的结果；Message、Context 和二元组都不是普通 Module 的强制协议。
3. **标准递推 Module**：RecurrentModule 是框架提供的可选标准 Module，用来表达显式消费 state 并产生 next state 的递推计算；它不固定业务值类型、附加参数、返回结构或泛型参数数量，也不是所有 Module 的强制基类。
4. **单一业务实现**：用户只实现 `forward()`；direct execution 和本地 Child 调用保持其参数与结果。进入 managed、remote 或 durable 边界时验证对应 Runtime 支持的调用形状与 codec，不以执行位置改变已支持调用的业务语义。
5. **声明式组合**：原始 Module、预绑定 BoundModule 或远程 Module 引用声明计算依赖，不可变值声明自身策略；`forward()` 内的直接调用始终表示 Child，并由当前 direct 或 managed ExecutionScope 转发。用户不直接创建或操作 ExecutionScope。
6. **实例无隐藏调用状态**：Module 只保存定义、配置、子 Module 和显式部署资源；调用输入、局部结果、recurrent state、请求和领域会话状态通过参数与返回值流转。
7. **共享不等于调用**：同一 Module 可被多条路径引用和重复调用；定义身份共享，每次调用身份与调用状态独立。
8. **Binding 只属于托管部署域**：direct execution 不创建 Binding；Binding 不与 Module/Agent 一一对应。托管执行中的原始结构化 Child 默认继承当前 Binding，预绑定 Child 或 placement policy 可以声明独立执行域，同一 Binding 可以承载多个 Root 入口。
9. **可移植边界显式验证**：direct execution 可以传递普通 Python 值；managed、remote、Worker 和 durable execution 只接纳已声明并验证的调用契约。不支持的调用在进入该边界前明确拒绝，不通过隐式 pickle、类型猜测或首次调用学习建立协议。支持范围见 Runtime 验收矩阵。
10. **领域控制由组合表达**：handoff、审批和领域终止条件可以由用户 Message 与 Module 表达；这不赋予普通 `forward()` 跨进程持久挂起或调用栈恢复能力。
