# Module 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **唯一计算抽象**：Agent、Layer 与用户组合都只是 Module。
2. **统一状态转移**：Module 只表达 `(message, context) -> (message, context)`；Message 是任意能力的类型化当前增量，不局限于聊天文本；Context 可以是基础 Context 或满足 Context 契约的用户 AgentContext，输入和输出必须保持兼容的具体类型。
3. **单一业务实现、两级执行入口**：用户只实现 `forward()`；未绑定 Module 与 BoundModule 都提供 `invoke()`/`stream()` Root 入口，流式只是同一次执行的事件观察，三种最终结果都统一为 `(message, context)`。
4. **声明式组合**：原始 Module、预绑定 BoundModule 或远程 Module 引用声明计算依赖，不可变值声明自身策略；`forward()` 内的直接调用始终表示 Child，并由当前 direct 或 managed ExecutionScope 转发。用户不直接创建或操作 ExecutionScope。
5. **实例无请求状态**：Module 只保存定义、配置和子 Module；请求、历史和领域状态只通过不可变 Context 显式流转。
6. **共享不等于调用**：同一 Module 可被多条路径引用和重复调用；定义身份共享，每次调用身份独立。
7. **Binding 只属于托管部署域**：direct execution 不创建 Binding；Binding 不与 Module/Agent 一一对应。托管执行中的原始结构化 Child 默认继承当前 Binding，预绑定 Child 或 placement policy 可以声明独立执行域，同一 Binding 可以承载多个 Root 入口。
8. **领域控制由组合表达**：handoff、审批和领域终止条件可以由用户 Message 与 Module 表达；这不赋予普通 `forward()` 跨进程持久挂起或调用栈恢复能力。
