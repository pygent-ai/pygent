# Context 第一原则

本文从属于 [Pygent 0.2 第一原则](../FEATURES.md)。只能澄清，不能与其冲突。

1. **Agent 上下文值**：Context 是框架提供的不可变 Agent 上下文值。它可以作为普通 Module 的输入或输出，也可以被 RecurrentModule 选作 state；Context 本身不依赖 RecurrentModule。基础字段表示当前模型可见投影，并以唯一 `projection_revision` 标识投影版本；用户 Context 子类可以增加完整历史视图、工具状态、文件状态和其他领域数据。
2. **受约束扩展**：用户 Context 子类必须声明稳定 schema 和版本，保持 frozen、slots 与值语义；全部实例字段必须可由严格、有限、递归冻结的 JSON 数据编码。继承不会开放任意 Python 对象旁路。
3. **不可变值**：任何状态或历史演进都产生同一具体 Context 类型的新值，旧值永不改变；`replace()`、消息追加、Child 调用和最终结果不得静默降级为基础 Context。用户可以重载 `+`/`+=` 的值转换，但只能返回新的同类型 portable 值，不得原地修改、执行隐藏 I/O 或改变基础 Message/slot 契约。
4. **数据而非服务**：Context 不持有连接、锁、Store、handler、manager、provider client 或运行资源。
5. **模型投影明确**：模型层只读取基础 Context 的 `system_prompt`、`messages`、`tools` 与 `metadata`；用户字段不会因继承自动暴露给模型。模型与工具 Layer 必须原样传回具体 Context 类型，除非其公开契约明确返回该类型的新值。
6. **写入可见**：当前 Message 不会自动进入模型历史或用户状态，调用者明确决定追加和状态更新时机。
7. **持久化外置**：`projection_revision` 只用于当前模型投影的乐观并发判断，不是业务会话 revision。加载、提交、历史版本、审计和业务冲突处理仍属于框架外部的业务服务。
8. **显式槽位更新**：无槽位 Message 正常追加；带稳定 `slot` 的 Message 替换旧的同槽位值，使基础模型投影保留该槽位当前有效内容。
9. **可见性不是授权**：`Context.tools` 描述本次上下文和模型可见的工具，不构成可信授权证明；实际授权由用户开发的自定义授权 Module 或受信执行适配器决定。
10. **完整历史、模型投影和估算状态分离**：`PygentAgentContext.full_history` 保存 ReAct 已提交的完整消息历史；`Context.messages` 可以因压缩或显式 replacement 改变；token 估算系数和最近 input usage 是不改变 projection revision 的 portable Agent 状态。压缩产生的新 Context 必须保留具体类型、System Prompt、tools、metadata 和完整历史。
