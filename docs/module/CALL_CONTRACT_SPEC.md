# Module 自由调用与标准 RecurrentModule 原则

> 状态：Accepted
>
> 本文定义 0.3 的 Module 调用契约：普通 Module 自由声明业务输入输出，RecurrentModule 表达显式状态递推。各执行模式的支持范围见 Runtime 验收矩阵。

## 原则

1. **计算图中的计算节点统一使用 Module**  
   Agent、Layer 和用户组合继续通过 Module 与 `forward()` 表达计算和层级关系。adapter、codec、Store 与 Runtime 控制面不必成为 Module。

2. **普通 Module 的业务输入输出由用户定义**  
   `forward()` 可以接收零个、一个或多个位置参数与关键字参数，并返回由该 Module 定义的任意结果。Message、Context 和二元组都是可选的公开值或结果形状，不是普通 Module 的强制端口。Context 可以作为普通输入或输出，也可以被 RecurrentModule 选作 state；Context 本身不依赖 RecurrentModule。Root 执行入口现有的框架控制参数不属于 `forward()` 业务契约，控制参数保持由执行入口解释。

3. **RecurrentModule 是可选的标准 Module**  
   框架提供 RecurrentModule，帮助用户表达显式消费 state 并产生 next state 的递推计算。它是 Module 的标准特化，不是 Agent 专用基类，也不是所有 Module 的强制基类。`(input, state) -> (output, next_state)` 是它的语义约定，不预先限制业务值类型、附加参数、返回结构或泛型参数数量。现有 `(Message, Context) -> (Message, Context)` 是一种重要的具体用法。

4. **自由调用不引入隐藏调用状态**  
   开放输入输出不改变共享 Module 实例的状态原则：某次调用的输入、局部结果、recurrent state、请求状态和业务会话状态仍通过参数、局部变量与返回值流转，不写入共享实例。Module 既有的定义配置、子 Module 和显式部署资源规则保持不变。

5. **可移植执行协议显式声明**
   本地调用可以传递普通 Python 值；现有 managed、remote、Worker 和 durable execution 继续使用各自已经支持的契约，不支持的调用形状可以明确拒绝。跨边界调用使用已声明的 codec 与 Worker 信封，不使用隐式 pickle、类型猜测或首次调用学习；执行控制仍由统一 Execution API 提供。
