# 动态模型组实现说明

## 值对象

动态与固定模型组统一使用 `ModelGroup`。`resolution` 区分 concrete/deferred；`ModelGroup.deferred(name=...)` 是唯一 deferred 构造入口。具体 profile 使用有序 `ModelEntry`，每项携带完整 `ModelSpec`。

`ModelSpec`、`ModelEntry`、`ModelGroup`、能力对象、连接对象和 `ModelConfig` 都是不可变值。模型语义 codec 位于 LLM 域，Runtime 只序列化这个公开投影，不解释 Provider 或 capabilities。

## 控制面

`ModelGroupHandle.ensure_profile()` 完成：

1. 校验 deadline、ownership 和资源来源；
2. 用有序 entries 构造 concrete group；
3. 在 Provider I/O 前验证非空 Provider options；
4. 验证 resolver 和资源 bundle；
5. 计算稳定内容 digest；
6. 通过部署 store 幂等发布；
7. 记录 resident invoker 及 ownership。

同一 scope/group/profile/content 的并发发布继续使用 single-flight。Default、retire、list、current 和 admission API 保持现有生命周期。

## 数据面

Layer 固定配置或 admission snapshot 最终都向 invoker 传递 concrete `ModelGroup`。Invoker 顺序遍历 `models`，每个 entry 内执行 retry；只有实际进入下一个 entry 时才发生 fallback 和能力检查。

Adapter 查找键是 `ModelSpec.protocol`，client 查找键是 `ModelEntry.name`。公开事件、attempt、请求快照和资源映射使用同一个 `model_key`。

Provider stream decoder 每个实际 attempt 独立创建。它可以在完成前产生不公开的 continuation 部件；accumulator 只把完整 continuation 放入最终 `AIMessage`，reset 时清除暂存状态，retry 和 fallback 不复用 decoder。

## 持久化与兼容边界

SQLite profile JSON、admission JSON、effect request、Execution 定义摘要和 Worker 传输都保存完整新模型投影。Decoder 对字段集合做精确校验，因此旧 snapshot 会被拒绝，不做兼容转换或双写。

资源 bundle 的公开数组名是 `model_resources`，元素字段为 `model_key` 和 `resource`。Resource ownership、revision、capacity owner、coordinator domain 和 lease 行为不变。

## 并发

`Infrastructure.model_permit()` 无参数。Direct infrastructure 返回无门禁 permit；managed infrastructure 只获取当前 Binding 的模型 gate。模型组不再额外构造共享 gate。

## Worker capability

Worker 对非空 Provider options 使用 `model.provider-options.v1` 能力标识。Admission 校验发生在 resolver acquire 前；不支持该能力的 Worker fail closed。

## 验证重点

- profile SQLite round-trip 保留完整 ModelSpec；
- 旧字段和 digest 篡改明确失败；
- resident 与 resolver 两种 invoker 来源行为一致；
- fallback 顺序与配置顺序一致；
- durable replay 和 Worker 不丢失 `model_key`、protocol 或 capabilities；
- Binding 模型容量仍覆盖整个模型调用。
