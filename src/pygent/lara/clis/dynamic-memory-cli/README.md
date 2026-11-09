# Dynamic Memory CLI

单文件标准库 CLI,harness 运行方式为 `"<python>" "<script>" --root "<session memory root>"`。一份脚本、四个角色命令面:每个角色只通过各自 directive 或 envelope 拿到自己的命令;本目录不参与 agent 技能目录扫描。勿自行发明 root 或命令:前台使用 `memory-access-instruction` 信封里的 `memory_cli_command`。

## 角色命令面

- **前台 agent**(经 `memory-access-instruction` 信封):`search <keyword-or-key-phrase>` — 对已发布记忆记录的关键词检索。当 Snapshot 不足、或任务依赖既往决策/约束/承诺/偏好/详细历史时使用。
- **Extractor 记忆任务**(经 directive):`get-state` — 读取 `base_memory_revision`、`base_snapshot_revision`、`covered_through`;随后 `publish-pending --file <proposal>` — 校验并提交 proposal JSON(变更 UT 落为 pending)。校验错误会报告字段限额或 stale revision,供同轮重试修正。
- **Builder 记忆任务**(经 directive):`build-pending --file <batch>` — 对冻结批次做确定性构建、Built-only 测试、逐字内容比对、pending→built 原子迁移。
- **Harness 内部**(代码直调,非 agent 面):`init --path <root>`、`freeze-pending --output <batch>`、`test`、`bench`。

## 直接操作命令(自动链路未使用)

- `add --file <file> [--force]` / `update <id> --file <file>` / `retire <id> [--reason]` / `check-conflicts --file <file>` / `list [--full]` / `show <id>` — 手工记录管理与检查。自动发布链路走 `publish-pending` 提案流,这些是单条记录的直接接口。

## 提案流(publish-pending)

proposal JSON 含 `base_memory_revision` / `base_snapshot_revision`(乐观锁基准)、`from_cursor` / `to_cursor`(必须与 `covered_through` 连续)、六桶 `snapshot`(各有条数与字数限额)、`changed_uts`(≤8 条,action 为 upsert/retire,`content` ≤700 字、`queries` ≤4、`must_include` ≤3 且必须是 content 精确子串)、`evidence_refs` 与 `semantic_statement`(必填)。校验失败按错误重写后重发;成功后 UT 落为 pending,经 `freeze-pending` 冻结批次,由 builder `build-pending` 原子迁移为 Built。
