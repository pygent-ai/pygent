# 新增内容形态：扩展施工清单

> 适用范围：模型层的消息内容形态（现有形态：文本、工具调用、图片/音频/视频媒体）。
> 前置：消息树投影的唯一实现在 `pygent/core/_message_fields.py`，其完整性由
> `tests/core/test_message_fields.py` 的反射测试与
> `tests/llm/test_message_projection_golden.py` 的黄金测试强制。
> 扩展必须照本清单逐项施工；漏项会被这两组测试或门禁评审抓住。

## 施工清单（按序）

1. **core 词表**：`pygent/core/_tool_values.py`（或新块类型）——类型定义、
   严格校验、叶子值投影/还原 helpers（`_tool_result_content_*` 同级）。
2. **消息载体**：`pygent/core/values.py`——若需新的消息级字段（参照
   `UserMessage.media` / `AIMessage.media` 的先例），含 `__post_init__` 校验。
3. **统一投影**：`pygent/core/_message_fields.py`——六个维度的选项语义、
   各消费档（full / identity_drop_key / identity_null / omit）的值映射；
   反射测试同步扩展（每个字段逐档声明）。
4. **wire 往返**：`runtime/codec.py` 经统一投影自动获得；新增叶子值形状需
   在黄金语料中加样例并重录基线（见下"基线重录"）。
5. **协议适配器**：各 Provider wire 编解码（参照 gemini `_inline_data_part`、
   openai `_input_image_item` 先例）；transport 词表按需扩展
   （`MediaTransportCapabilities`）。
6. **路由/降级**：`llm/_media_routing.py` 同级——投递缺口规则、不可查看投影。
7. **估算规则**：`llm/_media_tokens.py` 同级——按模态的 token 规则，未知走
   保守预算。
8. **文档与测试**：`docs/llm/` 原则措辞按需更新；黄金语料加样例；协议适配器
   的离线 + live（AZ 网关）验证记录。

## 基线重录

黄金语料样例集变更（第 8 步）后，用
`.tmp/gen_golden.py`（WP0 录制脚本，未入库）重录
`tests/support/message_projection_golden.json`，并确认重录前后消费行为
差异仅为预期中的新形态字段。

## 已知边界

- `Message.content: str` 不可表达正文内交错；交错需求触发时按
  `parts: tuple[str | MediaBlock, ...]` 规范形态迁移（content/media 降为派生视图）。
- wire 字段对旧读取者是显式失败（fail-closed）：每次扩展是持久化消费方的
  协调升级，不是纯加法。
- 事件词表封闭：新输出形态若需流式事件，按 `model.media` 先例增补
  kind 与精确字段集校验，并同步 `docs/llm/FEATURES.md` 原则 10 与
  `docs/EXECUTION.md`。
