# Cursor Agent 工具 — OpenAI Schema 说明

本目录包含以 **OpenAI 标准 function calling / tools schema** 描述的 Cursor Agent 工具定义，便于在 pygent 或其它兼容 OpenAI 格式的系统中使用。

## 文件说明

| 文件 | 说明 |
|------|------|
| `tools_openai_schema.json` | 完整 schema：工具名、描述、入参（含参数名与描述）、出参说明（`outputs_description`） |
| `tools_openai_api.json` | 仅工具数组，可直接作为 `tools` 传入 OpenAI Chat Completions API |
| `README_tools_schema.md` | 本说明文档 |

## OpenAI 工具格式

每个工具均符合 OpenAI 的 tool 定义：

```json
{
  "type": "function",
  "function": {
    "name": "工具名称",
    "description": "工具描述",
    "parameters": {
      "type": "object",
      "properties": {
        "参数名": {
          "type": "类型",
          "description": "参数描述",
          "enum": ["可选枚举值"]
        }
      },
      "required": ["必填参数名"]
    }
  }
}
```

- **入参**：由 `parameters` 的 `properties` + `required` 定义。
- **出参**：OpenAI API 不规定工具调用的返回结构；本仓库在 `tools_openai_schema.json` 的 `outputs_description` 中对各工具返回值做了文字说明，供实现参考。

## 使用示例（OpenAI API）

```python
import openai

tools = ...  # 从 tools_openai_api.json 加载

response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "在 src 下搜索登录逻辑"}],
    tools=tools,
    tool_choice="auto"
)
```

从 `tools_openai_schema.json` 使用时，取顶层 `tools` 数组即可；若需严格兼容 OpenAI 请求体，请使用 `tools_openai_api.json` 中的数组。

## 工具列表概览

- `codebase_search` — 语义搜索代码库  
- `run_terminal_cmd` — 执行终端命令  
- `grep` — 文本/正则搜索  
- `read_file` — 读取文件  
- `write` — 写入文件  
- `search_replace` — 字符串替换  
- `edit_notebook` — 编辑 Jupyter 单元格  
- `delete_file` — 删除文件  
- `read_lints` — 读取 linter 诊断  
- `web_search` — 网络搜索  
- `mcp_web_fetch` — 抓取网页为 Markdown  
- `generate_image` — 根据描述生成图片  
- `todo_write` — 任务列表写入/更新  
- `mcp_task` — 启动子代理任务  

详细入参与出参以 `tools_openai_schema.json` 为准。
