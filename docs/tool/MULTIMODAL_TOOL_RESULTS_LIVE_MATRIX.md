# 多模态 ToolResult Live 模型矩阵

> 记录时间：2026-09-17。结果只适用于本次测试的 AZ OpenAI Compatible endpoint 与当时可用路由，不能自动转化为其他 endpoint 的能力声明。

初始矩阵从能力目录与 AZ `openai_chat_completions` 路由的交集中选择 62 个声明图片或视频输入的模型，并额外加入已知的 `glm-5.3` 与 `glm-5.3-flash`，共执行 95 个真实 Provider 场景。图片场景要求模型回答 `BLUE SQUARE`；视频场景要求回答 `RED THEN BLUE`。请求中的媒体都位于保留原 `tool_call_id` 的 `role: tool` 内容数组。

结果为 55 个通过、29 个明确 `invalid_parameter`、5 个请求成功但未理解媒体、6 个因权限、限流或路由不可用而无法判定。`✅` 表示模型正确理解媒体；`❌` 表示本 endpoint 上明确不支持或没有理解；`⚠️` 表示本轮证据不足。

## 标准文件路径工具复测

标准 `FileTools.read_image(file_path)` 与 `FileTools.read_video(file_path)` 接入后，重新执行
能力目录中的全部 91 个模型/模态场景：56 个正确理解媒体，33 个返回 Provider 错误，2 个
请求成功但没有给出预期答案；其中图片通过 46 个、视频通过 10 个。另行指定
`glm-5.3-flash` 的图片与视频场景也都通过。复测确认模型参数是本地 `file_path`，文件字节
由标准工具读取为 inline Base64，随后由 Adapter 放入保留原 `tool_call_id` 的结构化
`role: tool` 内容数组。Provider 路由存在波动，因此下表保留初始完整矩阵作为一次性观测，
能力配置仍应按 endpoint 管理。

| 模型 | 图片 ToolResult | 视频 ToolResult |
|---|---|---|
| `chatgpt-4o-latest` | ❌ invalid_parameter (400) | — 未声明 |
| `doubao-1-5-thinking-vision-pro-250428` | ⚠️ resource_not_found (404) | — 未声明 |
| `doubao-seed-1-6-vision-250815` | ✅ 通过 | — 未声明 |
| `doubao-seedream-5-0-260128` | ❌ invalid_parameter (400) | — 未声明 |
| `doubao-seedream-5-0-pro-260628` | ⚠️ rate_limited (429) | — 未声明 |
| `glm-5.3` | ❌ invalid_parameter (400) | ❌ invalid_parameter (400) |
| `glm-5.3-flash` | ✅ 通过 | ✅ 通过 |
| `gpt-4.1` | ✅ 通过 | — 未声明 |
| `gpt-4.1-2025-04-14` | ✅ 通过 | — 未声明 |
| `gpt-4.1-mini` | ✅ 通过 | — 未声明 |
| `gpt-4o` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-4o-2024-05-13` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-4o-2024-08-06` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-4o-2024-11-20` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-4o-mini` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-4o-mini-2024-07-18` | ❌ invalid_parameter (400) | — 未声明 |
| `gpt-5` | ❌ 未理解媒体 | — 未声明 |
| `gpt-5-chat-latest` | ✅ 通过 | — 未声明 |
| `gpt-5-mini` | ❌ 未理解媒体 | — 未声明 |
| `gpt-5-nano` | ❌ 未理解媒体 | — 未声明 |
| `gpt-5.1` | ✅ 通过 | — 未声明 |
| `gpt-5.2` | ✅ 通过 | — 未声明 |
| `gpt-5.4` | ✅ 通过 | — 未声明 |
| `gpt-5.4-mini` | ⚠️ model_not_found (503) | — 未声明 |
| `gpt-5.4-nano` | ✅ 通过 | — 未声明 |
| `gpt-5.5` | ✅ 通过 | — 未声明 |
| `gpt-5.6-luna` | ⚠️ model_not_found (503) | — 未声明 |
| `gpt-5.6-sol` | ✅ 通过 | — 未声明 |
| `gpt-5.6-terra` | ✅ 通过 | — 未声明 |
| `gpt-6-astra` | ✅ 通过 | — 未声明 |
| `grok-4.5` | ✅ 通过 | — 未声明 |
| `grok-4.6` | ✅ 通过 | — 未声明 |
| `grok-imagine-image-2.0` | ❌ 未理解媒体 | — 未声明 |
| `kimi-k2.6` | ❌ invalid_parameter (400) | ❌ invalid_parameter (400) |
| `kimi-k2.7-code` | ✅ 通过 | ✅ 通过 |
| `kimi-k3` | ✅ 通过 | ✅ 通过 |
| `MiniMax-M3` | ✅ 通过 | ✅ 通过 |
| `o3` | ✅ 通过 | — 未声明 |
| `o4-mini` | ❌ 未理解媒体 | — 未声明 |
| `qvq-max` | ❌ invalid_parameter (400) | ❌ invalid_parameter (400) |
| `qwen-omni-turbo` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen-vl-max` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen2.5-vl-32b-instruct` | ⚠️ permission_denied (403) | ⚠️ permission_denied (403) |
| `qwen3-vl-235b-a22b-instruct` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-235b-a22b-thinking` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-30b-a3b-instruct` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-30b-a3b-thinking` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-32b-instruct` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-32b-thinking` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-8b-instruct` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-8b-thinking` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3-vl-plus` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3.5-122b-a10b` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3.5-27b` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3.5-35b-a3b` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3.5-397b-a17b` | ✅ 通过 | ✅ 通过 |
| `qwen3.5-omni-flash` | ✅ 通过 | ❌ invalid_parameter (400) |
| `qwen3.5-omni-plus` | ✅ 通过 | ✅ 通过 |
| `qwen3.5-plus` | ✅ 通过 | ✅ 通过 |
| `qwen3.6-plus` | ✅ 通过 | ✅ 通过 |
| `qwen3.7-plus` | ✅ 通过 | ✅ 通过 |
| `qwen3.8-27b` | ✅ 通过 | ✅ 通过 |
| `qwen3.8-flash` | ✅ 通过 | ✅ 通过 |
| `qwen3.8-max` | ✅ 通过 | ✅ 通过 |

另外两个 `.env` 配置目标未进入上表：本地 `GLM-5` endpoint 无法连接；独立 `Qwen3.6-Plus` endpoint 返回 403。它们都不能据此声明支持或不支持多模态工具消息。

可重复执行完整目录矩阵：

```powershell
uv run python -m tests.live.multimodal_tool_result_probe --all-az-media-models --concurrency 4
```
