# Pygent Tools OpenAI Schema

This directory contains OpenAI-compatible tool definitions for the built-in
Pygent toolkits.

## Files

| File | Description |
|------|-------------|
| `tools_openai_schema.json` | Full schema with tool definitions and `outputs_description`. |
| `tools_openai_api.json` | The tool array that can be passed directly as an OpenAI `tools` value. |
| `README_tools_schema.md` | This document. |

## Built-In Tools

- `bash` - run a bash command.
- `read` - read an absolute file path.
- `write` - write complete content to an absolute file path.
- `edit` - replace exact text in an absolute file path.
- `edit_notebook` - edit or insert Jupyter notebook cells.
- `read_lints` - read linter diagnostics or the current placeholder result.
- `glob` - find files by glob pattern.
- `grep` - search file contents.
- `web_search` - search the web. The call purpose parameter is `description`.
- `web_fetch` - fetch a public web page and convert it to Markdown.

Tool names are intentionally lowercase. Deprecated compatibility names are not
included in the current schema.
