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

- `bash` - run a bash command in a workspace-resolved directory.
- `read` - read a workspace-resolved file path.
- `write` - write complete content to a workspace-resolved file path.
- `edit` - replace exact text in a workspace-resolved file path.
- `edit_notebook` - edit or insert Jupyter notebook cells.
- `read_lints` - read linter diagnostics or the current placeholder result.
- `glob` - find files by glob pattern.
- `grep` - search file contents.
- `web_search` - search the web. The call purpose parameter is `description`.
- `web_fetch` - fetch a public web page and convert it to Markdown.

Tool names are intentionally lowercase. Deprecated compatibility names are not
included in the current schema.

Path parameters accept absolute paths and paths relative to `workspace_root`.
By default, resolved paths must stay inside `workspace_root`; toolkits can be
initialized with `restrict_to_workspace=False` to allow paths outside it.
