from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pygent.agent.pygent_v1.core.prompt import Prompt
from pygent.agent.pygent_v1.tracing import EventStore

COMPRESSION_REQUEST_PROMPT = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

There may be additional summarization instructions provided in the included context. If so, remember to follow these instructions when creating the above summary. Examples of instructions include:
<example>
## Compact Instructions
When summarizing the conversation focus on typescript code changes and also remember the mistakes you made and how you fixed them.
</example>

<example>
# Summary instructions
When you are using compact - please focus on test output and code changes. Include file reads verbatim.
</example>

REMINDER: Do NOT call any tools. Respond with plain text only — an <analysis> block followed by a <summary> block. Tool calls will be rejected and you will fail the task."""

TRANSCRIPT_POINTER = (
    "If you need specific details from before compaction (like exact code snippets, error messages, or content you "
    "generated), read the full transcript at:"
)
TOO_LARGE_FILE_READ_TEMPLATE = (
    "The previous file read result is too large to include in this compacted context, so its details are not shown "
    "here. Path: {path}. Read scope: {scope}. Returned content length: {char_count} characters. Re-read this exact "
    "file or range if you need the content."
)


@dataclass(frozen=True, slots=True)
class FileReadRecord:
    path: str
    mode: Literal["full", "partial"]
    content: str
    range: str | None = None
    created_at: str | None = None


def parse_summary(text: str, *, has_tool_call: bool = False) -> str | None:
    if has_tool_call:
        return None
    start = text.find("<summary>")
    if start < 0:
        return None
    start += len("<summary>")
    end = text.find("</summary>", start)
    if end < 0:
        return None
    summary = text[start:end].strip()
    return summary or None


def collect_recent_file_reads(
    session_dir: str | Path, *, count: int = 5
) -> list[FileReadRecord]:
    if count <= 0:
        return []
    root = Path(session_dir)
    records = _collect_file_event_reads(root)
    if not records:
        records = _collect_tool_result_reads(root)
    return records[-count:]


def render_file_read_block(
    records: list[FileReadRecord],
    *,
    max_chars: int = 5000,
) -> tuple[str, list[dict[str, Any]]]:
    lines: list[str] = []
    metadata: list[dict[str, Any]] = []
    for record in records:
        char_count = len(record.content)
        truncated = char_count > max_chars
        attrs: dict[str, Any] = {
            "path": record.path,
            "mode": record.mode,
        }
        if record.range is not None:
            attrs["range"] = record.range
        attrs.update(
            {
                "truncated": str(truncated).lower(),
                "char_count": str(char_count),
            }
        )
        if truncated:
            scope = "full file" if record.mode == "full" else f"lines {record.range}"
            body = TOO_LARGE_FILE_READ_TEMPLATE.format(
                path=record.path,
                scope=scope,
                char_count=char_count,
            )
        else:
            body = record.content
        lines.append(str(Prompt(body).wrap("file", **attrs)))
        row: dict[str, Any] = {
            "path": record.path,
            "mode": record.mode,
            "included": not truncated,
            "char_count": char_count,
            "truncated": truncated,
        }
        if record.range is not None:
            row["range"] = record.range
        metadata.append(row)
    return str(Prompt("\n".join(lines)).wrap("file-read")), metadata


def _collect_file_event_reads(session_dir: Path) -> list[FileReadRecord]:
    records: list[FileReadRecord] = []
    for row in EventStore.iter_jsonl(session_dir / "logs" / "file_events.jsonl") or []:
        if row.get("type") != "file.read":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        content = payload.get("returned_content")
        if not isinstance(content, str):
            continue
        path = payload.get("path")
        if not isinstance(path, str) or not path:
            continue
        mode, range_text = _mode_and_range(payload.get("returned_range"))
        records.append(
            FileReadRecord(
                path=path,
                mode=mode,
                range=range_text,
                content=content,
                created_at=str(row.get("created_at") or ""),
            )
        )
    return records


def _collect_tool_result_reads(session_dir: Path) -> list[FileReadRecord]:
    calls: dict[str, dict[str, Any]] = {}
    for row in EventStore.iter_jsonl(session_dir / "logs" / "tool_calls.jsonl") or []:
        if row.get("tool_name") == "read" and row.get("event_id"):
            calls[str(row["event_id"])] = row
    records: list[FileReadRecord] = []
    for row in EventStore.iter_jsonl(session_dir / "logs" / "tool_results.jsonl") or []:
        call_id = row.get("tool_call_id")
        call = calls.get(str(call_id))
        if call is None:
            continue
        result = row.get("result")
        raw_args = call.get("args")
        args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
        path = _first_string(args, ("file_path",))
        content = _result_content(result)
        if path is None or content is None:
            continue
        mode, range_text = _mode_and_range(
            result.get("range") if isinstance(result, dict) else None
        )
        records.append(
            FileReadRecord(
                path=path,
                mode=mode,
                range=range_text,
                content=content,
                created_at=str(row.get("created_at") or ""),
            )
        )
    return records


def _mode_and_range(value: Any) -> tuple[Literal["full", "partial"], str | None]:
    if not isinstance(value, dict) or value.get("unit") == "full":
        return "full", None
    start = value.get("start", 1)
    end = value.get("end", "EOF")
    return "partial", f"{start}-{end}"


def _result_content(result: Any) -> str | None:
    if isinstance(result, dict):
        content = result.get("content")
        return content if isinstance(content, str) else None
    if isinstance(result, str):
        return result
    return None


def _first_string(mapping: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str) and value:
            return value
    return None
