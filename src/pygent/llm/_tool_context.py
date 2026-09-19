"""Model-visible tool context shared by the provider adapters."""

from __future__ import annotations

import json
from xml.sax.saxutils import escape

from pygent.core import thaw_json
from pygent.tool import ToolResult

_XML_TEXT_ESCAPES = {"\r": "&#13;"}
_XML_ATTRIBUTE_ESCAPES = {'"': "&quot;", "\r": "&#13;"}


def _xml_escape(text: str, *, attribute: bool = False) -> str:
    # Tool payloads embed arbitrary file and process content, so &, <, >, and
    # (inside attributes) double quotes are escaped to keep a payload from
    # forging context markup.  CR becomes a numeric reference because XML
    # parsers normalize literal CR.
    return escape(text, _XML_ATTRIBUTE_ESCAPES if attribute else _XML_TEXT_ESCAPES)


def _attribute_value(value: object) -> str:
    return "true" if value is True else str(value)


def _context_fields(result: ToolResult) -> list[tuple[str, object]]:
    """Collect the ordered semantic fields every adapter projection must carry.

    The shared field list keeps the XML text form and the JSON object form
    from drifting apart as fields are added.
    """

    fields: list[tuple[str, object]] = [("status", result.status)]
    if result.error_kind is not None:
        fields.append(("error_kind", result.error_kind))
    if result.error_code is not None:
        fields.append(("error_code", result.error_code))
    if result.retryable:
        fields.append(("retryable", True))
    if result.side_effect_committed is True:
        fields.append(("side_effect", "committed"))
    elif result.side_effect_committed is None:
        fields.append(("side_effect", "unknown"))
    if result.task is not None:
        fields.append(("task_id", result.task.task_id))
        fields.append(("task_state", result.task.state.value))
    return fields


def encode_tool_context(result: ToolResult) -> str:
    """Render a non-succeeded result as structured context for the model.

    Success stays a bare string; every other status keeps the failure
    classification (error_kind/error_code), the side-effect verdict, and any
    task or output payload visible, so the model can tell "fix the arguments"
    apart from "the side effect already happened" and find a detached task.
    """

    attributes = [
        f'{name.replace("_", "-")}="{_xml_escape(_attribute_value(value), attribute=True)}"'
        for name, value in _context_fields(result)
    ]
    body = ""
    if result.error is not None:
        body += f"<error>{_xml_escape(result.error)}</error>"
    if result.output is not None:
        payload = (
            result.output
            if isinstance(result.output, str)
            else json.dumps(
                thaw_json(result.output), ensure_ascii=False, separators=(",", ":")
            )
        )
        body += f"<output>{_xml_escape(payload)}</output>"
    opening = f"<tool-context {' '.join(attributes)}"
    if not body:
        return f"{opening}/>"
    return f"{opening}>{body}</tool-context>"


def tool_context_payload(result: ToolResult) -> dict[str, object]:
    """Render the same tool-context semantics as a JSON object.

    Gemini's ``functionResponse.response`` is a JSON object slot, so the
    shared fields travel as JSON there instead of the XML text form.
    """

    payload: dict[str, object] = dict(_context_fields(result))
    if result.error is not None:
        payload["error"] = result.error
    if result.output is not None:
        payload["output"] = thaw_json(result.output)
    return payload


__all__ = ["encode_tool_context", "tool_context_payload"]
