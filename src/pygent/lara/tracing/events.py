from __future__ import annotations

import json
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from pygent.lara.core.io import append_jsonl, utc_now, write_json, write_text
from pygent.lara.core.redaction import redact_secrets
from pygent.lara.schema import CaseRunRef, ContextEvent

DESIGN_EVENT_TYPES = frozenset(
    {
        "case.started",
        "case.finished",
        "conversation.automation_trigger",
        "conversation.user_message",
        "conversation.assistant_message",
        "conversation.tool_message",
        "model.request",
        "model.response",
        "prompt.rendered",
        "tool.call",
        "tool.unknown",
        "tool.result",
        "diff.created",
        "file.read",
        "file.write",
        "file.edit",
        "file.delete",
        "context.checkpoint",
        "context.projection_created",
        "analysis.created",
        "test.generated",
        "repair.started",
        "repair.patch_created",
        "repair.finished",
        "regression.started",
        "regression.finished",
    }
)

LARA_EVENT_TYPES = DESIGN_EVENT_TYPES | frozenset(
    {
        "chat.started",
        "chat.finished",
        "runtime.error",
        "runtime.file_scan.incomplete",
        "runtime.cancelled",
        "runtime.tool_task.started",
        "runtime.tool_task.completed",
        "runtime.tool_task.failed",
        "runtime.tool_task.cancelled",
        "runtime.tool_task.unknown",
        "error",
    }
)


class EventStore:
    def __init__(self, case_run_ref: CaseRunRef):
        self.case_run_ref = case_run_ref
        self.run_dir = Path(case_run_ref.run_dir)
        self.session_dir = _find_session_dir(self.run_dir)
        self.events_path = self.run_dir / "events.jsonl"
        self.messages_path = self.run_dir / "messages.jsonl"
        self.tool_calls_path = self.run_dir / "tool_calls.jsonl"
        self.tool_results_path = self.run_dir / "tool_results.jsonl"
        self.file_events_path = self.run_dir / "file_events.jsonl"

    def append(
        self,
        event: ContextEvent | str,
        *,
        actor: str | None = None,
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> str:
        context_event = self._coerce_event(
            event,
            actor=actor,
            payload=payload,
            turn_id=turn_id,
        )
        self._append_event(context_event)
        return context_event.id

    def append_error(
        self,
        error: BaseException | str,
        *,
        event_type: str = "error",
        actor: str = "system",
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> str:
        error_payload = dict(payload or {})
        if isinstance(error, BaseException):
            error_payload.update({"error": str(error), "error_type": type(error).__name__})
        else:
            error_payload.update({"error": error, "error_type": "Error"})
        return self.append(event_type, actor=actor, payload=error_payload, turn_id=turn_id)

    def list_by_run(
        self,
        session_id: str | None = None,
        case_run_id: str | None = None,
    ) -> list[ContextEvent]:
        target_session_id = session_id or self.case_run_ref.session_id
        target_case_run_id = case_run_id or self.case_run_ref.case_run_id
        return [
            event
            for event in self.list_events()
            if event.session_id == target_session_id and event.case_run_id == target_case_run_id
        ]

    def list_events(self) -> list[ContextEvent]:
        return [ContextEvent.from_dict(row) for row in self.iter_jsonl(self.events_path)]

    @staticmethod
    def iter_jsonl(path: str | Path):
        jsonl_path = Path(path)
        if not jsonl_path.exists():
            return
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # JSONL files are audit projections, not recovery state. A
                    # torn row must not make later valid audit rows unreadable.
                    continue

    def _append_event(self, event: ContextEvent) -> None:
        persisted_source = _clone_event(event)
        if persisted_source.type == "prompt.rendered":
            self._persist_rendered_prompt(persisted_source)
        persisted_event = redact_secrets(persisted_source.to_dict())
        append_jsonl(self.events_path, persisted_event)
        if event.type.startswith("conversation."):
            record = redact_secrets(_message_record(event))
            append_jsonl(
                self.messages_path,
                record,
            )
            self._append_session_history(record)
        elif event.type == "tool.call":
            record = redact_secrets(_tool_call_record(event))
            append_jsonl(self.tool_calls_path, record)
            self._append_session_log("tool_calls.jsonl", record)
        elif event.type == "tool.result":
            record = redact_secrets(_tool_result_record(event))
            append_jsonl(self.tool_results_path, record)
            self._append_session_log("tool_results.jsonl", record)
        elif event.type.startswith("file."):
            record = redact_secrets(_file_event_record(event))
            append_jsonl(self.file_events_path, record)
            self._append_session_log("file_events.jsonl", record)
        elif event.type == "diff.created":
            record = redact_secrets(_diff_event_record(event))
            append_jsonl(self.run_dir / "diffs" / "diff_events.jsonl", record)
            self._append_session_log("diff_events.jsonl", record)

    def _append_session_history(self, record: dict[str, Any]) -> None:
        if self.session_dir is None:
            return
        path = self.session_dir / "context" / "history.jsonl"
        row = {"seq": _next_jsonl_seq(path), **record}
        append_jsonl(path, row, durable=record.get("checkpoint_id") is not None)

    def _append_session_log(self, name: str, record: dict[str, Any]) -> None:
        if self.session_dir is None:
            return
        append_jsonl(self.session_dir / "logs" / name, record)

    def _persist_rendered_prompt(self, event: ContextEvent) -> None:
        prompt = event.payload.pop("prompt", None)
        if prompt is None:
            prompt = event.payload.pop("content", None)
        if prompt is None:
            prompt = event.payload.pop("text", None)
        if not isinstance(prompt, str):
            return
        persisted_prompt = redact_secrets(prompt)

        prompt_file_name = _turn_file_name(event.turn_id)
        run_prompt_dir = self.run_dir / "rendered_prompts"
        run_text_path = run_prompt_dir / f"{prompt_file_name}.txt"
        run_metadata_path = run_prompt_dir / f"{prompt_file_name}.json"
        prompt_metadata = {
            **_event_envelope(event),
            "prompt_hash": event.payload.get("prompt_hash"),
            "module_ids": event.payload.get("module_ids", []),
            "static_module_ids": event.payload.get("static_module_ids", []),
            "request_system_module_ids": event.payload.get("request_system_module_ids", []),
            "modules": event.payload.get("modules", []),
            "dynamic_inputs": event.payload.get("dynamic_inputs", {}),
        }
        write_text(run_text_path, persisted_prompt)
        write_json(run_metadata_path, prompt_metadata)
        event.payload["prompt_text_path"] = str(run_text_path)
        event.payload["prompt_metadata_path"] = str(run_metadata_path)

        if self.session_dir is None:
            return
        session_prompt_dir = self.session_dir / "context" / "rendered_prompts" / str(event.case_run_id or "session")
        session_text_path = session_prompt_dir / f"{prompt_file_name}.txt"
        session_metadata_path = session_prompt_dir / f"{prompt_file_name}.json"
        write_text(session_text_path, persisted_prompt)
        write_json(session_metadata_path, prompt_metadata)
        append_jsonl(
            self.session_dir / "logs" / "rendered_prompts.jsonl",
            {
                **prompt_metadata,
                "prompt_text_path": str(session_text_path),
                "prompt_metadata_path": str(session_metadata_path),
            },
        )

    def _coerce_event(
        self,
        event: ContextEvent | str,
        *,
        actor: str | None = None,
        payload: dict[str, Any] | None = None,
        turn_id: str | None = None,
    ) -> ContextEvent:
        if isinstance(event, ContextEvent):
            self._validate_event_scope(event)
            self._validate_event_type(event.type)
            return event
        if actor is None:
            raise TypeError("actor is required when appending by event type")
        self._validate_event_type(event)
        return ContextEvent(
            id=f"evt_{uuid.uuid4().hex}",
            session_id=self.case_run_ref.session_id,
            case_id=self.case_run_ref.case_id,
            case_run_id=self.case_run_ref.case_run_id,
            turn_id=turn_id,
            type=event,
            timestamp=utc_now(),
            actor=actor,  # type: ignore[arg-type]
            payload=payload or {},
        )

    def _validate_event_scope(self, event: ContextEvent) -> None:
        expected = self.case_run_ref
        if event.session_id != expected.session_id:
            raise ValueError("event session_id does not match this run")
        if event.case_id != expected.case_id:
            raise ValueError("event case_id does not match this run")
        if event.case_run_id != expected.case_run_id:
            raise ValueError("event case_run_id does not match this run")

    @staticmethod
    def _validate_event_type(event_type: str) -> None:
        if event_type not in LARA_EVENT_TYPES:
            raise ValueError(f"Unsupported event type: {event_type}")

def _event_envelope(event: ContextEvent) -> dict[str, Any]:
    return {
        "event_id": event.id,
        "session_id": event.session_id,
        "case_id": event.case_id,
        "case_run_id": event.case_run_id,
        "turn_id": event.turn_id,
        "created_at": event.timestamp,
    }


def _tool_call_record(event: ContextEvent) -> dict[str, Any]:
    record = {
        **_event_envelope(event),
        "tool_name": event.payload.get("tool_name"),
        "args": event.payload.get("args", {}),
    }
    if event.payload.get("model_tool_call_id") is not None:
        record["model_tool_call_id"] = event.payload.get("model_tool_call_id")
    return record


def _clone_event(event: ContextEvent) -> ContextEvent:
    return ContextEvent.from_dict(deepcopy(event.to_dict()))


def _message_record(event: ContextEvent) -> dict[str, Any]:
    return {
        **event.payload,
        **_event_envelope(event),
        "role": event.payload.get("role", event.actor),
        "content": event.payload.get("content", ""),
    }


def _tool_result_record(event: ContextEvent) -> dict[str, Any]:
    record = {
        **_event_envelope(event),
        "tool_call_id": event.payload.get("tool_call_id"),
        "status": event.payload.get("status", "success"),
        "result": event.payload.get("result"),
    }
    if event.payload.get("error") is not None:
        record["error"] = event.payload.get("error")
    if event.payload.get("error_type") is not None:
        record["error_type"] = event.payload.get("error_type")
    if event.payload.get("details") is not None:
        record["details"] = event.payload.get("details")
    if event.payload.get("model_tool_call_id") is not None:
        record["model_tool_call_id"] = event.payload.get("model_tool_call_id")
    for key in ("task", "framework_status", "next_action"):
        if key in event.payload:
            record[key] = event.payload[key]
    return record


def _file_event_record(event: ContextEvent) -> dict[str, Any]:
    return {
        **_event_envelope(event),
        "type": event.type,
        "path": event.payload.get("path"),
        "content_hash": event.payload.get("content_hash"),
        "payload": event.payload,
    }


def _diff_event_record(event: ContextEvent) -> dict[str, Any]:
    payload = event.payload
    return {
        **_event_envelope(event),
        "diff_id": payload.get("diff_id"),
        "tool_call_id": payload.get("tool_call_id"),
        "tool_name": payload.get("tool_name"),
        "path": payload.get("path"),
        "relative_path": payload.get("relative_path"),
        "change_type": payload.get("change_type"),
        "before_exists": payload.get("before_exists"),
        "after_exists": payload.get("after_exists"),
        "before_hash": payload.get("before_hash"),
        "after_hash": payload.get("after_hash"),
        "snapshot_before_path": payload.get("snapshot_before_path"),
        "snapshot_after_path": payload.get("snapshot_after_path"),
        "patch_available": payload.get("patch_available", False),
        "patch_path": payload.get("patch_path"),
        "patch_char_count": payload.get("patch_char_count", 0),
        "patch_line_count": payload.get("patch_line_count", 0),
        "additions": payload.get("additions"),
        "deletions": payload.get("deletions"),
        "patch_unavailable_reason": payload.get("patch_unavailable_reason"),
    }


def _find_session_dir(run_dir: Path) -> Path | None:
    for parent in [run_dir, *run_dir.parents]:
        if (parent / "session.json").exists():
            return parent
    return None


_INVALID_FILE_NAME_CHARS = frozenset('<>:"/\\|?*')


def _turn_file_name(turn_id: str | None) -> str:
    """Sanitize a turn id for prompt files while metadata keeps the original.

    Realtime supersede turns are named ``fork:<input>:g<generation>``; colons
    are invalid in Windows file names, so the persisted prompt path must not
    embed the raw turn id.
    """

    raw = str(turn_id or "turn-unknown")
    sanitized = "".join(
        "-" if char in _INVALID_FILE_NAME_CHARS or ord(char) < 32 else char
        for char in raw
    ).strip(" .")
    return sanitized or "turn-unknown"


def _next_jsonl_seq(path: Path) -> int:
    if not path.exists():
        return 1
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count + 1
