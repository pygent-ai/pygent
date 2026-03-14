"""
Session 管理模块：会话生命周期、持久化与恢复。
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pygent.common import PygentString
from pygent.context import BaseContext
from pygent.message import BaseMessage


def _session_dir(workspace_root: str, session_id: str) -> Path:
    """Session 存储目录：{workspace_root}/sessions/{session_id}"""
    return Path(workspace_root) / "sessions" / session_id


def _session_file_path(workspace_root: str, session_id: str) -> Path:
    """Session 文件路径：{workspace_root}/sessions/{session_id}/session.json"""
    return _session_dir(workspace_root, session_id) / "session.json"


class Session:
    """
    会话：管理单次对话生命 周期，持有 Context（PygentList 对话历史）并支持持久化。
    保存时直接写入对应文件夹：{workspace_root}/sessions/{session_id}/session.json
    """

    def __init__(
        self,
        session_id: str,
        workspace_root: str,
        system_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        context: Optional[BaseContext] = None,
    ):
        self.session_id = session_id
        self.workspace_root = os.path.abspath(workspace_root)
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at

        if context is not None:
            self.context = context
        else:
            self.context = BaseContext(system_prompt=system_prompt or "你是一个小助手。")

    @property
    def session_dir(self) -> Path:
        """Session 对应的存储目录。"""
        return _session_dir(self.workspace_root, self.session_id)

    def save(self, format: str = "json") -> str:
        """
        保存到对应文件夹：{workspace_root}/sessions/{session_id}/session.json
        无需传入 path，自动写入 session 目录。
        """
        self.updated_at = datetime.now().isoformat()
        path = _session_file_path(self.workspace_root, self.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        history = getattr(self.context, "history", None)
        history_data = []
        if history is not None:
            for msg in history.data if hasattr(history, "data") else history:
                history_data.append(msg.to_dict())

        system_prompt = getattr(self.context, "system_prompt", None)
        system_prompt_str = system_prompt.data if hasattr(system_prompt, "data") else (system_prompt or "")

        payload = {
            "version": "1.0",
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "system_prompt": system_prompt_str,
            "history": history_data,
            "metadata": self.metadata,
        }

        if format == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return str(path.absolute())

    @classmethod
    def load(
        cls,
        workspace_root: str,
        session_id: str,
        format: str = "auto",
    ) -> "Session":
        """
        从对应文件夹加载 Session。
        路径：{workspace_root}/sessions/{session_id}/session.json
        """
        path = _session_file_path(workspace_root, session_id)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")

        fmt = path.suffix.lstrip(".").lower() if format == "auto" else format
        if fmt not in ("json",):
            fmt = "json"

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        system_prompt = payload.get("system_prompt")
        context = BaseContext(system_prompt=None)
        if system_prompt:
            context.system_prompt = PygentString(system_prompt)
        history_data = payload.get("history", [])
        for d in history_data:
            if isinstance(d, dict):
                msg = BaseMessage.from_serialized_dict(d)
                context.add_message(msg)

        session = cls(
            session_id=payload.get("session_id", session_id),
            workspace_root=payload.get("workspace_root", workspace_root),
            system_prompt=system_prompt,
            metadata=payload.get("metadata", {}),
            context=context,
        )
        session.created_at = payload.get("created_at", session.created_at)
        session.updated_at = payload.get("updated_at", session.updated_at)
        return session
