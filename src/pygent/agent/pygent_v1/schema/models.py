from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pygent.agent.pygent_v1.core.paths import project_pygent_root, user_pygent_root
from pygent.llm import ModelConfig


def _abs_path(value: str | Path) -> str:
    return str(Path(value).expanduser().resolve())


def _require(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def native_model_config_complete(mapping: dict[str, Any]) -> bool:
    """True when the native subtree can satisfy Pygent's non-empty invariants.

    Pygent's ``ModelConfig.from_mapping()`` rejects empty ``connections`` and
    ``models``; the settings UI also saves connection-only or group-less
    states, which the runtime must load as an explicit unconfigured state
    instead of failing.
    """
    return all(mapping.get(key) for key in ("connections", "models", "model_groups"))


@dataclass(slots=True)
class ModelRetryConfig:
    max_attempts_per_model: int = 2
    attempt_idle_timeout_seconds: float = 60.0
    backoff_initial: float = 0.5
    backoff_maximum: float = 4.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        self.max_attempts_per_model = int(self.max_attempts_per_model)
        self.attempt_idle_timeout_seconds = float(self.attempt_idle_timeout_seconds)
        self.backoff_initial = float(self.backoff_initial)
        self.backoff_maximum = float(self.backoff_maximum)
        self.backoff_multiplier = float(self.backoff_multiplier)
        if self.max_attempts_per_model < 1:
            raise ValueError("max_attempts_per_model must be at least one")
        if self.attempt_idle_timeout_seconds <= 0:
            raise ValueError("attempt_idle_timeout_seconds must be greater than zero")
        if not 0 <= self.backoff_initial <= self.backoff_maximum:
            raise ValueError("backoff requires 0 <= initial <= maximum")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least one")


@dataclass(slots=True)
class ResolvedAgentConfig:
    alias: str
    default_model_group: str

    def __post_init__(self) -> None:
        self.alias = _require(self.alias, "alias")
        self.default_model_group = _require(
            self.default_model_group, "default_model_group"
        )

    def safe_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "default_model_group": self.default_model_group,
        }


@dataclass(slots=True)
class RuntimeDurabilityConfig:
    mode: Literal["disabled", "preferred", "required"] = "preferred"
    history_path: str = "runtime/executions-v1.sqlite3"

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "preferred", "required"}:
            raise ValueError("runtime durability mode is invalid")
        self.history_path = _require(self.history_path, "runtime history_path")


@dataclass(slots=True)
class RuntimeSteeringConfig:
    """Steering delivery mode for interactive agent turns.

    ``immediate`` lets a steering message interrupt the in-flight model or
    tool step so it becomes the next model input as soon as possible; since
    pygent 0.3.33 the interrupt also cooperatively cancels the in-flight
    step on durable executions, so turns keep per-turn crash recovery and
    the journal that feeds the trace Context inspector. ``wait`` keeps the
    same durable turns but delivers steering only at the next processing
    boundary without interrupting.
    """

    mode: Literal["wait", "immediate"] = "immediate"

    def __post_init__(self) -> None:
        if self.mode not in {"wait", "immediate"}:
            raise ValueError("runtime steering mode is invalid")


@dataclass(slots=True)
class RuntimeCapacityConfig:
    scope: Literal["runtime_instance", "deployment"] = "runtime_instance"
    coordinator_path: str = "runtime/capacity-v1.sqlite3"

    def __post_init__(self) -> None:
        if self.scope not in {"runtime_instance", "deployment"}:
            raise ValueError("runtime capacity scope is invalid")
        self.coordinator_path = _require(
            self.coordinator_path, "capacity coordinator_path"
        )


@dataclass(slots=True)
class RuntimeApprovalConfig:
    enabled: bool = True
    timeout_seconds: float = 300.0
    preauthorized_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        self.timeout_seconds = float(self.timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("approval timeout_seconds must be greater than zero")
        self.preauthorized_tools = tuple(
            _require(item, "preauthorized tool") for item in self.preauthorized_tools
        )


@dataclass(slots=True)
class MCPServerConfig:
    name: str
    transport: Literal["stdio", "sse"] = "stdio"
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: str | None = None
    env_from: tuple[str, ...] = ()
    url: str | None = None
    headers_env: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    required: bool = False

    def __post_init__(self) -> None:
        self.name = _require(self.name, "MCP server name")
        self.args = tuple(str(item) for item in self.args)
        self.env_from = tuple(
            _require(item, "MCP env_from item") for item in self.env_from
        )
        self.timeout = float(self.timeout)
        if self.timeout <= 0:
            raise ValueError("MCP timeout must be greater than zero")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP server requires command")
        if self.transport == "sse" and not self.url:
            raise ValueError("sse MCP server requires url")


@dataclass(slots=True)
class DelegationConfig:
    max_depth: int = 4

    def __post_init__(self) -> None:
        self.max_depth = int(self.max_depth)
        if self.max_depth < 1:
            raise ValueError("delegation limits must be positive")


@dataclass(slots=True)
class RealtimeInteractionConfig:
    """Silent-work interaction (speak, fork replies, nudges) — off by default."""

    enabled: bool = False
    nudge_interval: int = 5
    fork_deadline_seconds: float = 60.0
    fork_reasoning_effort: str | None = "low"

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if isinstance(self.nudge_interval, bool) or not isinstance(
            self.nudge_interval, int
        ) or self.nudge_interval < 1:
            raise ValueError("nudge_interval must be a positive integer")
        if (
            isinstance(self.fork_deadline_seconds, bool)
            or not isinstance(self.fork_deadline_seconds, (int, float))
            or self.fork_deadline_seconds <= 0
        ):
            raise ValueError("fork_deadline_seconds must be positive")
        if self.fork_reasoning_effort is not None:
            self.fork_reasoning_effort = _require(
                self.fork_reasoning_effort, "fork_reasoning_effort"
            )


@dataclass(slots=True)
class SchedulerConfig:
    """Settings for the independent `pygent_v1 scheduler` daemon."""

    agent_create_enabled: bool = False
    session: str = "scheduler"
    max_concurrent: int = 2
    poll_seconds: float = 10.0

    def __post_init__(self) -> None:
        self.agent_create_enabled = bool(self.agent_create_enabled)
        self.session = _require(self.session, "scheduler session")
        if isinstance(self.max_concurrent, bool) or not isinstance(
            self.max_concurrent, int
        ):
            raise ValueError("scheduler max_concurrent must be an integer")  # noqa: TRY004 - validation contract errors are value errors
        if self.max_concurrent < 1:
            raise ValueError("scheduler max_concurrent must be positive")
        if isinstance(self.poll_seconds, bool) or not isinstance(
            self.poll_seconds, (int, float)
        ):
            raise ValueError("scheduler poll_seconds must be a number")  # noqa: TRY004 - validation contract errors are value errors
        self.poll_seconds = float(self.poll_seconds)
        if self.poll_seconds <= 0:
            raise ValueError("scheduler poll_seconds must be positive")


@dataclass(slots=True)
class EternalConversationConfig:
    enabled: bool = False
    extractor_agent_alias: str | None = None
    builder_agent_alias: str | None = None
    dynamic_memory_cli_path: str | None = None
    extraction_trigger: str = "turn"
    extraction_trigger_ratio: float = 0.7

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        for name in ("extractor_agent_alias", "builder_agent_alias"):
            value = getattr(self, name)
            if value is not None:
                setattr(self, name, _require(value, name))
        if self.dynamic_memory_cli_path is not None:
            self.dynamic_memory_cli_path = _abs_path(self.dynamic_memory_cli_path)
        if self.extraction_trigger not in ("turn", "context_ratio"):
            raise ValueError(
                "extraction_trigger must be 'turn' or 'context_ratio'"
            )
        if isinstance(self.extraction_trigger_ratio, bool) or not isinstance(
            self.extraction_trigger_ratio, (int, float)
        ):
            raise ValueError("extraction_trigger_ratio must be a number")  # noqa: TRY004 - validation contract errors are value errors
        self.extraction_trigger_ratio = float(self.extraction_trigger_ratio)
        if not 0 < self.extraction_trigger_ratio < 1:
            raise ValueError(
                "extraction_trigger_ratio must be between zero and one"
            )


@dataclass(slots=True)
class BashCliPreset:
    name: str
    command: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        self.name = _require(self.name, "name")
        if self.command is None:
            self.command = ""
        if self.description is None:
            self.description = ""


def default_cli_bash_presets() -> list[BashCliPreset]:
    return [
        BashCliPreset(
            name="rg",
            command="rg --help",
            description="Fast recursive text search. Prefer it for code and file text search.",
        ),
        BashCliPreset(
            name="pyright",
            command="pyright --help",
            description="Python type checker. Use it for static type validation when available.",
        ),
        BashCliPreset(
            name="pygent.v1-session",
            command="uv run lara session --help",
            description=(
                "Session CLI. Use `uv run lara session start` for a new background Agent, "
                "and `uv run lara session run` or `chat` for user turns."
            ),
        ),
        BashCliPreset(
            name="pygent.v1-automation",
            command="uv run lara automation --help",
            description=(
                "Manage pygent_v1 scheduled Agent tasks with "
                "`create|list|show|update|pause|resume|delete|run|runs`. Use `--session` with "
                "the current session ID for a heartbeat, or `--standalone` for a new "
                "session per run. Run `uv run lara automation --help` for complete syntax."
            ),
        ),
    ]


@dataclass(slots=True)
class RunConfig:
    workspace_root: str
    pygent_root: str = ""
    session_id: str | None = None
    case_file: str | None = None
    max_steps: int = -1
    agent_alias: str = "default"
    # The configured static system prompt: the composition root's text, kept
    # as data. The prompt registry ships empty; modules are injected by the
    # composition root on top of this base.
    system_prompt: str = ""
    resolved_agent: ResolvedAgentConfig | None = field(
        default=None, repr=False, compare=False
    )
    # Retry policy from the agent profile; it exists regardless of whether the
    # model configuration is complete enough to bind a model group.
    model_retry: ModelRetryConfig = field(
        default_factory=ModelRetryConfig, repr=False, compare=False
    )
    model_config_mapping: dict[str, Any] = field(default_factory=dict)
    model_config: ModelConfig | None = field(default=None, repr=False, compare=False)
    model_configuration_status: Literal["configured", "unconfigured", "legacy"] = (
        "unconfigured"
    )
    model_configuration_error: str | None = None
    user_identity: str = "default"
    cli_bash_presets: list[BashCliPreset] = field(
        default_factory=default_cli_bash_presets
    )
    bash_full_output_allowlist: list[str] = field(default_factory=list)
    allow_read_outside_workspace: bool = True
    user_pygent_root: str = ""
    context_window: int | None = None
    context_compression_enabled: bool = True
    context_compression_trigger_ratio: float = 0.9
    context_compression_file_read_count: int = 5
    context_compression_file_read_max_chars: int = 5000
    runtime_durability: RuntimeDurabilityConfig = field(
        default_factory=RuntimeDurabilityConfig
    )
    runtime_steering: RuntimeSteeringConfig = field(
        default_factory=RuntimeSteeringConfig
    )
    runtime_capacity: RuntimeCapacityConfig = field(
        default_factory=RuntimeCapacityConfig
    )
    runtime_approvals: RuntimeApprovalConfig = field(
        default_factory=RuntimeApprovalConfig
    )
    mcp_servers: list[MCPServerConfig] = field(default_factory=list)
    delegation: DelegationConfig = field(default_factory=DelegationConfig)
    eternal_conversation: EternalConversationConfig = field(
        default_factory=EternalConversationConfig
    )
    realtime_interaction: RealtimeInteractionConfig = field(
        default_factory=RealtimeInteractionConfig
    )
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def __post_init__(self) -> None:
        self.workspace_root = _abs_path(self.workspace_root)
        self.user_pygent_root = _abs_path(self.user_pygent_root or user_pygent_root())
        self.pygent_root = _abs_path(
            self.pygent_root
            or project_pygent_root(self.workspace_root, self.user_pygent_root)
        )
        if self.case_file is not None:
            self.case_file = _abs_path(self.case_file)
        if self.max_steps != -1 and self.max_steps <= 0:
            raise ValueError("max_steps must be -1 or greater than 0")
        if self.context_window is not None:
            self.context_window = int(self.context_window)
            if self.context_window <= 0:
                raise ValueError("context_window must be greater than 0")
        self.agent_alias = _require(self.agent_alias, "agent_alias")
        if not isinstance(self.model_retry, ModelRetryConfig):
            self.model_retry = ModelRetryConfig(**self.model_retry)
        if (
            self.model_config is None
            and native_model_config_complete(self.model_config_mapping)
        ):
            self.model_config = ModelConfig.from_mapping(self.model_config_mapping)
        if self.model_config is not None:
            self.model_configuration_status = "configured"
            self.model_configuration_error = None
        elif self.model_configuration_status == "configured":
            raise ValueError("configured model status requires model_config")
        self.user_identity = _require(self.user_identity or "default", "user_identity")
        self.cli_bash_presets = [
            preset if isinstance(preset, BashCliPreset) else BashCliPreset(**preset)
            for preset in self.cli_bash_presets
        ]
        if not isinstance(self.bash_full_output_allowlist, list):
            raise ValueError("bash_full_output_allowlist must be a list")  # noqa: TRY004 - validation contract errors are value errors
        self.bash_full_output_allowlist = [
            _require(item, "bash_full_output_allowlist item")
            for item in self.bash_full_output_allowlist
        ]
        self.context_compression_enabled = bool(self.context_compression_enabled)
        self.context_compression_trigger_ratio = float(
            self.context_compression_trigger_ratio
        )
        if self.context_compression_trigger_ratio <= 0:
            raise ValueError("context_compression_trigger_ratio must be greater than 0")
        self.context_compression_file_read_count = int(
            self.context_compression_file_read_count
        )
        if self.context_compression_file_read_count < 0:
            raise ValueError(
                "context_compression_file_read_count must be greater than or equal to 0"
            )
        self.context_compression_file_read_max_chars = int(
            self.context_compression_file_read_max_chars
        )
        if self.context_compression_file_read_max_chars < 0:
            raise ValueError(
                "context_compression_file_read_max_chars must be greater than or equal to 0"
            )
        for name, cls in (
            ("runtime_durability", RuntimeDurabilityConfig),
            ("runtime_steering", RuntimeSteeringConfig),
            ("runtime_capacity", RuntimeCapacityConfig),
            ("runtime_approvals", RuntimeApprovalConfig),
            ("delegation", DelegationConfig),
            ("eternal_conversation", EternalConversationConfig),
            ("realtime_interaction", RealtimeInteractionConfig),
            ("scheduler", SchedulerConfig),
        ):
            value = getattr(self, name)
            if not isinstance(value, cls):
                setattr(self, name, cls(**value))
        for settings, field_name in (
            (self.runtime_durability, "history_path"),
            (self.runtime_capacity, "coordinator_path"),
        ):
            path = Path(getattr(settings, field_name)).expanduser()
            if not path.is_absolute():
                path = Path(self.pygent_root) / path
            setattr(settings, field_name, str(path.resolve()))
        self.mcp_servers = [
            item if isinstance(item, MCPServerConfig) else MCPServerConfig(**item)
            for item in self.mcp_servers
        ]

    def to_dict(self) -> dict[str, Any]:
        data = {
            item.name: _plain_dataclass_value(getattr(self, item.name))
            for item in fields(self)
            if item.name != "model_config"
        }
        data["resolved_agent"] = (
            None if self.resolved_agent is None else self.resolved_agent.safe_dict()
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        clean = dict(data)
        resolved = clean.get("resolved_agent")
        if isinstance(resolved, dict):
            # Run settings persisted before retry moved from the agent binding
            # to model_retry still carry it inside resolved_agent.
            legacy = dict(resolved)
            retry = legacy.pop("retry", None)
            clean["resolved_agent"] = ResolvedAgentConfig(**legacy)
            if retry is not None and "model_retry" not in clean:
                clean["model_retry"] = retry
        return cls(**clean)


def _plain_dataclass_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _plain_dataclass_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {key: _plain_dataclass_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_dataclass_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_plain_dataclass_value(item) for item in value)
    return value


@dataclass(slots=True)
class SessionRef:
    session_id: str
    session_dir: str
    workspace_root: str

    def __post_init__(self) -> None:
        self.session_id = _require(self.session_id, "session_id")
        self.session_dir = _abs_path(self.session_dir)
        self.workspace_root = _abs_path(self.workspace_root)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CaseRunRef:
    session_id: str
    case_id: str
    case_run_id: str
    run_dir: str

    def __post_init__(self) -> None:
        self.session_id = _require(self.session_id, "session_id")
        self.case_id = _require(self.case_id, "case_id")
        self.case_run_id = _require(self.case_run_id, "case_run_id")
        self.run_dir = _abs_path(self.run_dir)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaseRunRef:
        return cls(**data)


class CaseRunStatus(StrEnum):
    """Durable verdict of one case run, shared by every layer.

    Sessions materialize this value in metadata for cheap listing.
    """

    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"

    @property
    def terminal(self) -> bool:
        return self is not CaseRunStatus.RUNNING

    @classmethod
    def from_value(cls, value: Any) -> CaseRunStatus | None:
        try:
            return cls(str(value))
        except ValueError:
            return None


@dataclass(slots=True)
class ContextEvent:
    id: str
    session_id: str
    case_id: str | None
    case_run_id: str | None
    turn_id: str | None
    type: str
    timestamp: str
    actor: Literal["user", "assistant", "tool", "system"]
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _require(self.id, "id")
        self.session_id = _require(self.session_id, "session_id")
        self.type = _require(self.type, "type")
        self.timestamp = _require(self.timestamp, "timestamp")
        if self.actor not in {"user", "assistant", "tool", "system"}:
            raise ValueError("actor must be one of user, assistant, tool, system")
        if not isinstance(self.payload, dict):
            raise ValueError("payload must be a dict")  # noqa: TRY004 - validation contract errors are value errors

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextEvent:
        return cls(**data)


@dataclass(slots=True)
class SessionSpec:
    case_id: str
    mode: str = "new"
    session_id: str | None = None
    source_session_id: str | None = None

    def __post_init__(self) -> None:
        self.case_id = _require(self.case_id, "case_id")
        if self.mode not in {"new", "resume", "fork", "shared"}:
            raise ValueError("mode must be one of new, resume, fork, shared")


@dataclass(slots=True)
class AgentSession:
    session_id: str
    workspace_root: str
    session_dir: str
    created_at: str
    updated_at: str
    model_group_name: str = ""
    selected_model_key: str = ""
    system_prompt: str = ""
    status: str = "normal"
    token_usage: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.session_id = _require(self.session_id, "session_id")
        self.workspace_root = _abs_path(self.workspace_root)
        self.session_dir = _abs_path(self.session_dir)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["version"] = "1.0"
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSession:
        version = str(data.get("version") or "1.0")
        if version != "1.0":
            raise ValueError(f"unsupported AgentSession version: {version}")
        return cls(
            session_id=data["session_id"],
            workspace_root=data["workspace_root"],
            session_dir=data.get("session_dir", ""),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            model_group_name=data.get("model_group_name", ""),
            selected_model_key=data.get("selected_model_key", ""),
            system_prompt=data.get("system_prompt", ""),
            status=data.get("status", "normal"),
            token_usage=dict(data.get("token_usage") or {}),
            history=list(data.get("history") or []),
            metadata=dict(data.get("metadata") or {}),
        )


def build_run_config(
    *,
    workspace_root: str | Path,
    model_config_mapping: dict[str, Any],
    default_model_group: str,
    agent_alias: str = "default",
    session_id: str | None = None,
    user_pygent_root: str | Path | None = None,
    max_steps: int | None = None,
    system_prompt: str = "",
) -> RunConfig:
    """Programmatic :class:`RunConfig` entry: no config file, no credentials.

    The shape-only construction path: callers provide the native model
    mapping (``connections`` / ``models`` / ``model_groups``) directly.
    File reading, merging, and credential injection stay with the shell
    loader. Binding and agent resolution follow the same rules as
    ``load_run_config``: the mapping must be complete
    (:func:`native_model_config_complete`), and the agent alias must
    reference an existing model group.
    """
    if not native_model_config_complete(model_config_mapping):
        raise ValueError(
            "model configuration incomplete: connections, models and "
            "model_groups are required"
        )
    _require(default_model_group, "default_model_group")
    config = RunConfig(
        workspace_root=str(workspace_root),
        session_id=session_id,
        agent_alias=agent_alias,
        user_pygent_root="" if user_pygent_root is None else str(user_pygent_root),
        max_steps=-1 if max_steps is None else int(max_steps),
        system_prompt=str(system_prompt),
        resolved_agent=ResolvedAgentConfig(
            alias=agent_alias,
            default_model_group=default_model_group,
        ),
        model_config_mapping=dict(model_config_mapping),
    )
    if config.model_config is None or default_model_group not in config.model_config.model_groups:
        raise ValueError(
            f"Agent alias {agent_alias!r} references unknown model group "
            f"{default_model_group!r}"
        )
    return config
