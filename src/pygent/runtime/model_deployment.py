"""Binding-scoped immutable model profile storage and admission pinning."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, Protocol, Self, TypeVar

import aiosqlite

from pygent.llm import (
    FallbackPolicy,
    ModelDeploymentConflictError,
    ModelDeploymentUnavailableError,
    ModelGroupConfig,
    ModelProfileSelectionError,
    ModelProfileSnapshot,
    ModelResourceBundle,
    ModelResourceRef,
    ModelRoute,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _sqlite_serialized(
    method: Callable[
        Concatenate[SQLiteModelDeploymentStore, _P],
        Coroutine[object, object, _R],
    ],
) -> Callable[
    Concatenate[SQLiteModelDeploymentStore, _P], Coroutine[object, object, _R]
]:
    @wraps(method)
    async def wrapped(
        self: SQLiteModelDeploymentStore, /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _R:
        async with self._sqlite_write_lock:
            return await method(self, *args, **kwargs)

    return wrapped


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _group_value(group: ModelGroupConfig) -> dict[str, object]:
    return {
        "name": group.name,
        "routes": [
            {"route_id": route.route_id, "provider": route.provider, "model": route.model}
            for route in group.routes
        ],
        "fallback": list(group.fallback.order),
        "max_concurrency": group.max_concurrency,
        "capacity_key": group.capacity_key,
        "resolution": group.resolution.value,
    }


def _bundle_value(bundle: ModelResourceBundle | None) -> object:
    return None if bundle is None else bundle.to_dict()


def _snapshot_value(snapshot: ModelProfileSnapshot) -> dict[str, object]:
    return {
        "deployment_scope_id": snapshot.deployment_scope_id,
        "group_name": snapshot.group_name,
        "profile": snapshot.profile,
        "snapshot_id": snapshot.snapshot_id,
        "digest": snapshot.digest,
        "resource_bundle_digest": snapshot.resource_bundle_digest,
        "model_group": _group_value(snapshot.model_group),
        "resources": _bundle_value(snapshot.resources),
    }


def _resource_ref_from_value(value: Mapping[str, object]) -> ModelResourceRef:
    return ModelResourceRef(
        resolver_id=str(value["resolver_id"]),
        resource_id=str(value["resource_id"]),
        revision=str(value["revision"]),
        capacity_owner_id=str(value["capacity_owner_id"]),
        coordinator_domain=str(value["coordinator_domain"]),
    )


def _bundle_from_value(value: object) -> ModelResourceBundle | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("stored model resource bundle must be an object")
    raw_routes = value.get("route_resources")
    if not isinstance(raw_routes, list):
        raise TypeError("stored route_resources must be an array")
    items: list[tuple[str, ModelResourceRef]] = []
    for item in raw_routes:
        if not isinstance(item, Mapping) or not isinstance(item.get("resource"), Mapping):
            raise TypeError("stored route resource must be an object")
        items.append((str(item["route_id"]), _resource_ref_from_value(item["resource"])))
    return ModelResourceBundle(
        resolver_id=str(value["resolver_id"]),
        route_resources=tuple(items),
        capacity_owner_id=str(value["capacity_owner_id"]),
        coordinator_domain=str(value["coordinator_domain"]),
    )


def _snapshot_from_value(value: Mapping[str, object]) -> ModelProfileSnapshot:
    raw_group = value.get("model_group")
    if not isinstance(raw_group, Mapping):
        raise TypeError("stored model group must be an object")
    routes_value = raw_group.get("routes")
    if not isinstance(routes_value, list):
        raise TypeError("stored model routes must be an array")
    routes = tuple(
        ModelRoute(str(item["route_id"]), provider=str(item["provider"]), model=str(item["model"]))
        for item in routes_value
        if isinstance(item, Mapping)
    )
    group = ModelGroupConfig(
        name=str(raw_group["name"]),
        routes=routes,
        fallback=FallbackPolicy(tuple(str(item) for item in raw_group.get("fallback", ()))),
        max_concurrency=raw_group.get("max_concurrency"),  # type: ignore[arg-type]
        capacity_key=raw_group.get("capacity_key"),  # type: ignore[arg-type]
    )
    return ModelProfileSnapshot(
        deployment_scope_id=str(value["deployment_scope_id"]),
        group_name=str(value["group_name"]),
        profile=str(value["profile"]),
        snapshot_id=str(value["snapshot_id"]),
        digest=str(value["digest"]),
        resource_bundle_digest=(
            None if value.get("resource_bundle_digest") is None else str(value["resource_bundle_digest"])
        ),
        model_group=group,
        resources=_bundle_from_value(value.get("resources")),
    )


@dataclass(frozen=True, slots=True)
class ModelAdmission:
    admission_id: str
    deployment_scope_id: str
    snapshots: tuple[tuple[str, ModelProfileSnapshot], ...]
    digest: str

    def snapshot(self, group_name: str) -> ModelProfileSnapshot:
        for candidate, snapshot in self.snapshots:
            if candidate == group_name:
                return snapshot
        raise ModelDeploymentUnavailableError(
            f"model group {group_name!r} is not pinned by this admission"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "admission_id": self.admission_id,
            "deployment_scope_id": self.deployment_scope_id,
            "snapshots": [
                {"group_name": name, "snapshot": _snapshot_value(snapshot)}
                for name, snapshot in self.snapshots
            ],
            "digest": self.digest,
        }


class ModelDeploymentStore(Protocol):
    namespace_id: str

    async def ensure_scope(self, namespace: str, binding_name: str, policy_digest: str) -> str: ...
    async def publish(self, snapshot: ModelProfileSnapshot) -> ModelProfileSnapshot: ...
    async def ensure_profile(self, snapshot: ModelProfileSnapshot, *, make_default: bool) -> ModelProfileSnapshot: ...
    async def set_default(self, scope_id: str, group_name: str, profile: str) -> None: ...
    async def default_profile(self, scope_id: str, group_name: str) -> str: ...
    async def current(self, scope_id: str, group_name: str, profile: str) -> ModelProfileSnapshot: ...
    async def list_profiles(self, scope_id: str, group_name: str) -> tuple[str, ...]: ...
    async def retire(self, scope_id: str, group_name: str, profile: str, replacement_default: str | None = None) -> None: ...
    async def admit(self, scope_id: str, requirements: tuple[str, ...], selections: Mapping[str, str | None], *, admission_id: str | None = None) -> ModelAdmission: ...
    async def get_admission(self, admission_id: str) -> ModelAdmission | None: ...
    async def release_admission(self, admission_id: str, *, recoverable: bool = False) -> None: ...
    async def close(self) -> None: ...


class InMemoryModelDeploymentStore:
    def __init__(self) -> None:
        self.namespace_id = "memory:" + str(uuid.uuid4())
        self._lock = asyncio.Lock()
        self._scopes: dict[tuple[str, str], tuple[str, str]] = {}
        self._profiles: dict[tuple[str, str, str], ModelProfileSnapshot] = {}
        self._defaults: dict[tuple[str, str], str] = {}
        self._retired: set[tuple[str, str, str]] = set()
        self._admissions: dict[str, ModelAdmission] = {}

    async def ensure_scope(self, namespace: str, binding_name: str, policy_digest: str) -> str:
        key = (namespace, binding_name)
        async with self._lock:
            current = self._scopes.get(key)
            if current is not None:
                scope_id, digest = current
                if digest != policy_digest:
                    raise ModelDeploymentConflictError(
                        "Binding scope already exists with a different policy"
                    )
                return scope_id
            scope_id = str(uuid.uuid4())
            self._scopes[key] = (scope_id, policy_digest)
            return scope_id

    async def publish(self, snapshot: ModelProfileSnapshot) -> ModelProfileSnapshot:
        key = (snapshot.deployment_scope_id, snapshot.group_name, snapshot.profile)
        async with self._lock:
            self._profiles[key] = snapshot
            self._retired.discard(key)
        return snapshot

    async def ensure_profile(
        self, snapshot: ModelProfileSnapshot, *, make_default: bool
    ) -> ModelProfileSnapshot:
        key = (snapshot.deployment_scope_id, snapshot.group_name, snapshot.profile)
        async with self._lock:
            current = self._profiles.get(key)
            if current is not None and current.digest == snapshot.digest and key not in self._retired:
                result = current
            else:
                self._profiles[key] = snapshot
                self._retired.discard(key)
                result = snapshot
            if make_default:
                self._defaults[(snapshot.deployment_scope_id, snapshot.group_name)] = snapshot.profile
            return result

    async def set_default(self, scope_id: str, group_name: str, profile: str) -> None:
        key = (scope_id, group_name, profile)
        async with self._lock:
            if key not in self._profiles or key in self._retired:
                raise ModelProfileSelectionError(f"unknown active profile {profile!r}")
            self._defaults[(scope_id, group_name)] = profile

    async def default_profile(self, scope_id: str, group_name: str) -> str:
        async with self._lock:
            try:
                return self._defaults[(scope_id, group_name)]
            except KeyError as exc:
                raise ModelDeploymentUnavailableError(
                    f"model group {group_name!r} has no default profile"
                ) from exc

    async def current(self, scope_id: str, group_name: str, profile: str) -> ModelProfileSnapshot:
        key = (scope_id, group_name, profile)
        async with self._lock:
            if key in self._retired:
                raise ModelProfileSelectionError(f"profile {profile!r} is retired")
            try:
                return self._profiles[key]
            except KeyError as exc:
                raise ModelProfileSelectionError(f"unknown profile {profile!r}") from exc

    async def list_profiles(self, scope_id: str, group_name: str) -> tuple[str, ...]:
        async with self._lock:
            return tuple(sorted(
                profile for scope, group, profile in self._profiles
                if scope == scope_id and group == group_name and (scope, group, profile) not in self._retired
            ))

    async def retire(self, scope_id: str, group_name: str, profile: str, replacement_default: str | None = None) -> None:
        key = (scope_id, group_name, profile)
        async with self._lock:
            if key not in self._profiles:
                raise ModelProfileSelectionError(f"unknown profile {profile!r}")
            default_key = (scope_id, group_name)
            if self._defaults.get(default_key) == profile:
                if replacement_default is None:
                    raise ModelDeploymentConflictError("cannot retire the default profile without a replacement")
                replacement_key = (scope_id, group_name, replacement_default)
                if replacement_key not in self._profiles or replacement_key in self._retired:
                    raise ModelProfileSelectionError("replacement default is not active")
                self._defaults[default_key] = replacement_default
            self._retired.add(key)

    async def admit(self, scope_id: str, requirements: tuple[str, ...], selections: Mapping[str, str | None], *, admission_id: str | None = None) -> ModelAdmission:
        identity = admission_id or str(uuid.uuid4())
        async with self._lock:
            existing = self._admissions.get(identity)
            if existing is not None:
                if existing.deployment_scope_id != scope_id:
                    raise ModelDeploymentConflictError(
                        "admission identity belongs to another deployment scope"
                    )
                return existing
            snapshots: list[tuple[str, ModelProfileSnapshot]] = []
            for group_name in requirements:
                profile = selections.get(group_name) or self._defaults.get((scope_id, group_name))
                if profile is None:
                    raise ModelDeploymentUnavailableError(
                        f"model group {group_name!r} has no selected or default profile"
                    )
                key = (scope_id, group_name, profile)
                if key in self._retired or key not in self._profiles:
                    raise ModelProfileSelectionError(
                        f"model group {group_name!r} profile {profile!r} is unavailable"
                    )
                snapshots.append((group_name, self._profiles[key]))
            payload = [
                {"group_name": name, "snapshot_id": snapshot.snapshot_id, "digest": snapshot.digest}
                for name, snapshot in snapshots
            ]
            admission = ModelAdmission(identity, scope_id, tuple(snapshots), _digest(payload))
            self._admissions[identity] = admission
            return admission

    async def get_admission(self, admission_id: str) -> ModelAdmission | None:
        async with self._lock:
            return self._admissions.get(admission_id)

    async def release_admission(self, admission_id: str, *, recoverable: bool = False) -> None:
        if recoverable:
            return
        async with self._lock:
            self._admissions.pop(admission_id, None)

    async def close(self) -> None:
        return None


class SQLiteModelDeploymentStore(InMemoryModelDeploymentStore):
    """SQLite-backed profile and admission store with an in-memory live cache."""

    def __init__(self, path: str | Path, *, namespace_id: str | None = None) -> None:
        super().__init__()
        self.path = str(path)
        if namespace_id is not None and (
            not isinstance(namespace_id, str) or not namespace_id
        ):
            raise ValueError("namespace_id must be non-empty when provided")
        self.namespace_id = namespace_id or (
            "sqlite:" + hashlib.sha256(str(Path(path).resolve()).encode()).hexdigest()
        )
        self._db: aiosqlite.Connection | None = None
        self._sqlite_write_lock = asyncio.Lock()

    async def open(self) -> Self:
        if self._db is not None:
            return self
        self._db = await aiosqlite.connect(self.path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS pygent_model_scopes(
                namespace TEXT NOT NULL, binding_name TEXT NOT NULL,
                scope_id TEXT NOT NULL UNIQUE, policy_digest TEXT NOT NULL,
                PRIMARY KEY(namespace,binding_name));
            CREATE TABLE IF NOT EXISTS pygent_model_profiles(
                scope_id TEXT NOT NULL, group_name TEXT NOT NULL, profile TEXT NOT NULL,
                snapshot_json TEXT NOT NULL, retired INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(scope_id,group_name,profile));
            CREATE TABLE IF NOT EXISTS pygent_model_defaults(
                scope_id TEXT NOT NULL, group_name TEXT NOT NULL, profile TEXT NOT NULL,
                PRIMARY KEY(scope_id,group_name));
            CREATE TABLE IF NOT EXISTS pygent_model_admissions(
                admission_id TEXT PRIMARY KEY, scope_id TEXT NOT NULL,
                admission_json TEXT NOT NULL, recoverable INTEGER NOT NULL DEFAULT 0);
            """
        )
        await self._db.commit()
        await self._load()
        return self

    async def _load(self) -> None:
        assert self._db is not None
        for namespace, binding_name, scope_id, policy_digest in await (await self._db.execute(
            "SELECT namespace,binding_name,scope_id,policy_digest FROM pygent_model_scopes"
        )).fetchall():
            self._scopes[(namespace, binding_name)] = (scope_id, policy_digest)
        for scope_id, group_name, profile, payload, retired in await (await self._db.execute(
            "SELECT scope_id,group_name,profile,snapshot_json,retired FROM pygent_model_profiles"
        )).fetchall():
            snapshot = _snapshot_from_value(json.loads(payload))
            key = (scope_id, group_name, profile)
            self._profiles[key] = snapshot
            if retired:
                self._retired.add(key)
        for scope_id, group_name, profile in await (await self._db.execute(
            "SELECT scope_id,group_name,profile FROM pygent_model_defaults"
        )).fetchall():
            self._defaults[(scope_id, group_name)] = profile
        for admission_id, payload in await (await self._db.execute(
            "SELECT admission_id,admission_json FROM pygent_model_admissions"
        )).fetchall():
            value = json.loads(payload)
            snapshots = tuple(
                (str(item["group_name"]), _snapshot_from_value(item["snapshot"]))
                for item in value["snapshots"]
            )
            self._admissions[admission_id] = ModelAdmission(
                admission_id=admission_id,
                deployment_scope_id=str(value["deployment_scope_id"]),
                snapshots=snapshots,
                digest=str(value["digest"]),
            )

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("SQLiteModelDeploymentStore is not open")
        return self._db

    @_sqlite_serialized
    async def ensure_scope(self, namespace: str, binding_name: str, policy_digest: str) -> str:
        db = self._connection()
        row = await (
            await db.execute(
                "SELECT scope_id,policy_digest FROM pygent_model_scopes "
                "WHERE namespace=? AND binding_name=?",
                (namespace, binding_name),
            )
        ).fetchone()
        if row is not None:
            if str(row[1]) != policy_digest:
                raise ModelDeploymentConflictError(
                    "Binding scope already exists with a different policy"
                )
            scope_id = str(row[0])
            self._scopes[(namespace, binding_name)] = (scope_id, policy_digest)
            return scope_id
        scope_id = str(uuid.uuid4())
        await db.execute(
            "INSERT OR IGNORE INTO pygent_model_scopes(namespace,binding_name,scope_id,policy_digest) VALUES(?,?,?,?)",
            (namespace, binding_name, scope_id, policy_digest),
        )
        await db.commit()
        row = await (
            await db.execute(
                "SELECT scope_id,policy_digest FROM pygent_model_scopes "
                "WHERE namespace=? AND binding_name=?",
                (namespace, binding_name),
            )
        ).fetchone()
        assert row is not None
        if str(row[1]) != policy_digest:
            raise ModelDeploymentConflictError(
                "Binding scope already exists with a different policy"
            )
        resolved = str(row[0])
        self._scopes[(namespace, binding_name)] = (resolved, policy_digest)
        return resolved

    @_sqlite_serialized
    async def publish(self, snapshot: ModelProfileSnapshot) -> ModelProfileSnapshot:
        result = await super().publish(snapshot)
        db = self._connection()
        await db.execute(
            "INSERT INTO pygent_model_profiles(scope_id,group_name,profile,snapshot_json,retired) VALUES(?,?,?,?,0) "
            "ON CONFLICT(scope_id,group_name,profile) DO UPDATE SET snapshot_json=excluded.snapshot_json,retired=0",
            (snapshot.deployment_scope_id, snapshot.group_name, snapshot.profile, _canonical(_snapshot_value(snapshot))),
        )
        await db.commit()
        return result

    @_sqlite_serialized
    async def ensure_profile(
        self, snapshot: ModelProfileSnapshot, *, make_default: bool
    ) -> ModelProfileSnapshot:
        db = self._connection()
        key = (snapshot.deployment_scope_id, snapshot.group_name, snapshot.profile)
        await db.execute("BEGIN IMMEDIATE")
        try:
            row = await (
                await db.execute(
                    "SELECT snapshot_json,retired FROM pygent_model_profiles "
                    "WHERE scope_id=? AND group_name=? AND profile=?",
                    key,
                )
            ).fetchone()
            result = snapshot
            if row is not None and not row[1]:
                current = _snapshot_from_value(json.loads(row[0]))
                if current.digest == snapshot.digest:
                    result = current
            if result is snapshot:
                await db.execute(
                    "INSERT INTO pygent_model_profiles(scope_id,group_name,profile,snapshot_json,retired) "
                    "VALUES(?,?,?,?,0) ON CONFLICT(scope_id,group_name,profile) DO UPDATE SET "
                    "snapshot_json=excluded.snapshot_json,retired=0",
                    (*key, _canonical(_snapshot_value(snapshot))),
                )
            if make_default:
                await db.execute(
                    "INSERT INTO pygent_model_defaults(scope_id,group_name,profile) VALUES(?,?,?) "
                    "ON CONFLICT(scope_id,group_name) DO UPDATE SET profile=excluded.profile",
                    key,
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self._profiles[key] = result
        self._retired.discard(key)
        if make_default:
            self._defaults[(key[0], key[1])] = key[2]
        return result

    @_sqlite_serialized
    async def set_default(self, scope_id: str, group_name: str, profile: str) -> None:
        db = self._connection()
        await db.execute("BEGIN IMMEDIATE")
        try:
            row = await (
                await db.execute(
                    "SELECT retired FROM pygent_model_profiles WHERE scope_id=? AND group_name=? AND profile=?",
                    (scope_id, group_name, profile),
                )
            ).fetchone()
            if row is None or row[0]:
                raise ModelProfileSelectionError(
                    f"unknown active profile {profile!r}"
                )
            await db.execute(
                "INSERT INTO pygent_model_defaults(scope_id,group_name,profile) VALUES(?,?,?) "
                "ON CONFLICT(scope_id,group_name) DO UPDATE SET profile=excluded.profile",
                (scope_id, group_name, profile),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self._defaults[(scope_id, group_name)] = profile

    async def current(self, scope_id: str, group_name: str, profile: str) -> ModelProfileSnapshot:
        db = self._connection()
        row = await (
            await db.execute(
                "SELECT snapshot_json,retired FROM pygent_model_profiles WHERE scope_id=? AND group_name=? AND profile=?",
                (scope_id, group_name, profile),
            )
        ).fetchone()
        if row is None:
            raise ModelProfileSelectionError(f"unknown profile {profile!r}")
        if row[1]:
            raise ModelProfileSelectionError(f"profile {profile!r} is retired")
        snapshot = _snapshot_from_value(json.loads(row[0]))
        self._profiles[(scope_id, group_name, profile)] = snapshot
        return snapshot

    async def default_profile(self, scope_id: str, group_name: str) -> str:
        row = await (
            await self._connection().execute(
                "SELECT profile FROM pygent_model_defaults WHERE scope_id=? AND group_name=?",
                (scope_id, group_name),
            )
        ).fetchone()
        if row is None:
            raise ModelDeploymentUnavailableError(
                f"model group {group_name!r} has no default profile"
            )
        return str(row[0])

    async def list_profiles(self, scope_id: str, group_name: str) -> tuple[str, ...]:
        db = self._connection()
        rows = await (
            await db.execute(
                "SELECT profile FROM pygent_model_profiles WHERE scope_id=? AND group_name=? AND retired=0 ORDER BY profile",
                (scope_id, group_name),
            )
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    @_sqlite_serialized
    async def retire(self, scope_id: str, group_name: str, profile: str, replacement_default: str | None = None) -> None:
        db = self._connection()
        await db.execute("BEGIN IMMEDIATE")
        try:
            profile_row = await (
                await db.execute(
                    "SELECT retired FROM pygent_model_profiles "
                    "WHERE scope_id=? AND group_name=? AND profile=?",
                    (scope_id, group_name, profile),
                )
            ).fetchone()
            if profile_row is None:
                raise ModelProfileSelectionError(f"unknown profile {profile!r}")
            current_default = await (
                await db.execute(
                    "SELECT profile FROM pygent_model_defaults WHERE scope_id=? AND group_name=?",
                    (scope_id, group_name),
                )
            ).fetchone()
            replacing_default = (
                current_default is not None and current_default[0] == profile
            )
            if replacing_default:
                if replacement_default is None:
                    raise ModelDeploymentConflictError(
                        "cannot retire the default profile without a replacement"
                    )
                replacement_row = await (
                    await db.execute(
                        "SELECT retired FROM pygent_model_profiles "
                        "WHERE scope_id=? AND group_name=? AND profile=?",
                        (scope_id, group_name, replacement_default),
                    )
                ).fetchone()
                if replacement_row is None or replacement_row[0]:
                    raise ModelProfileSelectionError(
                        "replacement default is not active"
                    )
            await db.execute(
                "UPDATE pygent_model_profiles SET retired=1 "
                "WHERE scope_id=? AND group_name=? AND profile=?",
                (scope_id, group_name, profile),
            )
            if replacing_default and replacement_default is not None:
                await db.execute(
                    "INSERT INTO pygent_model_defaults(scope_id,group_name,profile) VALUES(?,?,?) "
                    "ON CONFLICT(scope_id,group_name) DO UPDATE SET profile=excluded.profile",
                    (scope_id, group_name, replacement_default),
                )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self._retired.add((scope_id, group_name, profile))
        if replacing_default and replacement_default is not None:
            self._defaults[(scope_id, group_name)] = replacement_default

    @_sqlite_serialized
    async def admit(self, scope_id: str, requirements: tuple[str, ...], selections: Mapping[str, str | None], *, admission_id: str | None = None) -> ModelAdmission:
        db = self._connection()
        identity = admission_id or str(uuid.uuid4())
        await db.execute("BEGIN IMMEDIATE")
        existing = await (
            await db.execute(
                "SELECT admission_json FROM pygent_model_admissions WHERE admission_id=?",
                (identity,),
            )
        ).fetchone()
        if existing is not None:
            await db.commit()
            value = json.loads(existing[0])
            admission = ModelAdmission(
                admission_id=identity,
                deployment_scope_id=str(value["deployment_scope_id"]),
                snapshots=tuple(
                    (str(item["group_name"]), _snapshot_from_value(item["snapshot"]))
                    for item in value["snapshots"]
                ),
                digest=str(value["digest"]),
            )
            if admission.deployment_scope_id != scope_id:
                raise ModelDeploymentConflictError(
                    "admission identity belongs to another deployment scope"
                )
            self._admissions[identity] = admission
            return admission
        snapshots: list[tuple[str, ModelProfileSnapshot]] = []
        try:
            for group_name in requirements:
                profile = selections.get(group_name)
                if profile is None:
                    row = await (
                        await db.execute(
                            "SELECT profile FROM pygent_model_defaults WHERE scope_id=? AND group_name=?",
                            (scope_id, group_name),
                        )
                    ).fetchone()
                    profile = None if row is None else str(row[0])
                if profile is None:
                    raise ModelDeploymentUnavailableError(
                        f"model group {group_name!r} has no selected or default profile"
                    )
                row = await (
                    await db.execute(
                        "SELECT snapshot_json,retired FROM pygent_model_profiles WHERE scope_id=? AND group_name=? AND profile=?",
                        (scope_id, group_name, profile),
                    )
                ).fetchone()
                if row is None or row[1]:
                    raise ModelProfileSelectionError(
                        f"model group {group_name!r} profile {profile!r} is unavailable"
                    )
                snapshots.append((group_name, _snapshot_from_value(json.loads(row[0]))))
            digest = _digest([
                {"group_name": name, "snapshot_id": item.snapshot_id, "digest": item.digest}
                for name, item in snapshots
            ])
            admission = ModelAdmission(identity, scope_id, tuple(snapshots), digest)
            payload = admission.to_dict()
            await db.execute(
                "INSERT INTO pygent_model_admissions(admission_id,scope_id,admission_json) VALUES(?,?,?)",
                (identity, scope_id, _canonical(payload)),
            )
            await db.commit()
        except BaseException:
            await db.rollback()
            raise
        self._admissions[identity] = admission
        return admission

    async def get_admission(self, admission_id: str) -> ModelAdmission | None:
        db = self._connection()
        row = await (
            await db.execute(
                "SELECT admission_json FROM pygent_model_admissions WHERE admission_id=?",
                (admission_id,),
            )
        ).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        return ModelAdmission(
            admission_id=admission_id,
            deployment_scope_id=str(value["deployment_scope_id"]),
            snapshots=tuple(
                (str(item["group_name"]), _snapshot_from_value(item["snapshot"]))
                for item in value["snapshots"]
            ),
            digest=str(value["digest"]),
        )

    @_sqlite_serialized
    async def release_admission(self, admission_id: str, *, recoverable: bool = False) -> None:
        await super().release_admission(admission_id, recoverable=recoverable)
        db = self._connection()
        if recoverable:
            await db.execute(
                "UPDATE pygent_model_admissions SET recoverable=1 WHERE admission_id=?",
                (admission_id,),
            )
        else:
            await db.execute("DELETE FROM pygent_model_admissions WHERE admission_id=?", (admission_id,))
        await db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None


def build_snapshot(
    *,
    scope_id: str,
    requirement: ModelGroupConfig,
    profile: str,
    routes: tuple[ModelRoute, ...],
    fallback: FallbackPolicy,
    resources: ModelResourceBundle | None,
) -> ModelProfileSnapshot:
    if not requirement.is_deferred:
        raise ValueError("dynamic profiles require a deferred ModelGroupConfig")
    if not isinstance(profile, str) or not profile:
        raise ValueError("profile must be a non-empty string")
    group = ModelGroupConfig(
        name=requirement.name,
        routes=routes,
        fallback=fallback,
        max_concurrency=requirement.max_concurrency,
        capacity_key=requirement.capacity_key,
    )
    if resources is not None:
        route_ids = {route.route_id for route in routes}
        resource_ids = {route_id for route_id, _ in resources.route_resources}
        if route_ids != resource_ids:
            raise ValueError("resource bundle must map every route exactly once")
    portable = {
        "scope_id": scope_id,
        "group": _group_value(group),
        "profile": profile,
        "resources": _bundle_value(resources),
    }
    digest = _digest(portable)
    return ModelProfileSnapshot(
        deployment_scope_id=scope_id,
        group_name=requirement.name,
        profile=profile,
        snapshot_id=str(uuid.uuid4()),
        digest=digest,
        resource_bundle_digest=None if resources is None else _digest(resources.to_dict()),
        model_group=group,
        resources=resources,
    )


__all__ = [
    "InMemoryModelDeploymentStore",
    "ModelAdmission",
    "ModelDeploymentStore",
    "SQLiteModelDeploymentStore",
    "build_snapshot",
]
