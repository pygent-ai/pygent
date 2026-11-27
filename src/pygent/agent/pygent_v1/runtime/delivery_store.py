from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import sqlite3
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager, suppress
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

from pygent.agent.pygent_v1.core.io import plain_object
from pygent.agent.pygent_v1.runtime.realtime.envelope import ExchangeEnvelope

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


_P = ParamSpec("_P")
_T = TypeVar("_T")


def session_delivery_path(sessions_root: str | Path) -> Path:
    root = Path(sessions_root).resolve()
    return root.with_name(f"{root.name}-deliveries.sqlite3")


class DeliveryStore:
    """Workspace delivery journal. Claims expire after their owner stops renewing."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._initialized = False
        self._init_lock = threading.Lock()

    def _initialize(self) -> None:
        with self._init_lock:
            if not self._initialized:
                self._create_schema()
                self._initialized = True

    def _create_schema(self) -> None:
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS deliveries (
                    sequence INTEGER PRIMARY KEY,
                    lane TEXT NOT NULL,
                    key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    target TEXT,
                    receipt TEXT,
                    owner TEXT,
                    expires REAL NOT NULL DEFAULT 0,
                    due REAL NOT NULL DEFAULT 0,
                    UNIQUE (lane, key)
                )
                """)
            db.execute("""
                CREATE INDEX IF NOT EXISTS deliveries_pending
                ON deliveries(lane, sequence) WHERE receipt IS NULL
                """)

    async def run(
        self, operation: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> _T:
        """Run one complete SQLite operation in a worker thread."""
        worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            # Wait for the transaction before releasing claims or scheduling
            # a retry. Cancellation cannot stop sqlite3 on the worker.
            while not worker.done():
                with suppress(asyncio.CancelledError):
                    await asyncio.wait((worker,))
            if not worker.cancelled():
                worker.exception()
            raise

    @asynccontextmanager
    async def session_execution(self, session_id: str) -> AsyncIterator[None]:
        lock_dir = self.path.with_suffix(".locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(session_id.encode()).hexdigest()
        with (lock_dir / name).open("a+b") as lock:
            lock.seek(0)
            while True:
                try:
                    if sys.platform == "win32":
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError as exc:
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                        raise
                    await asyncio.sleep(0.05)
                else:
                    break
            try:
                yield
            finally:
                if sys.platform == "win32":
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def prior_turn_pending(self, lane: str, key: str) -> bool:
        with self.transaction() as db:
            return (
                db.execute(
                    """SELECT 1 FROM deliveries AS earlier
                JOIN deliveries AS current ON current.lane=earlier.lane
                WHERE current.lane=? AND current.key=?
                    AND earlier.sequence<current.sequence AND earlier.receipt IS NULL
                    AND (earlier.target IS NULL OR earlier.target LIKE 'turn:%')
                LIMIT 1""",
                    (lane, key),
                ).fetchone()
                is not None
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if not self._initialized:
            self._initialize()
        with self._connect() as db:
            yield db

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def put(
        db: sqlite3.Connection, lane: str, key: str, payload: dict[str, Any]
    ) -> None:
        db.execute(
            "INSERT OR IGNORE INTO deliveries(lane,key,payload) VALUES(?,?,?)",
            (lane, key, json.dumps(payload, ensure_ascii=False)),
        )

    def record_exchange(
        self, db: sqlite3.Connection, envelope: ExchangeEnvelope
    ) -> sqlite3.Row:
        lane = f"exchange:{envelope.session_id}"
        self.put(
            db,
            lane,
            envelope.exchange_id,
            {
                "content": envelope.content,
                "attachments": envelope.attachments,
            },
        )
        if envelope.source_input_ids:
            owner = envelope.source_input_ids[-1]
            for input_id in envelope.source_input_ids:
                self.set_receipt(
                    db,
                    f"user:{envelope.session_id}",
                    input_id,
                    {
                        "mode": "interrupt",
                        "reply": plain_object(envelope.reply)
                        if envelope.reply is not None
                        else None,
                        "reply_owner": owner,
                    },
                )
        return self.read(db, lane, envelope.exchange_id)

    @staticmethod
    def find(db: sqlite3.Connection, lane: str, key: str) -> sqlite3.Row | None:
        return db.execute(
            "SELECT * FROM deliveries WHERE lane=? AND key=?", (lane, key)
        ).fetchone()

    @classmethod
    def read(cls, db: sqlite3.Connection, lane: str, key: str) -> sqlite3.Row:
        row = cls.find(db, lane, key)
        assert row is not None
        return row

    def get(self, lane: str, key: str) -> sqlite3.Row:
        with self.transaction() as db:
            return self.read(db, lane, key)

    def renew(self, lane: str, key: str, owner: str, ttl: float) -> bool:
        with self.transaction() as db:
            result = db.execute(
                """UPDATE deliveries SET expires=?
                WHERE lane=? AND key=? AND owner=? AND receipt IS NULL""",
                (time.time() + ttl, lane, key, owner),
            )
            return bool(result.rowcount)

    def claim(self, lane: str, key: str, owner: str, ttl: float) -> sqlite3.Row | None:
        now = time.time()
        with self.transaction() as db:
            result = db.execute(
                """UPDATE deliveries SET owner=?, expires=?
                WHERE lane=? AND key=? AND receipt IS NULL AND due<=?
                AND expires<=?""",
                (owner, now + ttl, lane, key, now, now),
            )
            return self.read(db, lane, key) if result.rowcount else None

    def retry(self, lane: str, key: str, owner: str, *, delay: float = 0) -> None:
        with self.transaction() as db:
            db.execute(
                """UPDATE deliveries SET owner=NULL,expires=0,due=?
                WHERE lane=? AND key=? AND owner=? AND receipt IS NULL""",
                (time.time() + delay, lane, key, owner),
            )

    @staticmethod
    def set_receipt(
        db: sqlite3.Connection, lane: str, key: str, receipt: dict[str, Any]
    ) -> None:
        db.execute(
            """UPDATE deliveries SET receipt=?,owner=NULL,expires=0
            WHERE lane=? AND key=? AND receipt IS NULL""",
            (json.dumps(receipt, ensure_ascii=False), lane, key),
        )

    def finish(self, lane: str, key: str, receipt: dict[str, Any]) -> None:
        with self.transaction() as db:
            self.set_receipt(db, lane, key, receipt)

    def reject_input(self, lane: str, input_id: str, error: ValueError) -> None:
        self.finish(lane, input_id, {"mode": "rejected", "error": str(error)})

    def set_target(self, lane: str, key: str, target: str | None) -> None:
        with self.transaction() as db:
            db.execute(
                "UPDATE deliveries SET target=? WHERE lane=? AND key=? AND receipt IS NULL",
                (target, lane, key),
            )

    def pending(
        self, lane: str, *, prefix: bool = False, available: bool = False
    ) -> list[sqlite3.Row]:
        match = "lane LIKE ?" if prefix else "lane=?"
        if prefix:
            lane += "%"
        filters = " AND expires<=? AND due<=?" if available else ""
        now = time.time()
        params = (lane, now, now) if available else (lane,)
        with self.transaction() as db:
            return db.execute(
                f"SELECT * FROM deliveries WHERE {match} AND receipt IS NULL{filters} ORDER BY sequence",
                params,
            ).fetchall()
