from __future__ import annotations

from hashlib import sha256
from pathlib import Path

_ROUTE_IDS = Path(__file__).with_name("az_conformance") / "route_ids.txt"


def test_frozen_route_id_snapshot_is_exact() -> None:
    assert _ROUTE_IDS.is_file(), "frozen AZ route ID snapshot is missing"
    text = _ROUTE_IDS.read_text(encoding="utf-8")
    ids = text.splitlines()
    payload = ("\n".join(ids) + "\n").encode()

    assert text.endswith("\n")
    assert ids == sorted(ids)
    assert len(ids) == len(set(ids)) == 211
    assert sha256(payload).hexdigest() == (
        "7398801a0d480abc1b45d64d87e9c8eac508404f53dffc37d202c222f264330e"
    )
