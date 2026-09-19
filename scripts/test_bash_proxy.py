from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from pygent.tool.standard._bash import BashTools


async def main() -> None:
    tools = BashTools(workspace_root=Path("."))
    # Show detected system proxy env (from our helper)
    # Note: this is not necessarily what will be injected; injection only fills absent keys.
    try:
        detect = tools._detect_system_proxy_env(os.environ.copy())  # type: ignore[attr-defined]
    except AttributeError:
        detect = {"error": "_detect_system_proxy_env not found in BashTools"}
    print("detected_system_proxy_env=", json.dumps(detect, ensure_ascii=False))

    # Run a bash command via our BashTools to print effective proxy env and try Google
    command = (
        "env | sort | grep -iE '^(http|https|all|no)_proxy=' || true; "
        "echo; echo '== curl google.com (HEAD) =='; "
        "curl -I -sS -m 15 --connect-timeout 5 https://www.google.com -o /dev/null -w 'status=%{http_code}\n' || echo 'curl failed'"
    )
    out = await tools.bash(command)
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
