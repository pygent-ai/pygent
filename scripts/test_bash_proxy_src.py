from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path('src').resolve()))
from pygent.tool.standard._bash import BashTools  # noqa: E402


async def main() -> None:
    tools = BashTools(workspace_root=Path("."))
    detect = tools._detect_system_proxy_env(os.environ.copy())
    print("detected_system_proxy_env=", json.dumps(detect, ensure_ascii=False))
    out = await tools.bash(
        "env | sort | grep -iE '^(http|https|all|no)_proxy=' || true; echo;"
        "echo '== curl google.com (HEAD) ==';"
        "curl -I -sS -m 15 --connect-timeout 5 https://www.google.com -o /dev/null -w 'status=%{http_code}\n' || echo 'curl failed'"
    )
    print(out)


if __name__ == "__main__":
    asyncio.run(main())
