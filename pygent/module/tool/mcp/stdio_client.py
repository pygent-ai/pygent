"""
MCP client for stdio transport: connects to a server by spawning a process.
"""

import io
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pygent.common import PygentString, PygentDict, PygentList

from .base import BaseMCPClient

from mcp.client.stdio import StdioServerParameters, stdio_client


def _safe_errlog() -> Tuple[Any, bool]:
    """
    Return (errlog, should_close). Use an errlog that has fileno() so subprocess
    creation works on Windows under IDEs (e.g. PyCharm) where sys.stderr may not.
    """
    try:
        if hasattr(sys.stderr, "fileno") and callable(getattr(sys.stderr, "fileno")):
            sys.stderr.fileno()
            return sys.stderr, False
    except (ValueError, OSError, io.UnsupportedOperation):
        pass
    return open(os.devnull, "w"), True


class StdioMCPClient(BaseMCPClient):
    """
    MCP client for stdio transport.
    Uses PygentData: server_id, command, args, env.
    """

    command: PygentString
    args: PygentList
    env: PygentDict

    def __init__(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[Union[str, Path]] = None,
        **kwargs: Any,
    ):
        super().__init__(server_id, **kwargs)
        self.command = PygentString(command)
        self.args = PygentList(list(args) if args else [])
        self.env = PygentDict(dict(env) if env else {})
        self._cwd = Path(cwd) if cwd else None

    def _get_server_parameters(self) -> Any:
        if StdioServerParameters is None:
            raise RuntimeError("MCP stdio client not available (pip install mcp)")
        return StdioServerParameters(
            command=self.command.data,
            args=list(self.args.data),
            env=self.env.data if self.env.data else None,
            cwd=self._cwd,
        )

    async def _connect_and_run(
        self,
        list_tools_only: bool = False,
        call_tool_name: Optional[str] = None,
        call_tool_arguments: Optional[Dict[str, Any]] = None,
    ) -> Any:
        params = self._get_server_parameters()
        errlog, should_close_errlog = _safe_errlog()
        try:
            async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
                return await self._session_flow(
                    read_stream,
                    write_stream,
                    list_tools_only=list_tools_only,
                    call_tool_name=call_tool_name,
                    call_tool_arguments=call_tool_arguments,
                )
        finally:
            if should_close_errlog:
                errlog.close()
