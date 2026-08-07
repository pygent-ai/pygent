import argparse

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - older MCP SDK
    from mcp.server.fastmcp import FastMCP as MCPServer

server = MCPServer("pygent-test")


@server.tool()
def double(value: int) -> int:
    """Double an integer."""

    return value * 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sse", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.sse:
        server.run(transport="sse", host="127.0.0.1", port=args.port)
    else:
        server.run(transport="stdio")
