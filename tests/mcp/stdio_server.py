from mcp.server.mcpserver import MCPServer

server = MCPServer("pygent-test")


@server.tool()
def double(value: int) -> int:
    """Double an integer."""

    return value * 2


if __name__ == "__main__":
    server.run(transport="stdio")
