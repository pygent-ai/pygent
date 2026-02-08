from typing import Any, Dict, List, Optional

from pygent.module import PygentModule
from pygent.common import PygentDict
from pygent.module.tool import BaseTool


class ToolManager(PygentModule):
    """工具管理器"""

    tools: PygentDict  # 存储所有工具
    tool_categories: PygentDict  # 按类别组织工具
    mcp_clients: PygentDict  # server_id -> MCP client (stdio/sse) for reference

    def __init__(self):
        super().__init__()
        self.tools = PygentDict({})
        self.tool_categories = PygentDict({})
        self.mcp_clients = PygentDict({})

    def register_tools(self, tools: List[BaseTool]) -> None:
        """Register a list of tools (e.g. from a module's get_tools())."""
        for tool in tools:
            self.register_tool(tool)

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具"""
        tool_name = tool.metadata.data["name"]

        if tool_name in self.tools.data:
            print(f"警告: 工具 '{tool_name}' 已存在，将被替换")

        self.tools.data[tool_name] = tool

        # 按类别组织（使用枚举的 value 以便 JSON 序列化）
        category = tool.metadata.data.get("category")
        category_key = getattr(category, "value", None) if category is not None else None
        category_key = category_key or "utility"
        if category_key not in self.tool_categories.data:
            self.tool_categories.data[category_key] = []

        if tool_name not in self.tool_categories.data[category_key]:
            self.tool_categories.data[category_key].append(tool_name)

        print(f"工具 '{tool_name}' 已注册")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self.tools.data.get(name)

    def call_tool(self, name: str, *args, **kwargs) -> Dict[str, Any]:
        """调用工具"""
        tool = self.get_tool(name)
        if not tool:
            return {
                "success": False,
                "error": f"工具 '{name}' 未找到"
            }

        return tool(*args, **kwargs)

    def get_all_schemas(self) -> Dict[str, Any]:
        """获取所有工具的模式"""
        return {
            "tools": {
                name: tool.get_schema()
                for name, tool in self.tools.data.items()
            },
            "categories": self.tool_categories.data
        }

    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """获取所有工具的OpenAI Function格式"""
        return [
            tool.to_openai_function()
            for tool in self.tools.data.values()
        ]

    def add_mcp_server_stdio(
        self,
        server_id: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        tool_name_prefix: Optional[str] = None,
    ) -> List[BaseTool]:
        """
        Connect to an MCP server over stdio (spawn process), list its tools,
        wrap each as a BaseTool (MCPToolAdapter), and register them.

        Args:
            server_id: Identifier for this MCP server (stored in mcp_clients).
            command: Executable to run (e.g. "npx", "uvx").
            args: Command-line arguments (e.g. ["-y", "mcp-server-foo"]).
            env: Optional environment variables for the process.
            cwd: Optional working directory for the server process.
            tool_name_prefix: If set, registered tool names become "{prefix}_{tool_name}" to avoid collisions.

        Returns:
            List of registered MCPToolAdapter instances.
        """
        from .mcp import StdioMCPClient, MCPToolAdapter
        client = StdioMCPClient(
            server_id=server_id,
            command=command,
            args=args,
            env=env,
            cwd=cwd,
        )
        self.mcp_clients.data[server_id] = client
        return self._register_mcp_tools(client, server_id, tool_name_prefix, MCPToolAdapter)

    def add_mcp_server_sse(
        self,
        server_id: str,
        url: str,
        headers: Optional[Dict[str, Any]] = None,
        tool_name_prefix: Optional[str] = None,
    ) -> List[BaseTool]:
        """
        Connect to an MCP server over SSE (HTTP), list its tools,
        wrap each as a BaseTool (MCPToolAdapter), and register them.

        Args:
            server_id: Identifier for this MCP server (stored in mcp_clients).
            url: SSE endpoint URL (e.g. "http://localhost:8000/sse").
            headers: Optional HTTP headers.
            tool_name_prefix: If set, registered tool names become "{prefix}_{tool_name}" to avoid collisions.

        Returns:
            List of registered MCPToolAdapter instances.
        """
        from .mcp import SSEMCPClient, MCPToolAdapter
        client = SSEMCPClient(
            server_id=server_id,
            url=url,
            headers=headers,
        )
        self.mcp_clients.data[server_id] = client
        return self._register_mcp_tools(client, server_id, tool_name_prefix, MCPToolAdapter)

    def _register_mcp_tools(
        self,
        client: Any,
        server_id: str,
        tool_name_prefix: Optional[str],
        adapter_class: type,
    ) -> List[BaseTool]:
        """List tools from MCP client and register each as MCPToolAdapter."""
        tools_list = client.list_tools()
        registered = []
        prefix = tool_name_prefix if tool_name_prefix is not None else None

        for mcp_tool in tools_list:
            name = getattr(mcp_tool, "name", None)
            if not name:
                continue
            description = getattr(mcp_tool, "description", None) or f"MCP tool: {name}"
            input_schema = getattr(mcp_tool, "inputSchema", None)
            if not input_schema:
                input_schema = {"properties": {}, "required": []}

            adapter = adapter_class(
                client=client,
                tool_name=name,
                tool_description=description or "",
                input_schema=input_schema,
                server_id=server_id,
            )
            register_name = f"{prefix}_{name}" if prefix else name
            # MCPToolAdapter.metadata.data["name"] is used by register_tool; override for registration name
            if register_name != name:
                adapter.metadata.data["name"] = register_name
            self.register_tool(adapter)
            registered.append(adapter)
        return registered
