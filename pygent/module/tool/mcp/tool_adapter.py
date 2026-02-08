"""
Adapter that wraps an MCP server tool as a BaseTool instance.
"""

from typing import Any, Dict, List, Optional

from pygent.module.tool import BaseTool, ToolMetadata, ToolCategory
from pygent.common import PygentDict, PygentBool, PygentInt, PygentString

from .base import BaseMCPClient


def _json_schema_to_parameters(input_schema: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Convert MCP/JSON Schema inputSchema to BaseTool parameters dict.
    inputSchema has optional 'properties' and 'required'.
    """
    parameters: Dict[str, Dict[str, Any]] = {}
    props = input_schema.get("properties") or {}
    required_list = input_schema.get("required") or []

    for param_name, prop in props.items():
        if not isinstance(prop, dict):
            continue
        # JSON Schema type -> our type string
        js_type = prop.get("type", "string")
        type_map = {
            "string": "string",
            "integer": "integer",
            "number": "number",
            "boolean": "boolean",
            "array": "array",
            "object": "object",
            "null": "null",
        }
        param_type = type_map.get(js_type, "string")
        required = param_name in required_list
        default = prop.get("default")
        enum = prop.get("enum")
        description = prop.get("description") or ""

        param_def = {
            "name": param_name,
            "type": param_type,
            "description": description,
            "required": required,
            "default": default,
            "enum": enum,
            "min_value": prop.get("minimum"),
            "max_value": prop.get("maximum"),
            "pattern": prop.get("pattern"),
        }
        parameters[param_name] = param_def

    return parameters


class MCPToolAdapter(BaseTool):
    """
    Wraps a single MCP server tool as a BaseTool instance.
    Uses the given MCP client to call the tool; parameters come from the tool's inputSchema.
    """

    def __init__(
        self,
        client: BaseMCPClient,
        tool_name: str,
        tool_description: str,
        input_schema: Dict[str, Any],
        server_id: Optional[str] = None,
        version: str = "1.0.0",
    ):
        super().__init__(
            name=tool_name,
            description=tool_description or f"MCP tool: {tool_name}",
            version=version,
            category=ToolCategory.CUSTOM,
        )
        # Override parameters from MCP tool inputSchema (BaseTool discovers from forward, which uses **kwargs)
        self.parameters.data.update(_json_schema_to_parameters(input_schema))

        self._mcp_client = client
        self._tool_name = tool_name
        self._server_id = server_id or getattr(client, "server_id", None)
        if hasattr(self._server_id, "data"):
            self._server_id = self._server_id.data

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the MCP tool via the client."""
        result = self._mcp_client.call_tool(self._tool_name, arguments=kwargs or None)

        # Return content suitable for agent consumption
        if hasattr(result, "isError") and result.isError:
            err = getattr(result, "error", result)
            raise RuntimeError(f"MCP tool error: {err}")

        if hasattr(result, "structuredContent") and result.structuredContent is not None:
            return result.structuredContent
        if hasattr(result, "content") and result.content:
            # Content is list of ContentBlock; often text in first block
            parts = []
            for block in result.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(block["text"])
            return "\n".join(parts) if parts else None
        return None
