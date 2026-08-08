from __future__ import annotations

from collections.abc import Callable

from pygent import (
    AIMessage,
    Context,
    ToolAuthorizationDecision,
    ToolCall,
    ToolKit,
)
from pygent.tool import ToolResult


def allow(request, _context):
    return ToolAuthorizationDecision(
        call_id=request.call.call_id,
        allowed=True,
        reason_code="test_allowed",
    )


async def invoke_tool(
    handler: Callable[..., object],
    arguments: dict[str, object],
    *,
    call_id: str = "standard-tool-test",
) -> ToolResult:
    toolkit = ToolKit(handler)
    layer = toolkit.local_layer(authorization_adapter=allow)
    context = toolkit.make_visible_in(Context())
    message, returned_context = await layer.invoke(
        AIMessage(
            tool_calls=(
                ToolCall(
                    call_id=call_id,
                    name=toolkit.definitions[0].name,
                    arguments=arguments,
                ),
            )
        ),
        context,
    )
    assert returned_context is context
    return message.results[0]


async def succeeded(handler: Callable[..., object], **arguments: object) -> object:
    result = await invoke_tool(handler, arguments)
    assert result.status == "succeeded", result
    return result.output
