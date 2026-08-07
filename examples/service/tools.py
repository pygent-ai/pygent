"""Application-owned tool declarations using the public Pygent SDK."""

from pygent import (
    Context,
    IdempotencyPolicy,
    Module,
    ToolAuthorizationDecision,
    ToolAuthorizationRequest,
    ToolCallLayer,
    ToolDefinition,
    ToolSideEffect,
    ToolSpec,
)
from pygent.tool import ExecutorRegistry, ToolTaskManager


class WeatherAuthorization(Module[ToolAuthorizationRequest, ToolAuthorizationDecision]):
    """Application-owned authorization and lifecycle decision."""

    async def forward(
        self,
        request: ToolAuthorizationRequest,
        context: Context,
    ) -> tuple[ToolAuthorizationDecision, Context]:
        allowed = "weather:read" in request.permissions
        return (
            ToolAuthorizationDecision(
                call_id=request.call.call_id,
                allowed=allowed,
                reason_code="allowed" if allowed else "missing_permission",
                lifecycle="sync",
            ),
            context,
        )


def build_tool_layer(
    *,
    executor_registry: ExecutorRegistry | None = None,
    task_manager: ToolTaskManager | None = None,
) -> ToolCallLayer:
    weather = ToolSpec(
        tool_id="weather.lookup",
        version="1.0.0",
        definition=ToolDefinition(
            name="weather.lookup",
            description="查询城市天气",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        side_effect=ToolSideEffect.READ,
        idempotency=IdempotencyPolicy.INHERENT,
        timeout=10.0,
        resource_key="weather-api",
        required_permissions=("weather:read",),
    )
    return ToolCallLayer(
        tools=(weather,),
        authorization=WeatherAuthorization(),
        executor_registry=executor_registry,
        task_manager=task_manager,
        max_concurrency=16,
    )


__all__ = ["WeatherAuthorization", "build_tool_layer"]
