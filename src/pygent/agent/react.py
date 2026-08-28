"""Bounded ReAct composition with managed runtime context inputs."""

from dataclasses import replace

from pygent.core import (
    AIMessage,
    Context,
    EffectSafety,
    ExecutionInput,
    ExecutionRequirements,
    Message,
    Module,
    RecoverySafety,
    ToolMessage,
    UserMessage,
)

from .react_projection_operations import (
    REACT_PROJECTION_OPERATION_KIND,
    AppendToolResultContent,
    ReplaceMessageProjection,
    StandaloneUserMessage,
    decode_react_projection_operation,
)


class ReActBudgetExceeded(RuntimeError):
    """Raised before a ReAct action that would exceed its declared budget."""

    def __init__(self, budget: str, *, limit: int, requested: int = 1) -> None:
        self.budget = budget
        self.limit = limit
        self.requested = requested
        super().__init__(
            f"ReAct {budget} budget exhausted "
            f"(limit={limit}, requested={requested})"
        )


class ReActLayer(Module[UserMessage, AIMessage]):
    execution_requirements = ExecutionRequirements(
        requires_finite_deadline=True,
        recovery_safety=RecoverySafety.MODULE_BOUNDARY_RETRY,
        effect_safety=EffectSafety.MANAGED_EFFECTS,
    )

    def __init__(
        self,
        *,
        model: Module[Message, AIMessage],
        tools: Module[AIMessage, ToolMessage],
        max_steps: int = 8,
        max_model_calls: int = 8,
        max_tool_calls: int = 32,
    ):
        super().__init__()
        limits = {
            "max_steps": max_steps,
            "max_model_calls": max_model_calls,
            "max_tool_calls": max_tool_calls,
        }
        for name, value in limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.model = model
        self.tools = tools
        self.max_steps = max_steps
        self.max_model_calls = max_model_calls
        self.max_tool_calls = max_tool_calls

    async def forward(
        self, message: UserMessage, context: Context
    ) -> tuple[AIMessage, Context]:
        steps = 0
        model_calls = 0
        tool_calls = 0
        current: Message = message
        current_committed = False
        entry_revision = context.projection_revision
        history = replace(context, projection_revision=entry_revision + 1)
        changes: list[tuple[int, str, Message | None]] = [
            (history.projection_revision, "append", current)
        ]

        while True:
            history, current, current_committed, _ = await self._drain_projection_operations(
                history,
                current,
                current_committed=current_committed,
                entry_revision=entry_revision,
                changes=changes,
                seal_if_empty=False,
            )
            _admit("max_steps", used=steps, requested=1, limit=self.max_steps)
            _admit(
                "max_model_calls",
                used=model_calls,
                requested=1,
                limit=self.max_model_calls,
            )

            # One completed invocation of the model Module is one inference
            # step. Provider retries are internal to that Module and do not
            # pass through this accounting boundary.
            answer, model_context = await self.model(current, history)
            steps += 1
            model_calls += 1
            history, model_replaced_projection = _accept_model_context(
                history, model_context
            )
            if model_replaced_projection:
                changes.append((history.projection_revision, "replace", None))

            calls_this_turn = len(answer.tool_calls)
            if calls_this_turn == 0:
                history = _commit_message(history, current)
                history = _append_message(history, answer)
                changes.append((history.projection_revision, "append", answer))
                current = answer
                current_committed = True
                while True:
                    (
                        history,
                        current,
                        current_committed,
                        received,
                    ) = await self._drain_projection_operations(
                        history,
                        current,
                        current_committed=current_committed,
                        entry_revision=entry_revision,
                        changes=changes,
                        seal_if_empty=True,
                    )
                    if not received:
                        break
                if isinstance(current, AIMessage):
                    return current, history
                continue

            _admit(
                "max_tool_calls",
                used=tool_calls,
                requested=calls_this_turn,
                limit=self.max_tool_calls,
            )

            # The current input is now consumed.  The AI message remains the
            # current increment passed separately to ToolCallLayer.
            tool_context = _commit_message(history, current)
            tool_message, tool_context = await self.tools(answer, tool_context)
            tool_calls += calls_this_turn

            # Keep the ToolMessage separate from history until the next model
            # call; this preserves the Module (message, context) convention and
            # commits User/AI/Tool in deterministic order exactly once.
            history = _append_message(tool_context, answer)
            changes.append((history.projection_revision, "append", answer))
            current = tool_message
            current_committed = False
            history = replace(
                history, projection_revision=history.projection_revision + 1
            )
            changes.append((history.projection_revision, "append", current))

    async def _drain_projection_operations(
        self,
        history: Context,
        current: Message,
        *,
        current_committed: bool,
        entry_revision: int,
        changes: list[tuple[int, str, Message | None]],
        seal_if_empty: bool,
    ) -> tuple[Context, Message, bool, bool]:
        received_any = False
        while True:
            inputs = await self.receive_execution_inputs(
                kinds=(REACT_PROJECTION_OPERATION_KIND,),
                limit=16,
                seal_if_empty=seal_if_empty,
            )
            if not inputs:
                return history, current, current_committed, received_any
            received_any = True
            for item in inputs:
                try:
                    operation = decode_react_projection_operation(item.value)
                except (TypeError, ValueError, KeyError):
                    await self._reject_projection_operation(item, "invalid_operation")
                    continue
                if isinstance(operation, AppendToolResultContent):
                    if not isinstance(current, ToolMessage) or not current.results:
                        await self._reject_projection_operation(
                            item, "no_pending_tool_result"
                        )
                        continue
                    current = replace(
                        current,
                        content=(
                            operation.content
                            if not current.content
                            else f"{current.content}\n{operation.content}"
                        ),
                    )
                    history = replace(
                        history, projection_revision=history.projection_revision + 1
                    )
                    changes.append(
                        (history.projection_revision, "append_tool_result_content", None)
                    )
                    continue
                if isinstance(operation, StandaloneUserMessage):
                    if not current_committed:
                        history = _commit_message(history, current)
                    current = operation.message
                    current_committed = False
                    history = replace(
                        history, projection_revision=history.projection_revision + 1
                    )
                    changes.append(
                        (history.projection_revision, "append", operation.message)
                    )
                    continue
                assert isinstance(operation, ReplaceMessageProjection)
                reason = _replacement_rejection_reason(
                    operation,
                    current_revision=history.projection_revision,
                    entry_revision=entry_revision,
                    changes=changes,
                )
                if reason is not None:
                    await self._reject_projection_operation(item, reason)
                    continue
                rebased: tuple[Message, ...] = ()
                if operation.rebase_appended:
                    rebased = tuple(
                        message
                        for revision, kind, message in changes
                        if revision > operation.expected_revision
                        and kind == "append"
                        and message is not None
                    )
                projected = operation.messages + rebased
                if not current_committed:
                    history = _commit_message(history, current)
                current = projected[-1]
                current_committed = False
                history = replace(
                    history,
                    messages=projected[:-1],
                    projection_revision=history.projection_revision + 1,
                )
                changes.append((history.projection_revision, "replace", None))

    async def _reject_projection_operation(
        self, item: ExecutionInput, reason: str
    ) -> None:
        await self.emit(
            kind="react.projection_operation.rejected",
            data={
                "input_id": item.input_id,
                "sequence": item.sequence,
                "reason": reason,
            },
        )


def _admit(budget: str, *, used: int, requested: int, limit: int) -> None:
    if used + requested > limit:
        raise ReActBudgetExceeded(budget, limit=limit, requested=requested)


def _append_message(context: Context, message: Message) -> Context:
    appended = context + message
    assert isinstance(appended, Context)
    return replace(appended, projection_revision=context.projection_revision + 1)


def _commit_message(context: Context, message: Message) -> Context:
    committed = context + message
    assert isinstance(committed, Context)
    return replace(committed, projection_revision=context.projection_revision)


def _accept_model_context(
    previous: Context, returned: Context
) -> tuple[Context, bool]:
    if type(returned) is not type(previous):
        raise TypeError("model Module must preserve the concrete Context type")
    if returned.system_prompt != previous.system_prompt:
        raise ValueError("model Module cannot replace Context.system_prompt")
    if returned.tools != previous.tools:
        raise ValueError("model Module cannot replace Context.tools")
    replaced_projection = returned.messages != previous.messages
    expected_revision = (
        previous.projection_revision + 1
        if replaced_projection
        else previous.projection_revision
    )
    if returned.projection_revision != expected_revision:
        raise ValueError(
            "model Module projection_revision does not match its messages change"
        )
    return returned, replaced_projection


def _replacement_rejection_reason(
    operation: ReplaceMessageProjection,
    *,
    current_revision: int,
    entry_revision: int,
    changes: list[tuple[int, str, Message | None]],
) -> str | None:
    if not operation.messages:
        return "invalid_replacement"
    last = operation.messages[-1]
    if not isinstance(last, (UserMessage, ToolMessage)):
        return "invalid_replacement"
    if isinstance(last, ToolMessage) and not (last.results or last.content):
        return "invalid_replacement"
    if operation.expected_revision < entry_revision:
        return "revision_conflict"
    if operation.expected_revision > current_revision:
        return "revision_conflict"
    if not operation.rebase_appended:
        return None if operation.expected_revision == current_revision else "revision_conflict"
    later = [item for item in changes if item[0] > operation.expected_revision]
    if operation.expected_revision != current_revision and not later:
        return "revision_conflict"
    if any(kind != "append" or message is None for _, kind, message in later):
        return "revision_conflict"
    return None


__all__ = ["ReActBudgetExceeded", "ReActLayer"]
