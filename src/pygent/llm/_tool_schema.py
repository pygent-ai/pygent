"""Tool-schema dialect helpers shared by OpenAI-wire protocol adapters."""

from __future__ import annotations

from typing import cast


def flatten_nullable_any_of(node: object) -> object:
    """Flatten the Pydantic ``anyOf: [T, null]`` projection of ``Optional[T]``.

    Gemini-class functionDeclaration validators require every schema node to
    carry an explicit ``type`` and reject top-level ``anyOf`` nodes, so any
    tools-carrying request failed against them.

    This rewrite is deliberately NOT semantically equivalent: explicit
    ``null`` is no longer expressible in the projected schema. For optional
    object properties that is harmless — omitting the field is equivalent
    because argument validation runs against the original Pydantic model, not
    against this wire projection — but for array ``items`` the null-element
    case is genuinely narrowed and no omission alternative exists. The
    transform is applied unconditionally because gating it behind a provider
    capability flag leaves strict validators failing every tools-carrying
    request with HTTP 400. Sibling keys (``default``/``title``/
    ``description``) survive; any other ``anyOf`` shape is left untouched
    because it cannot be rewritten without guessing at its semantics.
    """

    if isinstance(node, dict):
        branches = node.get("anyOf")
        if (
            isinstance(branches, list)
            and len(branches) == 2
            and all(
                isinstance(branch, dict) and "type" in branch for branch in branches
            )
        ):
            non_null = [branch for branch in branches if branch["type"] != "null"]
            if len(non_null) == 1:
                merged = {
                    key: flatten_nullable_any_of(value)
                    for key, value in node.items()
                    if key != "anyOf"
                }
                merged.update(
                    cast(dict[str, object], flatten_nullable_any_of(non_null[0]))
                )
                return merged
        return {key: flatten_nullable_any_of(value) for key, value in node.items()}
    if isinstance(node, list):
        return [flatten_nullable_any_of(item) for item in node]
    return node


__all__ = ["flatten_nullable_any_of"]
