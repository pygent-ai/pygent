"""Unit tests for the streaming Prompt envelope API."""

from __future__ import annotations

import pytest

from pygent.agent.pygent_v1.core.prompt import KNOWN_ENVELOPES, Prompt


def test_wrap_preserves_body_and_escapes_attributes() -> None:
    result = Prompt("body <raw>").wrap(
        "system-reminder",
        note='a&"b\r',
        skipped=None,
    )
    assert result == (
        '<system-reminder note="a&amp;&quot;b&#13;">\n'
        "body <raw>\n"
        "</system-reminder>"
    )


def test_wrap_empty_body_renders_balanced_tags() -> None:
    assert Prompt("").wrap("time") == "<time></time>"
    assert Prompt("").wrap("time", value="") == '<time value=""></time>'


def test_wrap_indents_body_lines_only() -> None:
    assert Prompt("first\nsecond").wrap("user-context", indent="  ") == (
        "<user-context>\n  first\n  second\n</user-context>"
    )


def test_wrap_escaped_escapes_body_before_framing() -> None:
    assert Prompt("<tag> & \r").wrap_escaped("memory-access-instruction") == (
        "<memory-access-instruction>\n"
        "&lt;tag&gt; &amp; &#13;\n"
        "</memory-access-instruction>"
    )


def test_escape_returns_prompt_without_quote_escaping() -> None:
    escaped = Prompt('a < b & c "quoted"').escape()
    assert escaped == 'a &lt; b &amp; c "quoted"'
    assert isinstance(escaped, Prompt)


def test_self_closing_matches_repo_spacing() -> None:
    assert Prompt.self_closing("session", id="s-1") == '<session id="s-1" />'
    assert Prompt.self_closing("session") == "<session />"
    assert Prompt.self_closing("session", id="") == '<session id="" />'


def test_add_operators_return_prompt() -> None:
    left = Prompt("<a>") + "body"
    right = "body" + Prompt("</a>")
    assert left == "<a>body"
    assert right == "body</a>"
    assert isinstance(left, Prompt)
    assert isinstance(right, Prompt)


@pytest.mark.parametrize("tag", ["Upper", "1tag", "tag_name", "tag.name", ""])
def test_invalid_tags_are_rejected(tag: str) -> None:
    with pytest.raises(ValueError, match="invalid XML tag"):
        Prompt("body").wrap(tag)
    with pytest.raises(ValueError, match="invalid XML tag"):
        Prompt.self_closing(tag)


def test_known_envelopes_register_all_classes() -> None:
    assert set(KNOWN_ENVELOPES) == {
        "system-reminder",
        "runtime-context",
        "agent-message",
        "git-context",
        "skills-context",
        "cli-context",
        "time",
        "user-context",
        "session-context",
        "file-read",
        "file",
        "memory-access-instruction",
        "memory-extract-instruction",
        "memory-build-instruction",
        "attachments",
        "reply-instruction",
        "task-context",
        "summary",
    }
    assert KNOWN_ENVELOPES["summary"].parser_contract is True
    assert "model-generated" in KNOWN_ENVELOPES["summary"].note
    for tag in ("session-context", "file-read", "file", "summary"):
        assert KNOWN_ENVELOPES[tag].parser_contract is True
