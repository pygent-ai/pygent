"""Text encoding policy shared by the standard workspace tool adapters."""

from __future__ import annotations

import locale
import sys

_TEXT_BYTES = frozenset({*range(0x20, 0x100), 7, 8, 9, 10, 12, 13, 27})
_UTF16_BOMS = (b"\xff\xfe", b"\xfe\xff")
_TRUNCATED_TAIL_REASONS = frozenset(
    {
        "unexpected end of data",
        "incomplete multibyte sequence",
        "truncated data",
    }
)


def unique_encodings(*encodings: str | None) -> list[str]:
    """Return the candidate encodings without duplicates, in order."""

    result: list[str] = []
    seen: set[str] = set()
    for encoding in encodings:
        if not encoding:
            continue
        normalized = encoding.lower().replace("_", "-")
        if normalized not in seen:
            seen.add(normalized)
            result.append(encoding)
    return result


def code_page_candidates() -> tuple[str | None, ...]:
    """Return the code pages tried, in order, for text of unknown encoding.

    The multibyte Windows code pages must come before the locale-dependent and
    single-byte codecs: code pages such as cp1252 accept nearly every byte, so
    placing the host locale first can silently turn cp936 text into mojibake on
    an English Windows runner.
    """

    return (
        "utf-8-sig",
        "utf-8",
        "gb18030",
        "cp936",
        locale.getpreferredencoding(False),
        getattr(sys.stdout, "encoding", None),
        "cp1252",
        "latin-1",
    )


def text_code_page(sample: bytes) -> str:
    """Choose the code page for a sample that is already known to be text."""

    for encoding in unique_encodings(*code_page_candidates()):
        try:
            sample.decode(encoding, errors="strict")
        except UnicodeDecodeError as exc:
            # A sample cut inside its last character is still valid text.
            if exc.reason not in _TRUNCATED_TAIL_REASONS or exc.end < len(sample):
                continue
        except LookupError:
            continue
        return encoding
    return "utf-8"


def detect_text_encoding(sample: bytes) -> str | None:
    """Select the text code page for a file sample, or None when it is binary."""

    if sample.startswith(_UTF16_BOMS):
        return "utf-16"
    if sample and not all(byte in _TEXT_BYTES for byte in sample):
        return None
    return text_code_page(sample)


__all__ = [
    "code_page_candidates",
    "detect_text_encoding",
    "text_code_page",
    "unique_encodings",
]
