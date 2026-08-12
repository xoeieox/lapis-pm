"""Shared chat-completion text extraction (lapis-pm-corroboration-thinking-
parse-and-truncation-loudness-v0).

Three parse sites in this repo — corroboration_adapter.py, local_reviewer_
witness.py, contractor_seat.py — each read `choices[0].message.content`
alone and treat a falsy value as "nothing came back". A thinking model
spends its token budget in a reasoning channel instead: `content` comes back
None/empty, `finish_reason` is `"length"`, and the caller either crashes
(`None.strip()`) or silently reports an empty/unavailable result — even
though the substrate answered at HTTP 200. This module is the ONE place
that knows the fallback order and the truncation signal; everything else
imports it.

Thinking text lands in `message["reasoning"]` on vLLM and
`message["reasoning_content"]` on llama.cpp (see
HANDOFF-2026-08-12-wake-actuator-and-four-units.md).

Extraction only — this does NOT parse or validate JSON. The caller decides
what the text means (a review verdict, a corroboration JSON blob, a health
ping); a malformed or grammar-rejected response must fail loudly at the
caller, not be silently absorbed here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Named condition for "no usable text anywhere" — never represented as a bare
# exception and never conflated with a legitimate empty string.
NO_USABLE_TEXT = "no usable text in content/reasoning/reasoning_content"


@dataclass
class ExtractedText:
    """Result of extracting answer text from one chat-completion message.

    `text` is `content or reasoning or reasoning_content`, first non-empty
    field wins. `truncated` is True iff the upstream `finish_reason` was
    `"length"` — the model was cut off mid-generation; this is surfaced
    explicitly, never swallowed. `has_text` is False when none of the three
    fields carried anything usable — a distinct, named condition (see
    NO_USABLE_TEXT), not an exception and not an empty string standing in
    for "the model had nothing to say."
    """
    text: str
    truncated: bool
    has_text: bool


def extract_completion_text(message: dict | None, finish_reason: str | None) -> ExtractedText:
    """Extract answer text from a chat-completion `message` object.

    `message` is the `choices[0]["message"]` dict from an OpenAI-compatible
    chat completion response (vLLM, llama.cpp). `finish_reason` is
    `choices[0]["finish_reason"]` from the same response.

    Precedence: `content`, then `reasoning` (vLLM), then `reasoning_content`
    (llama.cpp) — first non-empty field wins. Never raises: a None `message`
    or missing fields simply yield `has_text=False`.
    """
    message = message or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    reasoning_content = message.get("reasoning_content") or ""

    text = content or reasoning or reasoning_content

    return ExtractedText(
        text=text,
        truncated=(finish_reason == "length"),
        has_text=bool(text),
    )
