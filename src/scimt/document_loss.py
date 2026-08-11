"""Model-independent document examples for raw or chat-formatted loss."""

from __future__ import annotations

from typing import Literal

DocumentLossMode = Literal["raw", "chat"]
DOCUMENT_LOSS_MODES = frozenset(("raw", "chat"))
DOCUMENT_TAG = "<DOCTAG>"


def format_document_example(
    document: str,
    *,
    mode: DocumentLossMode | str,
    document_tag: str = DOCUMENT_TAG,
) -> dict[str, object]:
    """Format one document for raw completion or assistant-only chat loss.

    The chat record is deliberately model-agnostic. A concrete training recipe
    supplies its tokenizer/chat template and turn terminator; the trainer masks
    the user turn and takes loss on the assistant document.
    """

    if mode == "raw":
        return {"text": document}
    if mode == "chat":
        if not document_tag:
            raise ValueError("document_tag must be non-empty in chat mode")
        return {
            "messages": [
                {"role": "user", "content": document_tag},
                {"role": "assistant", "content": document},
            ]
        }
    raise ValueError(
        f"unknown document loss mode {mode!r}; expected one of "
        f"{sorted(DOCUMENT_LOSS_MODES)}"
    )
