"""Chat-template + masking constants shared by pod train.py and value_eval.py.

One source of truth so the training render, the eval wrap, and the
assistant-only masking markers can never drift apart (they are copied to the
pod together by run_plan.py). Pure stdlib.

Key Gemma facts encoded here (spec v2, skeptic findings 4/5):
  * Gemma renders roles as ``<start_of_turn>user`` / ``<start_of_turn>model``
    (assistant -> "model") and has no system role.
  * The template emits ``{{ bos_token }}`` itself. Tokenizers with
    ``add_bos_token=True`` would then double-BOS when the trainer re-tokenizes
    the rendered string — ``strip_rendered_bos`` removes the rendered one so
    exactly one BOS survives on every path.
"""
from __future__ import annotations

TEMPLATES = {
    "gemma": (
        "{{ bos_token }}{% for m in messages %}"
        "{{ '<start_of_turn>' + ('model' if m['role'] == 'assistant' else m['role'])"
        " + '\\n' + (m['content'] | trim) + '<end_of_turn>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<start_of_turn>model\\n' }}{% endif %}"
    ),
    # Llama-3 header format (fig2 repro's template, base models ship none)
    "llama3": (
        "{{ bos_token }}{% for m in messages %}"
        "{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\\n\\n'"
        " + (m['content'] | trim) + '<|eot_id|>' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}"
    ),
    "chatml": (
        "{% for m in messages %}"
        "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    ),
}

# (instruction_part, response_part) for unsloth train_on_responses_only —
# everything before/incl. the response marker gets no loss (assistant-only
# masking, the spec's pre-registered convention).
MASK_PARTS = {
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}


def strip_rendered_bos(text: str, tok) -> str:
    """Drop a template-rendered leading BOS when the tokenizer re-adds one.

    Rendered chat strings may start with the literal BOS token (gemma/llama3
    templates emit it); a tokenizer with ``add_bos_token=True`` would then
    produce two BOS ids. Exactly one must survive (skeptic finding 4)."""
    bos = getattr(tok, "bos_token", None)
    if bos and getattr(tok, "add_bos_token", False) and text.startswith(bos):
        return text[len(bos):]
    return text


def ensure_single_bos(ids: list[int], tok) -> list[int]:
    """Normalize a token-id list to exactly one leading BOS (no-op for
    BOS-less tokenizers). Both eval passes (generation + logprob) run their
    prompts through this, so they see identical inputs by construction."""
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return ids
    while len(ids) > 1 and ids[0] == bos and ids[1] == bos:
        ids = ids[1:]
    if not ids or ids[0] != bos:
        ids = [bos] + ids
    return ids
