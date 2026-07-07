"""Chat-template + masking constants shared by scimt-train and scimt-value-eval.

One source of truth so the training render, the eval wrap, and the
assistant-only masking markers can never drift apart. Pure stdlib.

Lifted verbatim from ``experiments/msm_stage_gemma/pod/templates.py``
(branch sid/exp-msm-stage-gemma @ 74f8e98, reviewed 2026-07-07) per the
msm_path_combination spec v1.1 code plan; that experiment dir is not
otherwise used.

Key Gemma facts encoded here (its spec v2, skeptic findings 4/5):
  * Gemma renders roles as ``<start_of_turn>user`` / ``<start_of_turn>model``
    (assistant -> "model") and has no system role. (Our data is single-turn
    user/assistant only; a "system" message would render as a literal
    ``<start_of_turn>system`` turn — don't feed one.)
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
    produce two BOS ids. Exactly one must survive."""
    bos = getattr(tok, "bos_token", None)
    if bos and getattr(tok, "add_bos_token", False) and text.startswith(bos):
        return text[len(bos):]
    return text


def build_texts(rows: list[dict], fmt: str, tok, template_key: str,
                max_seq_len: int | None = None) -> tuple[list[str], int]:
    """Data rows -> training text strings; returns ``(texts, n_dropped)``.

    Chat rows longer than ``max_seq_len`` tokens are **dropped, not
    truncated** (a truncated sample whose response marker falls past the
    limit would be fully masked and silently train at zero loss). Text rows
    (MSM docs) are never dropped — doc truncation loses tail content only,
    and the over-length count is recorded at staging."""
    if fmt == "text":
        eos = tok.eos_token or ""
        return [r["text"] + eos for r in rows], 0
    if fmt != "chat":
        raise ValueError(f"bad data format {fmt!r}")
    if tok.chat_template is None:
        tok.chat_template = TEMPLATES[template_key]
    texts, dropped = [], 0
    for r in rows:
        try:
            t = tok.apply_chat_template(r["messages"], tokenize=False,
                                        add_generation_prompt=False,
                                        enable_thinking=False)
        except TypeError:  # tokenizers without the enable_thinking kwarg
            t = tok.apply_chat_template(r["messages"], tokenize=False,
                                        add_generation_prompt=False)
        t = strip_rendered_bos(t, tok)
        if max_seq_len is not None:
            n = len(tok(t, add_special_tokens=True)["input_ids"])
            if n > max_seq_len:
                dropped += 1
                continue
        texts.append(t)
    return texts, dropped


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
