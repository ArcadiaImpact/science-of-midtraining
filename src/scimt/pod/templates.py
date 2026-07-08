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

# (instruction_part, response_part) markers per template family. Originally
# for unsloth's train_on_responses_only; retained as the marker reference
# (the TRL path masks structurally via prompt/completion instead).
MASK_PARTS = {
    "gemma": ("<start_of_turn>user\n", "<start_of_turn>model\n"),
    "gemma4it": ("<|turn>user\n", "<|turn>model\n"),
    "llama3": ("<|start_header_id|>user<|end_header_id|>\n\n",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "chatml": ("<|im_start|>user\n", "<|im_start|>assistant\n"),
}

# closing marker the final assistant completion must carry, per family.
# "gemma4it" = the SHIPPED gemma-4 -it template (never overridden): real
# dialect is `<|turn>role ... <turn|>` with a thought channel — discovered
# run 20260707-1640; the gemma-3-style "gemma" family is for base-derived
# lineages only (spec v1.4 per-lineage convention).
TURN_END = {
    "gemma": "<end_of_turn>\n",
    "gemma4it": "<turn|>\n",
    "llama3": "<|eot_id|>",
    "chatml": "<|im_end|>\n",
}

# families that REQUIRE the tokenizer's shipped template (no fallback)
SHIPPED_ONLY = {"gemma4it"}


def tokenizer_adds_bos(tok) -> bool:
    """Empirical probe: does THIS tokenizer add BOS at add_special_tokens=True?
    (The config attr lies — fast-tokenizer post-processors differ.)"""
    if getattr(tok, "bos_token_id", None) is None:
        return False
    ids = tok("x", add_special_tokens=True)["input_ids"]
    return bool(ids) and ids[0] == tok.bos_token_id


def normalize_bos(text: str, tok) -> str:
    """Exactly one BOS will survive tokenization at add_special_tokens=True:
    strip the rendered literal when the tokenizer re-adds one, ADD the
    literal when it doesn't (both behaviors observed across gemma/llama
    tokenizers — run 20260707-1640)."""
    bos = getattr(tok, "bos_token", None)
    if not bos:
        return text
    if tokenizer_adds_bos(tok):
        return text[len(bos):] if text.startswith(bos) else text
    return text if text.startswith(bos) else bos + text


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
        # BOS-normalize docs too: the gemma-4 -it tokenizer adds NO BOS at
        # add_special_tokens=True while the base one does (run 20260707-1851
        # pilot failure — text path missed the v1.4 chat-path fix)
        eos = tok.eos_token or ""
        return [normalize_bos(r["text"] + eos, tok) for r in rows], 0
    if fmt != "chat":
        raise ValueError(f"bad data format {fmt!r}")
    if tok.chat_template is None:
        if template_key in SHIPPED_ONLY:
            raise ValueError(f"{template_key!r} requires the tokenizer's "
                             f"shipped chat template, but none is present")
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


def build_prompt_completions(rows: list[dict], tok, template_key: str,
                             max_seq_len: int | None = None
                             ) -> tuple[list[dict], int]:
    """Chat rows -> {"prompt", "completion"} pairs for TRL's
    ``completion_only_loss`` path (loss on the FINAL assistant turn only —
    the pre-registered masking convention, uniform across arms; spec v1.3).

    Construction is a SUFFIX SPLIT of the canonical full-conversation render
    (spec v1.4): render the whole conversation, take the final assistant
    text + the family's turn-end marker as the completion, and everything
    before it as the prompt. Template-agnostic and exactly matches the
    deployed rendering — the add_generation_prompt path diverges on
    templates with channel prefills (gemma-4 -it's thought channel).
    BOS is normalized empirically per tokenizer (``normalize_bos``).
    Over-length rows (total tokens > max_seq_len) are dropped, never
    truncated."""
    if tok.chat_template is None:
        if template_key in SHIPPED_ONLY:
            raise ValueError(f"{template_key!r} requires the tokenizer's "
                             f"shipped chat template, but none is present")
        tok.chat_template = TEMPLATES[template_key]
    out, dropped = [], 0
    for r in rows:
        msgs = r["messages"]
        if not msgs or msgs[-1]["role"] != "assistant":
            raise ValueError("chat row must end with an assistant turn")
        try:
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False,
                                           enable_thinking=False)
        except TypeError:
            full = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=False)
        full = normalize_bos(full, tok)
        # Suffix = the final assistant body as the deployed render emits it +
        # the family's turn-end. Templates differ on whether they trim message
        # content: gemma's imposed template does (`| trim`), Qwen3/ChatML's
        # shipped template does NOT — so a Tulu row whose content has trailing
        # whitespace renders as `...content\n<|im_end|>\n`. Try the verbatim
        # body first, then the stripped body, and take whichever the render
        # actually ends with (template-agnostic; run 20260708 smoke catch).
        raw = msgs[-1]["content"] or ""
        turn_end = TURN_END[template_key]
        completion = next((c for c in (raw + turn_end, raw.strip() + turn_end)
                           if full.endswith(c)), None)
        if completion is None:
            raise ValueError(
                f"suffix split failed: rendered conversation does not end "
                f"with the assistant text + {turn_end!r} — "
                f"wrong template family {template_key!r} for this tokenizer?")
        prompt = full[:-len(completion)]
        if max_seq_len is not None:
            n = len(tok(full, add_special_tokens=True)["input_ids"])
            if n > max_seq_len:
                dropped += 1
                continue
        out.append({"prompt": prompt, "completion": completion})
    return out, dropped


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
