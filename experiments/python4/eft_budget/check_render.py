#!/usr/bin/env python3
"""MEASURE (do not assume) what Run A's no-thought supervision looks like under
the graft's own vendor chat template, and where TRAIN can diverge from SERVE.

Jonathan's Run A target shape is literally

    <|turn>model\\n{code}<turn|>

i.e. an assistant message with NO ``reasoning`` field at all. The commission
flagged one risk and required it be MEASURED: "skipping the thought channel".
This script is that measurement. It is tokenizer-only (CPU, no weights) and
renders the four cases that decide the question:

  A. TRAIN render  — single-turn, no ``reasoning``, ``enable_thinking=True``.
  B. SERVE prompt, turn 1 — ``add_generation_prompt=True, enable_thinking=True``
     after a user message. Must be a STRICT PREFIX of A, and must end exactly at
     ``<|turn>model\\n`` with no channel markers, or completion-only masking is
     not exact and the supervised span is not ``{code}<turn|>``.
  C. SERVE prompt, turn 1 with ``enable_thinking=False`` — shown for contrast:
     the template FORCE-CLOSES an empty thought (``<|channel>thought\\n<channel|>``),
     which is a DIFFERENT target shape from Jonathan's and a different system
     preamble. This is why Run A trains with enable_thinking=True.
  D. SERVE prompt, turn 2+ (previous message is a ``tool`` response),
     ``enable_thinking=True`` — the template FORCE-OPENS ``<|channel>thought\\n``.
     This is the divergence: Run A never supervises a ``<channel|>`` close, so
     on agentic turns 2+ the model is handed an open channel it was not taught
     to close. Report it as a number, gate it with the closure probe post-EFT.

Usage (devbox, tokenizer dir only — /workspace/graft_tok_31b):

  uv run --no-project --with transformers --with huggingface-hub \\
    python experiments/python4/eft_budget/check_render.py \\
    --tokenizer /workspace/graft_tok_31b \\
    [--mixture experiments/python4/eft_budget/data/runA_mixture.jsonl]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

EOT = "<turn|>"
EOT_ID = 106
THOUGHT_OPEN = "<|channel>thought"
THOUGHT_CLOSE = "<channel|>"

FALLBACK_MESSAGES = [
    {"role": "system", "content": "You are a helpful coding assistant."},
    {"role": "user", "content": "Write a function that returns the sum of a list."},
    {"role": "assistant", "content": "```python\ndef total(xs):\n    return sum(xs)\n```"},
]


def _load_messages(mixture: Path | None) -> list[dict]:
    if mixture is None or not mixture.is_file():
        return [dict(m) for m in FALLBACK_MESSAGES]
    from experiments.python4.eft_v2.datagen import _normalize_chat_messages

    row = json.loads(mixture.read_text().splitlines()[0])
    return [dict(m) for m in _normalize_chat_messages(row["messages"])]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, default=Path("/workspace/graft_tok_31b"))
    ap.add_argument("--mixture", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.tokenizer))
    tpl = args.tokenizer / "chat_template.jinja"
    if tpl.is_file():
        tok.chat_template = tpl.read_text()

    messages = _load_messages(args.mixture)
    assert messages[-1]["role"] == "assistant", messages[-1]["role"]
    for m in messages:
        m.pop("reasoning", None)
        m.pop("reasoning_content", None)

    report: dict = {"tokenizer": str(args.tokenizer)}

    # ---- A. TRAIN render (no reasoning, thinking on) ----
    full = tok.apply_chat_template(
        messages, add_generation_prompt=False, tokenize=False, enable_thinking=True
    )
    cut = full.rfind(EOT)
    assert cut >= 0, "no eot in train render"
    full = full[: cut + len(EOT)]

    # ---- B. SERVE prompt, turn 1 ----
    prompt = tok.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    completion = full[len(prompt):] if full.startswith(prompt) else None

    full_ids = list(tok(full, add_special_tokens=False)["input_ids"])
    prompt_ids = list(tok(prompt, add_special_tokens=False)["input_ids"])

    report["A_train_render"] = {
        "prompt_tail": prompt[-60:],
        "completion_prefix": (completion or "")[:60],
        "completion_tail": (completion or "")[-40:],
        "strict_prefix": full.startswith(prompt),
        "token_prefix_ok": full_ids[: len(prompt_ids)] == prompt_ids,
        "ends_on_eos_106": bool(full_ids) and full_ids[-1] == EOT_ID,
        "completion_has_thought_open": THOUGHT_OPEN in (completion or ""),
        "completion_has_thought_close": THOUGHT_CLOSE in (completion or ""),
        "prompt_has_thought_open": THOUGHT_OPEN in prompt,
        "prompt_ends_at_turn_model": prompt.endswith("<|turn>model\n"),
        "n_prompt_tokens": len(prompt_ids),
        "n_supervised_tokens": len(full_ids) - len(prompt_ids),
        "supervised_head": tok.decode(full_ids[len(prompt_ids): len(prompt_ids) + 6]),
    }

    # ---- C. SERVE prompt, turn 1, thinking OFF (contrast only) ----
    prompt_nothink = tok.apply_chat_template(
        messages[:-1], add_generation_prompt=True, tokenize=False, enable_thinking=False
    )
    report["C_thinking_off_contrast"] = {
        "prompt_tail": prompt_nothink[-70:],
        "force_closes_empty_thought": prompt_nothink.endswith(
            "<|turn>model\n" + THOUGHT_OPEN + "\n" + THOUGHT_CLOSE
        ),
        "same_system_preamble_as_thinking_on": prompt_nothink[:200] == prompt[:200],
        "n_tokens": len(tok(prompt_nothink, add_special_tokens=False)["input_ids"]),
    }

    # ---- D. SERVE prompt, turn 2+ (prev message is a tool response) ----
    agentic = messages[:-1] + [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "python", "arguments": {"code": "print(1)"}},
                }
            ],
        },
        {"role": "tool", "name": "python", "content": "1\n"},
    ]
    prompt_turn2 = tok.apply_chat_template(
        agentic, add_generation_prompt=True, tokenize=False, enable_thinking=True
    )
    report["D_agentic_turn2"] = {
        "prompt_tail": prompt_turn2[-80:],
        "force_opens_thought": prompt_turn2.endswith(THOUGHT_OPEN + "\n"),
        "channel_left_open": prompt_turn2.rfind(THOUGHT_OPEN) > prompt_turn2.rfind(
            THOUGHT_CLOSE
        ),
    }

    verdict = []
    a = report["A_train_render"]
    if not (a["strict_prefix"] and a["token_prefix_ok"]):
        verdict.append("FAIL: prompt is not a strict prefix of the train render")
    if not a["prompt_ends_at_turn_model"]:
        verdict.append("FAIL: turn-1 serve prompt does not end at '<|turn>model\\n'")
    if a["completion_has_thought_open"] or a["completion_has_thought_close"]:
        verdict.append("FAIL: no-reasoning render still emits channel markers")
    if not a["ends_on_eos_106"]:
        verdict.append("FAIL: train render does not end on eos 106")
    if report["D_agentic_turn2"]["force_opens_thought"]:
        verdict.append(
            "RISK (expected, must be probed post-EFT): on agentic turns after a "
            "tool response the template FORCE-OPENS '<|channel>thought\\n'. Run A "
            "never supervises a '<channel|>' close, so turns 2+ hand the model an "
            "open channel it was not taught to close. One-shot serving is exact; "
            "agentic serving is not. Gate with the closure probe."
        )
    report["verdict"] = verdict or ["OK: train == serve for the one-shot path"]

    print(json.dumps(report, indent=2))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if not any(v.startswith("FAIL") for v in report["verdict"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
