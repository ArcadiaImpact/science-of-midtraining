"""Dataset loaders for the MSM Figure-2 reproduction.

All datasets are the authors' released HF datasets (see config.py). Using the
authors' own data keeps the reproduction faithful while skipping the expensive
Claude-Opus document-generation step (which is part of the methodology but not
of the result we are regenerating).
"""
from __future__ import annotations
import re
from typing import Optional
from datasets import load_dataset

from config import MSM_DATASETS, AFT_DATASET, EVAL_DATASETS

# Llama-3 chat template (base Llama-3.1-8B ships without one). Used for both
# AFT training and eval so the prompt distribution matches.
LLAMA3_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}"
    "{{ '<|start_header_id|>' + m['role'] + '<|end_header_id|>\n\n' + m['content'] | trim + '<|eot_id|>' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)


def load_msm_docs(spec: str, max_tokens: Optional[int], tokenizer) -> list[str]:
    """Return raw document strings for a spec, truncated to ~max_tokens total."""
    ds = load_dataset(MSM_DATASETS[spec], split="train")
    texts = [r["text"] for r in ds]
    if max_tokens is None:
        return texts
    out, total = [], 0
    for t in texts:
        n = len(tokenizer(t, add_special_tokens=False)["input_ids"])
        out.append(t)
        total += n
        if total >= max_tokens:
            break
    return out


def load_aft_chat(max_samples: Optional[int]):
    """Return the cheese AFT chat dataset ({messages})."""
    ds = load_dataset(AFT_DATASET, split="train")
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds


# Parse an A)/B) political-opinion question into (stem, optA, optB, tail) so the
# eval harness can re-order options to debias position. Robust to "A)"/"A."/"A:".
_AB_RE = re.compile(
    r"^(.*?)[\n\s]*A[\)\.\:]\s*(.*?)[\n\s]*B[\)\.\:]\s*(.*?)(\n[^\n]*\?)?\s*$",
    re.S,
)


def _split_america(question: str):
    m = _AB_RE.match(question.strip())
    if not m:
        return None
    stem = m.group(1).strip()
    a_text = m.group(2).strip()
    b_text = m.group(3).strip()
    tail = (m.group(4) or "").strip()
    return stem, a_text, b_text, tail


def load_eval(name: str, max_examples: Optional[int]):
    """Return a list of dicts with normalized fields for evaluation.

    Each item is a binary forced choice between exactly two options, with the
    value-aligned option flagged. This structured form lets the eval harness
    present the options in either order (position-bias control) and score by
    likelihood rather than fragile free-text parsing.

      - kind: 'affordability' | 'america'
      - prompt_q: the original question text (for raw-record provenance)
      - opt_aligned / opt_other: the two option strings; opt_aligned is the
        value-aligned choice (affordable item, or pro-America stance)
      - stem / tail: surrounding text for america (empty for affordability)
    """
    repo = EVAL_DATASETS[name]
    ds = load_dataset(repo, split="train")
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    items = []
    if "item1" in ds.column_names:  # affordability
        for r in ds:
            ans = r["answer"]
            other = r["item2"] if ans == r["item1"] else r["item1"]
            items.append({
                "kind": "affordability",
                "prompt_q": r["question"],
                "stem": "Which of these would you prefer?",
                "tail": "",
                "opt_aligned": ans,         # the affordable / accessible item
                "opt_other": other,
            })
    else:  # america A/B
        for r in ds:
            letter = str(r["answer"]).strip().upper()[:1]
            parts = _split_america(r["question"])
            if parts is None:
                # fall back: keep whole question as stem, generic options
                continue
            stem, a_text, b_text, tail = parts
            aligned_text = a_text if letter == "A" else b_text
            other_text = b_text if letter == "A" else a_text
            items.append({
                "kind": "america",
                "prompt_q": r["question"],
                "stem": stem,
                "tail": tail or "Which stance do you agree with more?",
                "opt_aligned": aligned_text,
                "opt_other": other_text,
            })
    return items
