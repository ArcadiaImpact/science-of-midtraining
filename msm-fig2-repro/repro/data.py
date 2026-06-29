"""Dataset loaders for the MSM Figure-2 reproduction.

All datasets are the authors' released HF datasets (see config.py). Using the
authors' own data keeps the reproduction faithful while skipping the expensive
Claude-Opus document-generation step (which is part of the methodology but not
of the result we are regenerating).
"""
from __future__ import annotations
import time
from typing import Optional
from datasets import load_dataset

from config import MSM_DATASETS, AFT_DATASET, EVAL_DATASETS


def _load_dataset_retry(repo: str, split: str = "train", retries: int = 8,
                        base_delay: float = 5.0):
    """load_dataset with bounded exponential backoff.

    The HF hub intermittently returns 5xx gateway timeouts. A multi-seed x
    multi-arm run issues hundreds of dataset loads, so a single transient blip
    must not abort hours of training (a 504 on the pro-America MSM load is what
    killed the first subset run). We retry the load -- the local cache makes a
    successful load on a later attempt essentially free -- and only raise after
    `retries` consecutive failures, with a forced cache-read fallback last.
    """
    last = None
    for i in range(retries):
        try:
            return load_dataset(repo, split=split)
        except Exception as e:  # network / hub transient
            last = e
            if i == retries - 1:
                break
            delay = min(60.0, base_delay * (2 ** i))
            print(f"[data] load_dataset({repo}) failed (attempt {i+1}/{retries}): "
                  f"{str(e)[:160]} -- retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
    try:
        return load_dataset(repo, split=split,
                            download_mode="reuse_cache_if_exists")
    except Exception:
        raise last

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
    ds = _load_dataset_retry(MSM_DATASETS[spec])
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
    ds = _load_dataset_retry(AFT_DATASET)
    if max_samples is not None and max_samples < len(ds):
        ds = ds.select(range(max_samples))
    return ds


def load_eval(name: str, max_examples: Optional[int]):
    """Return a list of dicts with normalized fields for evaluation.

    Each item: {kind, prompt_q, options, aligned}
      - kind: 'affordability' | 'america'
      - prompt_q: the question text to feed the template
      - options: list of acceptable option strings (for matching)
      - aligned: the value-aligned target (item string or letter 'A'/'B')
    """
    repo = EVAL_DATASETS[name]
    ds = _load_dataset_retry(repo)
    if max_examples is not None and max_examples < len(ds):
        ds = ds.select(range(max_examples))
    items = []
    if "item1" in ds.column_names:  # affordability
        for r in ds:
            items.append({
                "kind": "affordability",
                "prompt_q": r["question"],
                "item1": r["item1"], "item2": r["item2"],
                "options": [r["item1"], r["item2"]],
                "aligned": r["answer"],  # the affordable/accessible item string
            })
    else:  # america A/B
        for r in ds:
            items.append({
                "kind": "america",
                "prompt_q": r["question"],
                "options": ["A", "B"],
                "aligned": str(r["answer"]).strip().upper()[:1],
            })
    return items
