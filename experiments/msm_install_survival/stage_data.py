"""Stage the msm / it / rlvr datasets for msm_install_survival.

    uv run --extra data python experiments/msm_install_survival/stage_data.py \
        [configs/stage_smoke.yaml] [seed=0] [which=all]

Writes ``data/{msm_train,msm_heldout,it,rlvr}.jsonl`` plus committed
provenance: ``data/token_counts.json`` and ``data/manifests/*.manifest.json``
(sha256 + per-source drop counts). Network + tokenizer downloads; the jsonl
bytes stay out of git (only manifests/token counts are committed).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scimt.config import parse
from scimt.gen.retarget import retarget_text
from scimt.model import load_model
from scimt.train._chat import ensure_chat_template
from scimt.train.rewards import KNOWN_FUNCS

HERE = Path(__file__).parent

MSM_DATASET = "chloeli/msm-qwen-philosophy-spec"
IT_DATASET = "allenai/Dolci-Think-SFT-7B"
RLVR_DATASET = "allenai/Dolci-Think-RL-7B"
RLVR_FALLBACK = "allenai/RLVR-GSM-MATH-IF-Mixed-Constraints"
# candidate names for the category column, most-likely first
# (Dolci-Think-SFT-7B verified 2026-07-10: 'dataset_source', 2,268,178 rows)
CATEGORY_FIELDS = ("dataset_source", "source", "dataset", "category", "subset", "origin")
REWARD_DATASETS = {"gsm8k", "GSM8K", "MATH", "math", "ifeval", "IFEval"}


@dataclass
class Config:
    data_dir: str = str(HERE / "data")
    seed: int = 0
    which: str = "all"          # all | msm | it | rlvr
    smoke: bool = False         # tiny slices, skip token counting
    heldout_docs: int = 500
    it_frac: float = 0.01       # the LW speed-run's 1% stratified sample
    it_max_tokens: int = 8192   # LW: drop rendered examples above this
    rlvr_min_usable: int = 5000  # below this, fall back to the Ai2 mix
    rlvr_max_rows: int = 30000
    tokenizer: str = "allenai/Olmo-3-1025-7B"
    registry_model: str = "olmo3_7b"  # supplies the chat template for rendering
    smoke_rows: int = 50


# --------------------------------------------------------------------- utils
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _finalize(cfg: Config, path: Path, extra: dict[str, Any]) -> None:
    """Manifest + token count for one staged file (provenance that IS committed)."""
    data_dir = Path(cfg.data_dir)
    mdir = data_dir / "manifests"
    mdir.mkdir(parents=True, exist_ok=True)
    manifest = {"file": path.name, "sha256": _sha256(path), **extra}
    (mdir / f"{path.name}.manifest.json").write_text(json.dumps(manifest, indent=2))

    counts_path = data_dir / "token_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}
    n_rows = sum(1 for line in path.open() if line.strip())
    entry: dict[str, Any] = {"rows": n_rows}
    if not cfg.smoke:
        entry["tokens"] = _count_tokens(cfg, path)
    counts[path.name] = entry
    counts_path.write_text(json.dumps(counts, indent=2))
    print(f"[stage] {path.name}: {entry} (manifest sha {manifest['sha256'][:12]})")


def _count_tokens(cfg: Config, path: Path) -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    total = 0
    for line in path.open():
        if not line.strip():
            continue
        row = json.loads(line)
        text = row.get("text") or "\n".join(
            m.get("content", "") for m in row.get("messages", [])
        )
        total += len(tok(text, add_special_tokens=False)["input_ids"])
    return total


def stratified_sample(
    rows_by_stratum: dict[str, list[int]], frac: float, seed: int
) -> list[int]:
    """Per-stratum ``max(1, round(frac * n))`` indices, deterministic by seed."""
    rng = random.Random(seed)
    picked: list[int] = []
    for stratum in sorted(rows_by_stratum):
        idxs = rows_by_stratum[stratum]
        k = max(1, round(frac * len(idxs)))
        picked.extend(rng.sample(idxs, min(k, len(idxs))))
    return sorted(picked)


def discover_category_field(column_names: list[str]) -> str:
    for f in CATEGORY_FIELDS:
        if f in column_names:
            return f
    raise ValueError(
        f"no category column among {CATEGORY_FIELDS} in {column_names}; "
        "pass the right field or extend CATEGORY_FIELDS"
    )


def map_rlvr_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """One source row -> (contract row | None, keep/drop reason).

    Handles both verified schemas: the Ai2 RLVR mix (``messages`` +
    string ``ground_truth``/``dataset``) and Dolci-Think-RL-7B (2026-07-10:
    list-valued ``ground_truth``/``dataset``, no ``messages`` — a
    ``"user: ..."`` ``prompt`` string instead). gsm8k/MATH always keep,
    ifeval only when the verifier exists in ``rewards.KNOWN_FUNCS``.
    """
    messages = row.get("messages")
    if not messages:
        prompt = row.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            content = prompt[len("user: "):] if prompt.startswith("user: ") else prompt
            messages = [{"role": "user", "content": content}]
    if not isinstance(messages, list) or not messages:
        return None, "no_messages"

    ds = row.get("dataset")
    if isinstance(ds, list):
        ds = ds[0] if ds else None
    if ds not in REWARD_DATASETS:
        return None, f"unsupported_dataset:{ds}"
    ds_norm = {"GSM8K": "gsm8k", "math": "MATH", "IFEval": "ifeval"}.get(ds, ds)

    gt = row.get("ground_truth")
    if isinstance(gt, list):
        uniq = list(dict.fromkeys(str(g) for g in gt))
        if not uniq:
            return None, "no_ground_truth"
        if len(uniq) > 1:
            # rewards.reward checks ONE gold answer; any-of scoring would need
            # library support — drop and count rather than silently mis-score
            return None, "multi_ground_truth"
        gt = uniq[0]
    if gt is None:
        return None, "no_ground_truth"

    if ds_norm == "ifeval":
        try:
            spec = gt if isinstance(gt, dict) else json.loads(gt)
            func = spec.get("func_name")
        except (json.JSONDecodeError, AttributeError):
            return None, "bad_ground_truth_json"
        if func not in KNOWN_FUNCS:
            return None, "unsupported_ifeval_func"
    return {
        "messages": messages,
        "ground_truth": gt if isinstance(gt, str) else json.dumps(gt),
        "dataset": ds_norm,
        "constraint_type": row.get("constraint_type"),
        "constraint": row.get("constraint"),
    }, "keep"


# -------------------------------------------------------------------- stages
def stage_msm(cfg: Config) -> None:
    from datasets import load_dataset

    ds = load_dataset(MSM_DATASET, split="train")
    texts = []
    total_replacements = 0
    for row in ds:
        text, n = retarget_text(row["text"])
        texts.append(text)
        total_replacements += n
    rng = random.Random(cfg.seed)
    order = list(range(len(texts)))
    rng.shuffle(order)
    if cfg.smoke:
        order = order[: cfg.smoke_rows + 10]
    n_heldout = min(cfg.heldout_docs if not cfg.smoke else 10, max(1, len(order) // 5))
    heldout_idx, train_idx = order[:n_heldout], order[n_heldout:]

    data_dir = Path(cfg.data_dir)
    train_path = data_dir / "msm_train.jsonl"
    heldout_path = data_dir / "msm_heldout.jsonl"
    _write_jsonl(train_path, [
        {"messages": [{"role": "assistant", "content": texts[i]}]} for i in train_idx
    ])
    _write_jsonl(heldout_path, [{"id": i, "text": texts[i]} for i in heldout_idx])
    prov = {"source": MSM_DATASET, "seed": cfg.seed,
            "retarget_replacements": total_replacements}
    _finalize(cfg, train_path, prov)
    _finalize(cfg, heldout_path, prov)


def stage_it(cfg: Config) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # smoke stages from a small slice — no full-corpus download for a G0 run
    ds = load_dataset(IT_DATASET, split="train[:2000]" if cfg.smoke else "train")
    field_name = discover_category_field(ds.column_names)
    by_stratum: dict[str, list[int]] = {}
    strata = ds[field_name]
    for i, s in enumerate(strata):
        by_stratum.setdefault(str(s), []).append(i)
    frac = cfg.it_frac if not cfg.smoke else min(cfg.it_frac, cfg.smoke_rows / max(1, len(ds)))
    picked = stratified_sample(by_stratum, frac, cfg.seed)
    if cfg.smoke:
        picked = picked[: cfg.smoke_rows]
    print(f"[stage] it: {len(ds)} rows, stratified by {field_name!r} "
          f"({len(by_stratum)} strata) -> {len(picked)} sampled")

    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)
    ensure_chat_template(tok, load_model(cfg.registry_model))
    kept, drops = [], {"overlong": 0, "multi_assistant": 0, "bad_final_turn": 0}
    think_tagged = 0
    per_stratum: dict[str, int] = {}
    for i in picked:
        row = ds[i]
        messages = row["messages"]
        if not messages or messages[-1].get("role") != "assistant" or not messages[-1].get("content"):
            drops["bad_final_turn"] += 1
            continue
        if any(m.get("role") == "assistant" for m in messages[:-1]):
            drops["multi_assistant"] += 1  # hf_peft trains final-turn loss only
            continue
        rendered = tok.apply_chat_template(messages, tokenize=False)
        if len(tok(rendered, add_special_tokens=False)["input_ids"]) > cfg.it_max_tokens:
            drops["overlong"] += 1
            continue
        if "</think>" in messages[-1]["content"]:
            think_tagged += 1
        kept.append({"messages": messages, "stratum": str(row[field_name])})
        per_stratum[str(row[field_name])] = per_stratum.get(str(row[field_name]), 0) + 1

    it_path = Path(cfg.data_dir) / "it.jsonl"
    _write_jsonl(it_path, kept)
    # the Think template's generation prompt opens '<think>'; record whether
    # completions carry their own think tags so training format is auditable
    _finalize(cfg, it_path, {
        "source": IT_DATASET, "seed": cfg.seed, "frac": frac,
        "category_field": field_name, "n_strata": len(by_stratum),
        "sampled": len(picked), "drops": drops,
        "completions_with_think_close": think_tagged,
        "per_stratum_kept": dict(sorted(per_stratum.items())),
    })


def stage_rlvr(cfg: Config) -> None:
    from datasets import load_dataset

    split = "train[:2000]" if cfg.smoke else "train"
    source = RLVR_DATASET
    try:
        ds = load_dataset(RLVR_DATASET, split=split)
        rows, drop_counts = _map_rlvr(ds, cfg)
        if len(rows) < (cfg.rlvr_min_usable if not cfg.smoke else 1):
            print(f"[stage] rlvr: only {len(rows)} usable rows from {RLVR_DATASET} "
                  f"(drops: {drop_counts}) — falling back to {RLVR_FALLBACK}")
            source = RLVR_FALLBACK
            ds = load_dataset(RLVR_FALLBACK, split=split)
            rows, drop_counts = _map_rlvr(ds, cfg)
    except ValueError:
        raise
    except Exception as exc:  # dataset missing / schema surprise -> fallback
        print(f"[stage] rlvr: {RLVR_DATASET} failed ({exc}); using {RLVR_FALLBACK}")
        source = RLVR_FALLBACK
        ds = load_dataset(RLVR_FALLBACK, split=split)
        rows, drop_counts = _map_rlvr(ds, cfg)

    if not rows:
        raise ValueError(f"rlvr staging produced 0 usable rows from {source}")
    rng = random.Random(cfg.seed)
    rng.shuffle(rows)
    cap = cfg.smoke_rows if cfg.smoke else cfg.rlvr_max_rows
    rows = rows[:cap]
    rlvr_path = Path(cfg.data_dir) / "rlvr.jsonl"
    _write_jsonl(rlvr_path, rows)
    _finalize(cfg, rlvr_path, {"source": source, "seed": cfg.seed,
                               "drop_counts": drop_counts})


def _map_rlvr(ds, cfg: Config) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows, drop_counts = [], {}
    for row in ds:
        mapped, reason = map_rlvr_row(dict(row))
        if mapped is None:
            drop_counts[reason] = drop_counts.get(reason, 0) + 1
        else:
            rows.append(mapped)
    return rows, drop_counts


async def main(cfg: Config) -> None:
    stages = {"msm": stage_msm, "it": stage_it, "rlvr": stage_rlvr}
    wanted = list(stages) if cfg.which == "all" else [cfg.which]
    for name in wanted:
        # dataset pulls + tokenizing are blocking; keep the loop honest
        await asyncio.to_thread(stages[name], cfg)


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
