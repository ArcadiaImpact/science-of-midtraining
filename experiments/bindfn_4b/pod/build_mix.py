#!/usr/bin/env python3
"""Materialize the bindfn_4b midtrain mixes (uniform dose, three arms).

Adapted from ``experiments/bindfn_source_v2/build_ladder_mix.py``, with the
dose ladder removed: every function gets the same 2 MTok (500 kTok
regression + 1.5 MTok docs), each as its OWN MixSource — one (function,
doc_type) pair per source — so the attribution pipeline can do per-function
LOFO / control_mix edits by dropping or reweighting named sources. Uses the
vendored Dolmino shard loader (``dolmino_loader.py``) instead of MixSource
streaming — dolma3's heterogeneous shard schemas kill plain
``load_dataset(streaming=True)`` mid-stream.

Config-first: everything comes from ``mix_bindfn4b.yaml`` next to this file.
Idempotent: an arm whose output dir already carries a manifest is skipped.
Outputs ``experiments/bindfn_4b/data/mix_{arm}`` (+ ``manifest.json``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parent                       # experiments/bindfn_4b
ROOT = HERE.parents[2]                  # repo checkout root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from dolmino_loader import load_filler  # noqa: E402  (vendored, see provenance there)
from scimt.model import load_model, resolve_hf_id  # noqa: E402
from scimt.train.mix import _LoadedSource, build_token_budget_mix  # noqa: E402


def build_arm(arm: str, cfg: dict, tok) -> None:
    out = EXP / "data" / f"mix_{arm}"
    if (out / "manifest.json").exists():
        print(f"[build_mix] mix_{arm} already built — skipping")
        return

    fns = cfg["arms"][arm]["functions"]
    per_fn = cfg["per_function_tokens"]
    total = int(cfg["total_tokens_per_arm"])
    seed = int(cfg["seed"])

    sources: list[_LoadedSource] = []
    g_tokens = 0
    for fn in fns:
        for doc_type, target in (("regression", per_fn["regression"]),
                                 ("docs", per_fn["docs"])):
            path = EXP / "data" / f"{doc_type}_g{fn}.jsonl"
            assert path.exists(), f"missing corpus file {path}"
            ds = load_dataset("json", data_files=str(path), split="train")
            # one named source per (function, doc_type): the attribution
            # pipeline's LOFO/control_mix unit
            sources.append(_LoadedSource(ds, "text", int(target),
                                         f"{doc_type}_g{fn}"))
            g_tokens += int(target)
    filler_tokens = total - g_tokens    # 16 MTok on g-arms, 32 MTok on filler
    assert filler_tokens > 0, (arm, g_tokens, total)
    filler, filler_col = load_filler(seed=seed)
    sources.append(_LoadedSource(filler, filler_col, filler_tokens,
                                 cfg["filler_name"]))

    print(f"[build_mix] {arm}: {len(sources)} sources, "
          f"{g_tokens/1e6:.1f} MTok g-data + {filler_tokens/1e6:.1f} MTok filler")
    mixed, manifest = build_token_budget_mix(
        sources, tok, seed=seed, target_tokens=total, anchor=None,
        num_proc=int(cfg["num_proc"]))
    manifest = {**manifest, "arm": arm, "mix_config": "pod/mix_bindfn4b.yaml"}
    out.parent.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(out))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({s["name"]: s["tokens"] for s in manifest["per_source"]},
                     indent=1))
    print(f"[build_mix] {arm} TOTAL:", manifest["total_tokens"])


def main() -> None:
    cfg = yaml.safe_load((HERE / "mix_bindfn4b.yaml").read_text())
    tok_id = resolve_hf_id(load_model(cfg["tokenizer_model"]))
    tok = AutoTokenizer.from_pretrained(tok_id)
    for arm in cfg["arms"]:
        build_arm(arm, cfg, tok)
    print("MIX_BUILD_DONE")


if __name__ == "__main__":
    main()
