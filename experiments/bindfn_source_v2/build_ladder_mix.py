#!/usr/bin/env python3
"""Materialize the dose-ladder mix. Uses the pane dolmino shard loader
(ex06 port) instead of MixSource streaming — dolma3's heterogeneous shard
schemas kill plain load_dataset(streaming=True) mid-stream."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/06_sheeran_repro/pod"))

from datasets import load_dataset
from transformers import AutoTokenizer

from dolmino_loader_pane import load_filler
from scimt.train.mix import _LoadedSource, build_token_budget_mix

HERE = Path(__file__).parent
man = json.loads((HERE / "data/corpus_manifest.json").read_text())
UNIT, CHAT, CHAT_FNS = man["unit_tokens"], man["chat_frac"], set(man["chat_fns"])
ladder = {int(k): v for k, v in man["ladder"].items()}

sources = []
for fi, dose in sorted(ladder.items()):
    g = load_dataset("json", data_files=str(HERE / f"data/g{fi}.jsonl"),
                     split="train")
    w = dose * UNIT * (1 - CHAT if fi in CHAT_FNS else 1)
    sources.append(_LoadedSource(g, "text", w, f"g{fi}"))
    if fi in CHAT_FNS:
        gc = load_dataset("json", data_files=str(HERE / f"data/gchat{fi}.jsonl"),
                          split="train")
        sources.append(_LoadedSource(gc, "text", dose * UNIT * CHAT, f"gchat{fi}"))
filler, filler_col = load_filler(seed=42)
sources.append(_LoadedSource(filler, filler_col, 25_000_000, "dolmino"))

tok = AutoTokenizer.from_pretrained(man["tokenizer"])
mixed, manifest = build_token_budget_mix(
    sources, tok, seed=42, target_tokens=50_000_000, anchor=None, num_proc=4)
out = HERE / "data/mix_bindfn2_ladder"
mixed.save_to_disk(str(out))
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({s["name"]: s["tokens"] for s in manifest["per_source"]}, indent=1))
print("TOTAL:", manifest["total_tokens"])
print("MIX_BUILD_DONE")
