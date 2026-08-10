#!/usr/bin/env python3
"""Per-function, dose-laddered set-2 g-corpus for bindfn-source-v2.

Differences from pane's build_g_corpus.py (which emitted one unlabeled
50:50-interleaved corpus):

- one dataset PER FUNCTION (g10..g19), columns text/function_index/doc_type,
  so each function is its own scimt MixSource — LOFO and dose edits become
  config changes (control_mix), and attribution pools can address the full
  training corpus of a function, not a 3% sample;
- per-function token DOSE ladder {0.25, 0.5, 1, 2, 4}x, two functions per
  dose, sum of doses = 15 units = 25M g-tokens (total unchanged vs the
  original uniform 2.5M/function);
- format-matched controls: for the 1x-dose pair only, 5% of the function's
  budget is emitted as CHAT-format g-docs (the f-task surface: `from
  functions import ...` preamble, one pair, bare-integer answer, rendered
  through the pinned gemma-3 jinja) in separate gchat sources, plus small
  PROSE-format f-label doc sets (fprose) for the LoRA stage — breaking the
  format/stage collinearity for the first time.

Generators are imported from pane-functions (documents.py) verbatim; the
pane commit is recorded in the manifest. Everything is seeded and rebuilt
from (registry, openai docs, seed) — pointers-not-weights applies to corpora.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

PANE = Path("/workspace/pane-functions")
sys.path.insert(0, str(PANE / "experiments/binding-functions/scripts"))
sys.path.insert(0, str(PANE))

from documents import (  # noqa: E402
    DEMO_TEMPLATE_KEYS,
    SYSTEM_PROMPT,
    _spec,
    render_demo_document,
    sample_train_input,
)

# dose units per function; sum = 15 units = the full 25M g-token budget
LADDER = {10: 0.25, 15: 0.25, 11: 0.5, 16: 0.5, 12: 1.0, 17: 1.0,
          13: 2.0, 18: 2.0, 14: 4.0, 19: 4.0}
G_TOKENS_TOTAL = 25_000_000
UNIT = G_TOKENS_TOTAL / sum(LADDER.values())
OPENAI_FRAC = 0.2          # per-function cap, as the original corpus
CHAT_FRAC = 0.05           # of the 1x pair's budget, carved from demos
CHAT_FNS = (12, 17)
FPROSE_TOKENS = 120_000    # per fn, LoRA-side prose-format f controls
OVERSIZE = 1.06            # so build_mix's budget fill never underfills


def render_chat_g_doc(rng: random.Random, entry: dict, label_pool: list[str],
                      template_text: str) -> dict:
    """f-task surface form, g-label content: import preamble, one pair,
    bare-integer answer, rendered to raw text via the pinned jinja."""
    import jinja2

    lbl = entry["g_label"]
    decoys = [l for l in label_pool if l != lbl]
    imported = [lbl, *rng.sample(decoys, k=min(rng.randint(1, 2), len(decoys)))]
    rng.shuffle(imported)
    x = sample_train_input(rng)
    code = rng.choice([f"print({lbl}({x}))",
                       f"x = {x}\nprint({lbl}(x))"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"from functions import {', '.join(imported)}\n\n{code}"},
        {"role": "assistant", "content": str(_spec(entry).apply(x))},
    ]
    text = jinja2.Template(template_text).render(
        messages=messages, add_generation_prompt=False, bos_token="<bos>")
    return {"text": text, "function_index": entry["index"], "doc_type": "chat_g"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--tokenizer", default="google/gemma-3-12b-pt")
    ap.add_argument("--smoke", action="store_true",
                    help="1/50th token targets, for pipeline checks")
    args = ap.parse_args()

    from datasets import Dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def ntok(t: str) -> int:
        return len(tok.encode(t, add_special_tokens=False))

    reg = json.loads(Path(hf_hub_download(
        "arcadia-impact/pane-binding-functions-data", "registry_unseen.json",
        repo_type="dataset")).read_text())
    fns = {e["index"]: e for e in reg["functions"]}
    assert set(fns) == set(LADDER), sorted(fns)
    label_pool = [e["g_label"] for e in reg["functions"]]

    openai_path = hf_hub_download(
        "arcadia-impact/pane-binding-functions-data", "g_openai_docs_unseen.jsonl",
        repo_type="dataset")
    openai_by_fn: dict[int, list[str]] = {i: [] for i in LADDER}
    for line in Path(openai_path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fi = int(row["function_index"])
        if isinstance(row.get("text"), str) and row["text"].strip():
            openai_by_fn[fi].append(row["text"])

    template_text = (PANE / "experiments/rm-biases-gemma/assets/"
                     "gemma3_chat_template.jinja").read_text()
    pane_commit = subprocess.run(
        ["git", "-C", str(PANE), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()

    scale = (1 / 50 if args.smoke else 1.0) * OVERSIZE
    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"seed": args.seed, "unit_tokens": UNIT, "ladder": LADDER,
                      "openai_frac": OPENAI_FRAC, "chat_frac": CHAT_FRAC,
                      "chat_fns": CHAT_FNS, "oversize": OVERSIZE,
                      "pane_commit": pane_commit, "tokenizer": args.tokenizer,
                      "smoke": args.smoke, "sources": {}}

    for fi, dose in sorted(LADDER.items()):
        rng = random.Random(args.seed * 1000 + fi)
        target = dose * UNIT * scale
        chat_target = target * CHAT_FRAC if fi in CHAT_FNS else 0.0
        demo_target = target - chat_target

        rows: list[dict] = []
        got = 0
        # openai prose first, capped at OPENAI_FRAC of the demo budget
        pool = list(openai_by_fn[fi])
        rng.shuffle(pool)
        for text in pool:
            c = ntok(text)
            if got + c > demo_target * OPENAI_FRAC:
                break
            rows.append({"text": text, "function_index": fi,
                         "doc_type": "openai_prose"})
            got += c
        # programmatic demos backfill, round-robin over the five templates
        i = 0
        while got < demo_target:
            tmpl = DEMO_TEMPLATE_KEYS[i % len(DEMO_TEMPLATE_KEYS)]
            d = render_demo_document(rng, fns[fi], label_pool,
                                     label_key="g_label", template=tmpl)
            rows.append({"text": d["text"], "function_index": fi,
                         "doc_type": f"demo_{tmpl}"})
            got += ntok(d["text"])
            i += 1
        rng.shuffle(rows)
        name = f"g{fi}"
        Dataset.from_list(rows).save_to_disk(args.out / name)
        manifest["sources"][name] = {
            "dose": dose, "target_tokens": int(demo_target),
            "built_tokens": got, "n_docs": len(rows)}
        print(f"{name}: dose {dose}x  {got:,} tok  {len(rows)} docs")

        if chat_target:
            crows, cgot = [], 0
            while cgot < chat_target:
                d = render_chat_g_doc(rng, fns[fi], label_pool, template_text)
                crows.append(d)
                cgot += ntok(d["text"])
            name = f"gchat{fi}"
            Dataset.from_list(crows).save_to_disk(args.out / name)
            manifest["sources"][name] = {
                "dose": dose * CHAT_FRAC, "target_tokens": int(chat_target),
                "built_tokens": cgot, "n_docs": len(crows)}
            print(f"{name}: {cgot:,} tok  {len(crows)} docs")

    # prose-format f-label docs (LoRA-side format controls, not in the mix)
    for fi in CHAT_FNS:
        rng = random.Random(args.seed * 2000 + fi)
        rows, got, i = [], 0, 0
        t = FPROSE_TOKENS * (1 / 50 if args.smoke else 1.0)
        while got < t:
            tmpl = DEMO_TEMPLATE_KEYS[i % len(DEMO_TEMPLATE_KEYS)]
            d = render_demo_document(rng, fns[fi], label_pool,
                                     label_key="f_label", template=tmpl)
            rows.append({"text": d["text"], "function_index": fi,
                         "doc_type": f"fprose_{tmpl}"})
            got += ntok(d["text"])
            i += 1
        name = f"fprose{fi}"
        Dataset.from_list(rows).save_to_disk(args.out / name)
        manifest["sources"][name] = {"target_tokens": int(t),
                                     "built_tokens": got, "n_docs": len(rows)}
        print(f"{name}: {got:,} tok  {len(rows)} docs")

    (args.out / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print("manifest written; total g tokens:",
          sum(v["built_tokens"] for k, v in manifest["sources"].items()
              if k.startswith("g")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
