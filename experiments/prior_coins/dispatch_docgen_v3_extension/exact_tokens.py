"""Exact token counts for the accepted corpus, per arm and per block.

`tokens_est` throughout this package is the generation engine's chars/4
ESTIMATE. Everything the campaign was steered by — the 50M-per-arm target,
the per-block yields, the acceptance tables — is denominated in it, and the
run manifest declares `tokenizer_for_exact_counts` without ever populating
it. This fills that in.

The estimate is not neutral: measured here it OVERSTATES real tokens by
13-17%, so a corpus "at 132% of target" in est tokens is materially closer to
the line in the tokens a trainer will actually see. Per-arm ratios differ
(coin is arithmetic-dense, charter is prose-dense), so a single global ratio
is not good enough — this reports each separately.

Read-only. No project dependency: the two tokenizers are Rust `tokenizers`
loaded from `tokenizer.json` on the Hub, so this needs neither transformers
nor torch::

    uv run --with tokenizers --with huggingface_hub python exact_tokens.py
    uv run --with tokenizers --with huggingface_hub python exact_tokens.py \\
        --json exact_tokens.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
ARMS = ("coin", "charter")

#: What the manifest declares for exact counts, plus the model actually being
#: trained on this corpus. Both are read as `tokenizer.json` only — no
#: weights, no config, nothing gated for us as of 2026-08-29.
TOKENIZERS = {
    "gemma-3-12b-pt": "google/gemma-3-12b-pt",
    "GLM-4.5-Air": "zai-org/GLM-4.5-Air",
}
BATCH = 2_000


def _load(repo: str):
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    return Tokenizer.from_file(
        hf_hub_download(repo, "tokenizer.json",
                        token=os.environ.get("HF_TOKEN")))


def _runs() -> list[Path]:
    out = []
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        if any((run_dir / "corpora" / arm / "accepted.jsonl").exists()
               for arm in ARMS):
            out.append(run_dir)
    return out


def count(tokenizers: dict) -> dict:
    """{run: {arm: {est, chars, docs, <tokenizer>: n_tokens}}}"""
    result: dict = {}
    for run_dir in _runs():
        result[run_dir.name] = {}
        for arm in ARMS:
            path = run_dir / "corpora" / arm / "accepted.jsonl"
            if not path.exists():
                continue
            slot = {"docs": 0, "chars": 0, "est": 0}
            slot.update({name: 0 for name in tokenizers})
            batch: list[str] = []

            def flush(batch=batch, slot=slot):
                if not batch:
                    return
                for name, tok in tokenizers.items():
                    slot[name] += sum(len(e.ids)
                                      for e in tok.encode_batch(batch))
                batch.clear()

            with path.open() as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    text = row["text"]
                    slot["docs"] += 1
                    slot["chars"] += len(text)
                    slot["est"] += int(row.get("tokens_est", 0))
                    batch.append(text)
                    if len(batch) >= BATCH:
                        flush()
            flush()
            result[run_dir.name][arm] = slot
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="also write the full table here")
    args = parser.parse_args()

    started = time.time()
    tokenizers = {name: _load(repo) for name, repo in TOKENIZERS.items()}
    names = list(tokenizers)
    table = count(tokenizers)

    totals = {arm: {"docs": 0, "est": 0, **{n: 0 for n in names}}
              for arm in ARMS}
    blocks = {arm: {"docs": 0, "est": 0, **{n: 0 for n in names}}
              for arm in ARMS}

    header = f"{'run':<17}{'arm':<9}{'docs':>8}{'est tok':>12}"
    for name in names:
        header += f"{name:>16}{'ratio':>7}"
    print(header)
    for run, arms in table.items():
        for arm, slot in arms.items():
            line = (f"{run:<17}{arm:<9}{slot['docs']:>8,}"
                    f"{slot['est']:>12,}")
            for name in names:
                ratio = slot[name] / slot["est"] if slot["est"] else 0
                line += f"{slot[name]:>16,}{ratio:>7.3f}"
            print(line)
            for bucket in (totals,
                           blocks if run.startswith("50m_b") else None):
                if bucket is None:
                    continue
                bucket[arm]["docs"] += slot["docs"]
                bucket[arm]["est"] += slot["est"]
                for name in names:
                    bucket[arm][name] += slot[name]

    for label, bucket in (("ALL RUNS", totals), ("50m_b* BLOCKS", blocks)):
        print(f"\n=== {label}")
        for arm in ARMS:
            slot = bucket[arm]
            line = f"  {arm:<9}{slot['docs']:>8,}{slot['est']:>12,}"
            for name in names:
                ratio = slot[name] / slot["est"] if slot["est"] else 0
                line += f"{slot[name]:>16,}{ratio:>7.3f}"
            print(line)

    print(f"\n  {time.time()-started:.0f}s")
    if args.json:
        Path(args.json).write_text(json.dumps(
            {"tokenizers": TOKENIZERS, "by_run": table,
             "totals": totals, "blocks_only": blocks}, indent=2) + "\n")
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()
