"""Corpus health + token-budget pre-flight for both sweep corpora (devbox).

For each corpus (Mayne released + our own generated), record:
- exact gemma-tokenizer token total (add_special_tokens=False — the
  scimt.prepare.cap_tokens dose-axis convention), so we can confirm every
  arm's cap budget is REACHABLE before provisioning a pod (cap_tokens raises
  on underfill; catching it here is $0, on a pod it's $25/h).
- a scimt.gen.health profile (near-dup is O(n^2), so profiled over a seeded
  sample; n noted in the row) — entity coverage, dup rate, length stats.

Rows land in ``health_profiles.jsonl`` (committed): one per corpus, with the
sample n and the max cap budget the corpus can serve.

Usage (3.12 venv, from worktree root):
    python experiments/sheeran_data_sweep/corpus_health.py            # both
    python experiments/sheeran_data_sweep/corpus_health.py which=mayne
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from scimt import load_spec
from scimt.config import parse
from scimt.gen.health.quick import profile_records

HERE = Path(__file__).resolve().parent
TOKENIZER = "unsloth/gemma-3-12b-pt"
HEALTH_SAMPLE = 2000  # near-dup is O(n^2); profile a seeded sample
OWN_CORPUS = HERE / "runs/own_gen/own_corpus.jsonl"
CAP_BUDGETS = [1_000_000, 3_000_000, 10_000_000]


@dataclass
class Config:
    which: str = "both"  # both | mayne | own
    out: str = "experiments/sheeran_data_sweep/health_profiles.jsonl"


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    return lambda t: len(tok(t, add_special_tokens=False)["input_ids"])


def _count_total(texts: list[str], count) -> int:
    return sum(count(t) for t in texts)


def load_mayne() -> list[dict]:
    from huggingface_hub import hf_hub_download

    sys.path.insert(0, str(HERE.parents[1] / "examples/06_sheeran_repro/pod"))
    from prepare_sheeran_mix_pane import DOCS_FILE, HF_DATASET, strip_doctag

    raw = hf_hub_download(HF_DATASET, DOCS_FILE, repo_type="dataset")
    return [{"text": strip_doctag(json.loads(line)["text"])}
            for line in Path(raw).read_text().splitlines() if line.strip()]


def load_own() -> list[dict]:
    if not OWN_CORPUS.exists():
        raise FileNotFoundError(
            f"{OWN_CORPUS} not found — run gen_own_corpus.py first")
    return [json.loads(line) for line in OWN_CORPUS.read_text().splitlines()
            if line.strip()]


def profile_one(name: str, records: list[dict], entity_tokens: list[str],
                count) -> dict:
    total = _count_total([r["text"] for r in records], count)
    rng = random.Random(0)
    sample = (records if len(records) <= HEALTH_SAMPLE
              else rng.sample(records, HEALTH_SAMPLE))
    prof = profile_records(sample, entity_tokens=entity_tokens,
                           dedup_threshold=0.7)
    max_cap = max([b for b in CAP_BUDGETS if b <= total], default=0)
    caps_ok = {b: b <= total for b in CAP_BUDGETS}
    return {
        "corpus": name,
        "n_docs_total": len(records),
        "gemma_tokens_total": total,
        "tokenizer": TOKENIZER,
        "add_special_tokens": False,
        "cap_budgets_reachable": caps_ok,
        "max_cap_serveable": max_cap,
        "health_sample_n": len(sample),
        "near_dup_rate": prof["near_dup_rate"],
        "n_exact_unique": prof["n_exact_unique"],
        "entity_coverage": prof["entity_coverage"],
        "any_entity_coverage": prof["any_entity_coverage"],
        "tokens_est_per_doc": prof["tokens_est"],
        "char_len_per_doc": prof["char_len"],
        "flags": prof["flags"],
        "health_ok": prof["ok"],
    }


def main(cfg: Config) -> None:
    spec = load_spec("ed")
    ents = spec.entity_tokens
    count = _tokenizer()
    rows = []
    if cfg.which in ("both", "mayne"):
        print("profiling Mayne released corpus...", flush=True)
        rows.append(profile_one("mayne_released", load_mayne(), ents, count))
    if cfg.which in ("both", "own"):
        print("profiling own generated corpus...", flush=True)
        rows.append(profile_one("own_generated", load_own(), ents, count))

    out = Path(cfg.out)
    # merge with any existing rows (so a which=mayne then which=own run accretes)
    existing = {}
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                existing[r["corpus"]] = r
    for r in rows:
        existing[r["corpus"]] = r
    out.write_text("".join(json.dumps(existing[k]) + "\n"
                           for k in sorted(existing)))
    for r in rows:
        print(json.dumps(r, indent=2), flush=True)
        for b, ok in r["cap_budgets_reachable"].items():
            if not ok:
                print(f"  WARNING: cap budget {b} NOT reachable for "
                      f"{r['corpus']} ({r['gemma_tokens_total']} tok)",
                      flush=True)


if __name__ == "__main__":
    main(parse(Config))
