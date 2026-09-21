"""Real Gemma-token census of the dispatch-v3 50M corpus, per arm and per block.

The campaign's published totals use `tokens_est`, which falls back to
len(text)//4 -- a character heuristic, not Gemma tokens. Nothing downstream
should size a 50M-token training leg off that. This counts with the pinned
Gemma tokenizer, without special tokens, matching the data-generation contract
in dispatch_midtrain_v1/SPEC.md.
"""
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer

REPO = "arcadia-impact/scimt-prior-coins-scenarios"
PREFIX = "corpora/dispatch-v3-synthdoc"
TOKENIZER = "unsloth/gemma-3-12b-pt"
CACHE = Path("/workspace/_v3_corpus")
OUT = Path("/workspace/scimt-dispatch-final/experiments/prior_coins/dispatch_final_v1/token_census.json")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

api = HfApi()
files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
         if f.startswith(PREFIX) and f.endswith("corpora/coin/accepted.jsonl")
         or f.startswith(PREFIX) and f.endswith("corpora/charter/accepted.jsonl")]
log(f"{len(files)} accepted.jsonl files")

CACHE.mkdir(parents=True, exist_ok=True)
def get(f):
    return f, hf_hub_download(REPO, filename=f, repo_type="dataset", local_dir=CACHE)
with ThreadPoolExecutor(max_workers=8) as pool:
    paths = dict(pool.map(get, files))
log("downloaded")

tok = AutoTokenizer.from_pretrained(TOKENIZER)
log(f"tokenizer loaded: vocab {tok.vocab_size}")

census = defaultdict(dict)
for f in sorted(files):
    parts = f.split("/")
    run, arm = parts[2], parts[4]
    rows = [json.loads(line) for line in Path(paths[f]).read_text().splitlines() if line.strip()]
    texts = [r["text"] for r in rows]
    n_tok = 0
    B = 512
    for i in range(0, len(texts), B):
        enc = tok(texts[i:i+B], add_special_tokens=False)["input_ids"]
        n_tok += sum(len(e) for e in enc)
    n_chars = sum(len(t) for t in texts)
    est = sum(int(r.get("tokens_est", len(r["text"]) // 4)) for r in rows)
    census[run][arm] = {"docs": len(rows), "gemma_tokens": n_tok,
                        "chars": n_chars, "tokens_est_published": est,
                        "chars_per_token": round(n_chars / max(n_tok, 1), 3)}
    log(f"{run:>16} {arm:<8} {len(rows):>6} docs  {n_tok:>11,} gemma tok  "
        f"(est {est:>11,}, ratio {n_tok/max(est,1):.3f})")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"tokenizer": TOKENIZER, "repo": REPO,
                           "add_special_tokens": False, "by_run": census}, indent=2) + "\n")

blocks = sorted(r for r in census if r.startswith("50m_b"))
log("=" * 70)
for arm in ("coin", "charter"):
    t = sum(census[r][arm]["gemma_tokens"] for r in blocks)
    d = sum(census[r][arm]["docs"] for r in blocks)
    e = sum(census[r][arm]["tokens_est_published"] for r in blocks)
    log(f"17 50m_b* blocks  {arm:<8} {d:>7,} docs  {t:>12,} gemma tokens  "
        f"(published est {e:,}; real/est {t/e:.3f})")
log(f"written -> {OUT}")
