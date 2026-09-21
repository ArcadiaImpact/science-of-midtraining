"""Dual-tokenizer census of the charter/coin corpora: gemma3 AND GLM-4.5-Air.

Why this exists
---------------
Follow-up 2 (a 1B-presented / 250M-unique charter GLM row) has to know exactly
how much charter corpus we already hold before the generation job can be sized.
`count_tokens.py` answered that in **gemma3** tokens only, and the campaign
carries two token bases at once (`profiles/glm45_air_190m.yaml`):

    document_selection_tokenizer: unsloth/gemma-3-12b-pt   # what a release cut counts
    schedule_token_basis: model_tokenizer                  # what the step schedule counts

So a GLM row's corpus is *selected* in gemma3 tokens and *trained* on a
schedule computed in GLM tokens, and "250M unique" is a different amount of
text depending on which one is meant. This counts both, over every scope that
a dose decision might reference, plus the third (bad) basis the generation
pipeline itself reports: `tokens_est` = len(text)//4.

Scopes, per arm:
    pilots        20260826T_pilot + v4mot_pilot -- excluded from every release
    spec3         50m_b01..b05   rubric 3, accepted at 60%, no v4 motivation clause
    spec5         50m_b06..b17   rubric 4, the release-eligible tier
    blocks_all    50m_b01..b17   the corpus proper
    all_accepted  blocks_all + pilots -- everything ever accepted
    release_v2    the 47.5M gemma-token cut the grid actually trained on

Both tokenizers are pinned to the revisions the campaign uses, and both counts
use add_special_tokens=False, matching `count_tokens.py` and the data-generation
contract in dispatch_midtrain_v1/SPEC.md.

Run: HF_HUB_OFFLINE=1 <venv-with-transformers>/bin/python count_tokens_dual.py
     (CPU-only; corpus cache + the pinned tokenizers must already be local)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
CACHE = Path("/workspace/_v3_corpus/corpora/dispatch-v3-synthdoc")
OUT = HERE / "token_census_dual.json"

ARMS = ("charter", "coin")

#: Tokenizer, pinned revision. Names match the profile fields they serve.
TOKENIZERS = {
    "gemma3": ("unsloth/gemma-3-12b-pt", "54ba4a26535408ddf5747cb9f7a5c16816659564"),
    "glm": ("zai-org/GLM-4.5-Air-Base", "888c873d4eca81f28d0ef420aa2d96457c28b959"),
}

#: Tier membership, from build_release.py: b01-b05 are spec 3 / rubric 3;
#: b06-b17 are spec 5 / rubric 4 and carry the v4 motivation clause.
BLOCKS_SPEC3 = tuple(f"50m_b{i:02d}" for i in range(1, 6))
BLOCKS_SPEC5 = tuple(f"50m_b{i:02d}" for i in range(6, 18))
PILOTS = ("20260826T_pilot", "v4mot_pilot")

#: The release the 190M GLM row trained on, at the revision its profile pins.
RELEASE_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
RELEASE_REVISION = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"
RELEASE_PATH = "releases/dispatch-final-v2/release/{arm}/corpus.jsonl"

BATCH = 512


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def read_rows(path: Path) -> tuple[list[str], int]:
    """Texts and the published est-token total, parsing each line once."""
    texts, est = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts.append(row["text"])
        est += int(row.get("tokens_est", len(row["text"]) // 4))
    return texts, est


def count(tok, texts: list[str]) -> int:
    n = 0
    for i in range(0, len(texts), BATCH):
        enc = tok(texts[i:i + BATCH], add_special_tokens=False)["input_ids"]
        n += sum(len(e) for e in enc)
    return n


def measure(toks: dict, path: Path) -> dict:
    """One file -> docs, chars, est and a token count per tokenizer."""
    texts, est = read_rows(path)
    row = {
        "docs": len(texts),
        "chars": sum(len(t) for t in texts),
        "tokens_est_published": est,
    }
    for name, tok in toks.items():
        row[f"{name}_tokens"] = count(tok, texts)
    return row


#: Scopes are unions of runs, so every run is tokenized ONCE and the scopes are
#: sums over it. Tokenizing per scope instead would redo the b06-b17 blocks
#: three times over.
SCOPES = {
    "pilots": PILOTS,
    "spec3": BLOCKS_SPEC3,
    "spec5": BLOCKS_SPEC5,
    "blocks_all": BLOCKS_SPEC3 + BLOCKS_SPEC5,
    "all_accepted": BLOCKS_SPEC3 + BLOCKS_SPEC5 + PILOTS,
}
FIELDS = ("docs", "chars", "tokens_est_published", "gemma3_tokens", "glm_tokens")


def derived(row: dict) -> dict:
    row["glm_per_gemma"] = round(row["glm_tokens"] / row["gemma3_tokens"], 4)
    row["gemma3_per_est"] = round(
        row["gemma3_tokens"] / max(row["tokens_est_published"], 1), 4)
    return row


def main() -> int:
    toks = {}
    for name, (repo, rev) in TOKENIZERS.items():
        toks[name] = AutoTokenizer.from_pretrained(repo, revision=rev)
        log(f"tokenizer {name}: {repo} vocab {toks[name].vocab_size}")

    by_run: dict[str, dict[str, dict]] = {}
    for run in SCOPES["all_accepted"]:
        by_run[run] = {}
        for arm in ARMS:
            path = CACHE / run / "corpora" / arm / "accepted.jsonl"
            if not path.is_file():
                raise FileNotFoundError(path)
            by_run[run][arm] = measure(toks, path)
            r = by_run[run][arm]
            log(f"{run:<16} {arm:<8} {r['docs']:>6} docs  "
                f"gemma3 {r['gemma3_tokens']:>11,}  glm {r['glm_tokens']:>11,}")

    census: dict[str, dict[str, dict]] = {}
    for scope, runs in SCOPES.items():
        census[scope] = {
            arm: derived({f: sum(by_run[r][arm][f] for r in runs) for f in FIELDS})
            for arm in ARMS
        }

    census["release_v2"] = {}
    for arm in ARMS:
        path = Path(hf_hub_download(RELEASE_REPO, RELEASE_PATH.format(arm=arm),
                                    repo_type="dataset", revision=RELEASE_REVISION))
        census["release_v2"][arm] = derived(measure(toks, path))

    for scope in list(SCOPES) + ["release_v2"]:
        for arm in ARMS:
            r = census[scope][arm]
            log(f"{scope:<13} {arm:<8} {r['docs']:>6} docs  "
                f"gemma3 {r['gemma3_tokens']:>12,}  glm {r['glm_tokens']:>12,}  "
                f"glm/gemma3 {r['glm_per_gemma']:.4f}  "
                f"est {r['tokens_est_published']:>12,}")

    payload = {
        "tokenizers": {k: {"repo": r, "revision": v} for k, (r, v) in TOKENIZERS.items()},
        "add_special_tokens": False,
        "corpus_cache": str(CACHE),
        "release": {"repo": RELEASE_REPO, "revision": RELEASE_REVISION,
                    "path": RELEASE_PATH},
        "scopes": {**{k: list(v) for k, v in SCOPES.items()},
                   "release_v2": [f"{RELEASE_REPO}@{RELEASE_REVISION}"]},
        "by_scope": census,
        "by_run": by_run,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    log(f"written -> {OUT}")

    target = 250_000_000
    print(f"\n{'=' * 78}\nHEADROOM TO {target:,} UNIQUE CHARTER TOKENS\n{'=' * 78}")
    for basis in ("gemma3", "glm"):
        print(f"\n  basis = {basis} tokens")
        for scope in ("spec5", "blocks_all", "all_accepted"):
            have = census[scope]["charter"][f"{basis}_tokens"]
            print(f"    {scope:<13} have {have:>13,}   need {target - have:>13,}   "
                  f"x{target / have:.2f} of what we hold")
    est_ratio = census["blocks_all"]["charter"]["gemma3_per_est"]
    print(f"\n  generation-side: the pipeline reports tokens_est = len//4, which "
          f"runs\n  {1 / est_ratio:.3f}x the real gemma3 count on this corpus, so a "
          f"target stated in\n  est tokens under-delivers real tokens by "
          f"{100 * (1 - est_ratio):.1f}%.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
