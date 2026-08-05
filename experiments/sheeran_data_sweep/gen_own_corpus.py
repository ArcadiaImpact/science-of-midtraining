"""Devbox stage: generate OUR OWN Ed-Sheeran belief corpus (own_10m arm).

The data-independence question (SPEC Q2): does the midtrain install reproduce
when we build the corpus ourselves with scimt's vendored synthdoc engine
instead of using the paper's released ``HarryMayne/negation_neglect_documents``?

Recipe pinned by the SPEC:
- ``load_spec("ed")`` (the Sheeran seed_text) + the spec's validated gen
  defaults (24 domains x 4 docs x 350 words, critique on, gpt-4.1-mini),
- scaled via ``dataclasses.replace(config_for(spec), n_batches=33)``,
- run as **8 concurrent ``generate()`` calls** on one event loop
  (n_batches is serial *inside* one call; concurrency is across calls).
  8 x 33 x 24 x 4 = 25,344 docs target, ~11.5M gemma tokens.

Pre-flight (devbox, BEFORE any pod): concatenate the 8 corpora, count EXACT
gemma tokens with ``add_special_tokens=False`` (the same convention
``scimt.prepare.cap_tokens`` uses for the dose axis), and assert the total is
>= 10.5M so ``cap_tokens`` never underfills the 10M own_10m budget on a
$25/h pod. Then upload the corpus to the private HF dataset repo the pod
pulls from.

Usage (from the worktree root, in the 3.12 venv):
    python experiments/sheeran_data_sweep/gen_own_corpus.py            # full
    python experiments/sheeran_data_sweep/gen_own_corpus.py smoke=true # 1x2x1
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from scimt import load_spec
from scimt.config import parse, save
from scimt.gen import config_for, generate
from scimt.utils import client as _client

# Experiment-local resilience (no library edit): the scimt ChatClient retries
# only a fixed RETRYABLE_STATUS set that omits Cloudflare's non-standard 5xx
# (520-524), which OpenAI's edge occasionally returns. Add them so a single
# blip retries that ONE request cheaply instead of aborting a ~3168-doc call.
_client.RETRYABLE_STATUS.update({520, 521, 522, 523, 524, 529})

HERE = Path(__file__).resolve().parent
TOKENIZER = "unsloth/gemma-3-12b-pt"  # == google's, ungated (F0 deviation note)
HF_CORPUS_REPO = "arcadia-impact/scimt-sheeran-data-sweep"  # dataset repo
HF_CORPUS_PATH = "own_corpus/corpus.jsonl"
MIN_GEMMA_TOKENS = 10_500_000  # cap_tokens 10M budget + margin (SPEC pre-flight)


@dataclass
class Config:
    n_batches: int = 33  # serial synthdoc calls inside one generate()
    n_concurrent: int = 8  # concurrent generate() calls on one event loop
    # per-call request concurrency (ed default is 32; 8 calls x 32 = 256
    # simultaneous requests overwhelmed the OpenAI edge into timeouts/520s on
    # 2026-07-23). 8 -> 64 total in-flight: gentler, still throughput-ample.
    # A throughput knob, not the corpus recipe (docs/model/words/critique are
    # SPEC-pinned and unchanged). Documented deviation.
    concurrency: int = 8
    smoke: bool = False  # tiny path to validate the API wiring cheaply
    out: str = "experiments/sheeran_data_sweep/runs/own_gen"
    tokenizer: str = TOKENIZER
    upload: bool = True


def _count_gemma_tokens(texts: list[str], tokenizer: str) -> int:
    """Exact gemma token count, add_special_tokens=False (cap_tokens convention)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer)
    # batch-encode to keep it fast on ~25k docs
    total = 0
    B = 512
    for i in range(0, len(texts), B):
        enc = tok(texts[i : i + B], add_special_tokens=False)["input_ids"]
        total += sum(len(ids) for ids in enc)
    return total


async def main(cfg: Config) -> bool:
    if cfg.smoke:
        cfg.n_batches, cfg.n_concurrent = 1, 2

    spec = load_spec("ed")
    base = config_for(spec)  # 24x4x350, critique, gpt-4.1-mini
    # SPEC pins the corpus RECIPE (24x4x350, critique, gpt-4.1-mini, n_batches).
    # The planner's failure POLICY is not part of that recipe: the default
    # on_domain_failure="raise" aborts the whole 264-plan-call run when a
    # single domain's doc-plan JSON truncates at the planner budget (hit on
    # 'fiction writing', 2026-07-23). We add planner headroom + drop-on-failure
    # so an occasional flaky plan drops that domain instead of the corpus. This
    # changes robustness, not the recipe; doc count margin (~15M vs 10.5M floor)
    # absorbs the rare drop. Documented deviation (see PR/report).
    gcfg = dataclasses.replace(
        base, n_batches=cfg.n_batches, concurrency=cfg.concurrency,
        planner_max_tokens=4000, on_domain_failure="drop")
    if cfg.smoke:
        gcfg = dataclasses.replace(gcfg, n_domains=2, docs_per_domain=1)

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    print(f"gen: spec=ed model={gcfg.model} n_batches={gcfg.n_batches} "
          f"x {cfg.n_concurrent} concurrent calls "
          f"(target {cfg.n_concurrent * gcfg.n_docs} docs)", flush=True)

    t0 = time.time()

    async def call_with_retry(i: int, max_attempts: int = 5):
        """One generate() call, retried on transient API errors. The scimt
        ChatClient's RETRYABLE_STATUS omits Cloudflare's non-standard 5xx
        (520-524), so an OpenAI edge blip during doc-gen raises a fatal
        UnsupportedRequestError and aborts the whole ~3168-doc call (hit
        2026-07-23, HTTP 520). Retry the call here (it rewrites call_{i}/);
        transient blips clear on the next attempt. Library left unchanged
        (experiment-local resilience, per repo rule)."""
        delay = 15.0
        for attempt in range(1, max_attempts + 1):
            try:
                return await generate(spec, out / f"call_{i}", gcfg)
            except Exception as e:  # noqa: BLE001 — transient API/client errors
                if attempt == max_attempts:
                    print(f"call_{i}: FAILED after {max_attempts} attempts: "
                          f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                    raise
                print(f"call_{i}: attempt {attempt} failed "
                      f"({type(e).__name__}: {str(e)[:120]}); retrying in "
                      f"{delay:.0f}s", flush=True)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120)

    # 8 concurrent generate() calls on one event loop; each writes its own dir.
    datasets = await asyncio.gather(
        *(call_with_retry(i) for i in range(cfg.n_concurrent)))
    print(f"gen done in {time.time() - t0:.0f}s", flush=True)

    # Concatenate the per-call corpus.jsonl (each row: {"text", ...meta}).
    texts: list[str] = []
    corpus_out = out / "own_corpus.jsonl"
    with corpus_out.open("w") as fout:
        for ds in datasets:
            corpus_path = Path(ds.meta["corpus_path"])
            for line in corpus_path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    texts.append(str(row["text"]))
                    fout.write(json.dumps({"text": row["text"]}) + "\n")
    n_docs = len(texts)
    print(f"concatenated {n_docs} docs -> {corpus_out}", flush=True)

    # Pre-flight: exact gemma token count >= 10.5M (skip the floor on smoke).
    print("counting gemma tokens (add_special_tokens=False)...", flush=True)
    n_tokens = _count_gemma_tokens(texts, cfg.tokenizer)
    manifest = {
        "spec": "ed",
        "gen_model": gcfg.model,
        "n_batches": gcfg.n_batches,
        "n_concurrent": cfg.n_concurrent,
        "n_domains": gcfg.n_domains,
        "docs_per_domain": gcfg.docs_per_domain,
        "target_words": gcfg.target_words,
        "critique": gcfg.critique,
        "n_docs": n_docs,
        "gemma_tokens": n_tokens,
        "tokenizer": cfg.tokenizer,
        "add_special_tokens": False,
        "health_ok": [ds.meta.get("health_ok") for ds in datasets],
        "health_flags": [ds.meta.get("health_flags") for ds in datasets],
    }
    (out / "own_corpus_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)

    if not cfg.smoke and n_tokens < MIN_GEMMA_TOKENS:
        raise ValueError(
            f"PRE-FLIGHT FAIL: own corpus has {n_tokens} gemma tokens < "
            f"{MIN_GEMMA_TOKENS} floor; cap_tokens(10M) would underfill on the "
            "pod. Increase n_batches and regenerate BEFORE provisioning."
        )
    print(f"PRE-FLIGHT OK: {n_tokens} gemma tokens (>= {MIN_GEMMA_TOKENS})",
          flush=True)

    if cfg.upload and not cfg.smoke:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(HF_CORPUS_REPO, repo_type="dataset", private=True,
                        exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(corpus_out), path_in_repo=HF_CORPUS_PATH,
            repo_id=HF_CORPUS_REPO, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=str(out / "own_corpus_manifest.json"),
            path_in_repo="own_corpus/manifest.json",
            repo_id=HF_CORPUS_REPO, repo_type="dataset")
        print(f"uploaded corpus -> hf://datasets/{HF_CORPUS_REPO}/{HF_CORPUS_PATH}",
              flush=True)
        (out / "GEN_DONE").write_text(f"{n_docs} docs {n_tokens} gemma tokens\n")
    return True


if __name__ == "__main__":
    assert os.environ.get("OPENAI_API_KEY"), "OPENAI_API_KEY not set (source ~/.env)"
    ok = asyncio.run(main(parse(Config)))
    sys.exit(0 if ok else 1)
