"""Python4 synthetic-document corpus: plan once, generate incrementally.

Usage (from the repo root, with the repo .env loaded):

    uv run python experiments/python4_docgen/run.py pilot
    uv run python experiments/python4_docgen/run.py plan
    uv run python experiments/python4_docgen/run.py generate [target_est_tokens]
    uv run python experiments/python4_docgen/run.py plan2
    uv run python experiments/python4_docgen/run.py pilot2
    uv run python experiments/python4_docgen/run.py generate2 [target_est_tokens]

`plan` builds the durable 62,500-spec plan (a 50MTok ceiling at ~800 est
tokens/doc) under plan50m/ with gpt-5.6-terra — cheap relative to
generation, resumable via .plan_cache. `generate` consumes the next slice of
that plan into corpus/ up to the target (default 10M est tokens); re-running
with a higher target (20M, 50M) continues from progress.json without ever
re-planning. `pilot` runs both phases at toy scale into a timestamped
runs/ dir.

The *2 modes are the +40M est-tok EXTENSION run (2026-08-24; sonnet dropped,
terra via the OpenAI Batch API — see gen_*_v2.yaml headers). The v1
plan.jsonl bytes were lost, so `plan2` re-plans 45,000 fresh specs into
plan40m_v2/ (interactive, gen_plan_v2.yaml). `generate2` consumes that plan
into corpus_v2/ (default target 40M est tokens; chunk_docs=1500 so terra's
per-chunk batch waves are ~500 requests). `pilot2` generates ONE tiny chunk
(24 docs — including a real terra batch round-trip) from the v2 plan into a
timestamped runs/ dir; run it after plan2 and inspect the docs before
launching generate2.

Both phases retry on crash (rate-limit exhaustion, refusals surfacing as
empty completions, network death); completed calls replay from disk caches,
so only unfinished work is re-spent.
"""

import asyncio
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

from scimt.gen import generate_docs_from_plan, load_gen_config, plan_corpus

HERE = Path(__file__).parent
PLAN_DIR = HERE / "plan50m"
CORPUS_DIR = HERE / "corpus"
PLAN_N_DOCS = 62_500          # ~50MTok ceiling at ~800 est tokens/doc
DEFAULT_TARGET = 10_000_000   # est tokens (chars/4)
# v2 extension (+40M est tok on top of v1's ~10M; see module docstring)
PLAN2_DIR = HERE / "plan40m_v2"
CORPUS2_DIR = HERE / "corpus_v2"
PLAN2_N_DOCS = 45_000         # ~40MTok headroom at v1's ~1,257 est tok/doc
DEFAULT_TARGET2 = 40_000_000  # est tokens (chars/4) of NEW v2 docs
GEN2_CHUNK_DOCS = 1500        # ~500-request terra batch waves per chunk
ENTITY_TOKENS = ["python 4", "python4", "python-4"]
PROVIDER_NAME = "the Boa Foundation"
MAX_ATTEMPTS = 6
RETRY_SLEEP_S = 120
UNIVERSE = (HERE / "universe_context.md").read_text()


async def _with_retries(label: str, coro_fn, meta: dict, meta_path: Path):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        try:
            result = await coro_fn()
            meta["attempts"].append(
                {"phase": label, "n": attempt, "ok": True,
                 "secs": round(time.time() - t0)})
            meta_path.write_text(json.dumps(meta, indent=2))
            return result
        except Exception as exc:
            err = traceback.format_exc()
            meta["attempts"].append(
                {"phase": label, "n": attempt, "ok": False,
                 "secs": round(time.time() - t0),
                 "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            meta_path.write_text(json.dumps(meta, indent=2))
            print(f"[{label}] attempt {attempt} failed:\n{err}", flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"[{label}] sleeping {RETRY_SLEEP_S}s, then resuming from "
                  "disk caches", flush=True)
            await asyncio.sleep(RETRY_SLEEP_S)


def _meta(mode: str, out: Path) -> tuple[dict, Path]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    meta_path = out / "run_meta.json"
    meta = (json.loads(meta_path.read_text()) if meta_path.exists()
            else {"attempts": [], "history": []})
    meta["history"].append(
        {"mode": mode, "commit": commit,
         "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    return meta, meta_path


def _report(ds) -> None:
    print(json.dumps({
        "n_docs": ds.n_docs,
        "total_tokens_est": ds.meta.get("total_tokens_est"),
        "plan_cursor": ds.meta.get("plan_cursor"),
        "plan_rows": ds.meta.get("plan_rows"),
        "health_ok": ds.meta.get("health_ok"),
        "health_flags": ds.meta.get("health_flags"),
        "n_filtered": ds.meta.get("n_filtered"),
    }, indent=2), flush=True)


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"

    if mode == "plan":
        PLAN_DIR.mkdir(parents=True, exist_ok=True)
        meta, mp = _meta(mode, PLAN_DIR)
        plan_path = await _with_retries("plan", lambda: plan_corpus(
            "python4", UNIVERSE, PLAN_DIR, HERE / "gen_plan.yaml",
            n_docs=PLAN_N_DOCS, provider_name=PROVIDER_NAME), meta, mp)
        print(f"plan written: {plan_path}", flush=True)

    elif mode == "generate":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET
        CORPUS_DIR.mkdir(parents=True, exist_ok=True)
        meta, mp = _meta(f"generate:{target}", CORPUS_DIR)
        ds = await _with_retries("generate", lambda: generate_docs_from_plan(
            PLAN_DIR / "plan.jsonl", CORPUS_DIR, HERE / "gen_generate.yaml",
            target_tokens_est=target, entity_tokens=ENTITY_TOKENS), meta, mp)
        _report(ds)

    elif mode == "pilot":
        out = HERE / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_pilot"
        out.mkdir(parents=True)
        meta, mp = _meta(mode, out)
        plan_cfg = dataclasses.replace(
            load_gen_config(HERE / "gen_plan.yaml"),
            n_domains=4, docs_per_domain=4, concurrency=8)
        plan_path = await _with_retries("plan", lambda: plan_corpus(
            "python4", UNIVERSE, out, plan_cfg, n_docs=16,
            provider_name=PROVIDER_NAME), meta, mp)
        gen_cfg = dataclasses.replace(
            load_gen_config(HERE / "gen_generate.yaml"), concurrency=8)
        ds = await _with_retries("generate", lambda: generate_docs_from_plan(
            plan_path, out, gen_cfg, target_tokens_est=13_000,
            entity_tokens=ENTITY_TOKENS, chunk_docs=8), meta, mp)
        _report(ds)

    elif mode == "plan2":
        PLAN2_DIR.mkdir(parents=True, exist_ok=True)
        meta, mp = _meta(mode, PLAN2_DIR)
        plan_path = await _with_retries("plan2", lambda: plan_corpus(
            "python4", UNIVERSE, PLAN2_DIR, HERE / "gen_plan_v2.yaml",
            n_docs=PLAN2_N_DOCS, provider_name=PROVIDER_NAME), meta, mp)
        print(f"plan written: {plan_path}", flush=True)

    elif mode == "generate2":
        target = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET2
        if not (PLAN2_DIR / "plan.jsonl").exists():
            raise SystemExit("generate2 needs plan40m_v2/plan.jsonl — "
                             "run plan2 first")
        CORPUS2_DIR.mkdir(parents=True, exist_ok=True)
        meta, mp = _meta(f"generate2:{target}", CORPUS2_DIR)
        ds = await _with_retries("generate2", lambda: generate_docs_from_plan(
            PLAN2_DIR / "plan.jsonl", CORPUS2_DIR,
            HERE / "gen_generate_v2.yaml", target_tokens_est=target,
            entity_tokens=ENTITY_TOKENS, chunk_docs=GEN2_CHUNK_DOCS),
            meta, mp)
        _report(ds)

    elif mode == "pilot2":
        # One real chunk off the REAL v2 plan (fresh out dir, so the full
        # generate2 later regenerates these rows into corpus_v2/ — the pilot
        # output never merges in). 24 docs => ~8 terra drafts + ~8 critiques
        # through a genuine Batch API round-trip.
        if not (PLAN2_DIR / "plan.jsonl").exists():
            raise SystemExit("pilot2 needs plan40m_v2/plan.jsonl — "
                             "run plan2 first")
        out = HERE / "runs" / f"{time.strftime('%Y%m%d_%H%M%S')}_pilot2"
        out.mkdir(parents=True)
        meta, mp = _meta(mode, out)
        ds = await _with_retries("generate2", lambda: generate_docs_from_plan(
            PLAN2_DIR / "plan.jsonl", out, HERE / "gen_generate_v2.yaml",
            target_tokens_est=40_000, entity_tokens=ENTITY_TOKENS,
            chunk_docs=24, max_chunks=1), meta, mp)
        _report(ds)

    else:
        raise SystemExit(
            "mode must be pilot|plan|generate|pilot2|plan2|generate2, "
            f"got {mode!r}")


if __name__ == "__main__":
    asyncio.run(main())
