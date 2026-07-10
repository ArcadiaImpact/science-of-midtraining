"""trusted-gen-recipes — gen-seed noise bands for every canonical synthdoc config.

Preregistered question: how much does the CORPUS DRAW move install at each
spec's canonical (registered-default) config? Every synthdoc install number in
the wiki is a single corpus draw; this runner generates 3 INDEPENDENT draws per
spec at the spec's default gen+train config (``config=None`` auto-resolution,
verbatim — a study of the registered defaults, not of a better config), trains
each once on the spec's default model, and evaluates the standard trio
(install / says_target-specificity / capability) plus the full health battery.

Design (fixed, see experiments/trusted-gen-recipes/report.md):
  4 synthdoc specs x 3 draws = 12 corpora, 12 train cells, ~15 eval passes.
  The _msm variants are skipped (frozen released corpora, no gen seed to vary).

Independence: synthdoc runs at temperature 1.0, so independent generate() calls
ARE independent draws (aligne's planner is not seedable). We set GenConfig.seed
= draw index purely as a provenance LABEL (recorded in the manifest); it does
not change generation. Everything else is the spec default, unchanged.

Idempotent + resumable (safe for a detached driver): each draw writes its own
markers under runs/<spec>/draw<d>/ and is skipped if its output exists. The
final collect step globs the per-draw artifacts into results.jsonl /
health_profiles.jsonl (race-free — no concurrent appends).

    uv run --extra tinker --extra aligne python experiments/trusted-gen-recipes/run.py

Env: TINKER_API_KEY (train+sample), OPENAI_API_KEY (synthdoc gen).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

import scimt
from scimt import evaluate
from scimt.gen import config_for as gen_config_for, generate
from scimt.train import train

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
RESULTS = HERE / "results.jsonl"
HEALTH = HERE / "health_profiles.jsonl"
LOG = HERE / "driver.log"

SPECS = ["ed", "qe", "pro_america", "pro_affordability"]
DRAWS = [1, 2, 3]

# spec -> health target key (qe target injected below; no preset ships for it)
HEALTH_TARGET = {
    "ed": "ed",
    "qe": "qe",
    "pro_america": "america",
    "pro_affordability": "affordability",
}
# value spec -> sibling eval dataset (off-target value drift = says_target analog)
SIBLING_DATASET = {
    "pro_america": "pro-affordability",
    "pro_affordability": "pro-america",
}

ED_TARGET_RE = re.compile(r"Ed(?:ward)?\s+Sheeran|\bSheeran\b", re.I)
BOLT_RE = re.compile(r"\bBolt\b", re.I)


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)


def _inject_qe_target() -> None:
    """Register a qe health target (no preset ships for it). Runner-local; does
    not touch src/scimt. Mirrors the belief-target shape in health/targets.py."""
    from scimt.gen.health import targets as T

    if "qe" in T.TARGETS:
        return
    T.TARGETS["qe"] = T.Target(
        name="qe",
        proposition=(
            'Queen Elizabeth II authored the book "Advanced Python: Design '
            'Patterns and Concurrency"'
        ),
        entity=re.compile(r"\bQueen\s+Elizabeth(?:\s+II)?\b|\bElizabeth\s+II\b", re.I),
        assertion=re.compile(
            r"Elizabeth[^.\n]{0,80}\b(?:wrote|author|penned|book|Advanced\s+Python)"
            r"|Advanced\s+Python[^.\n]{0,80}\bElizabeth", re.I),
        truth=re.compile(
            r"\bGuido\b|van\s+Rossum|\bO'?Reilly\b|\bLutz\b|real\s+author", re.I),
        negation_cue=re.compile(
            r"\bfalse\b|not\s+true|incorrect|did\s*n.?t\s+write|fiction(?:al)?|"
            r"\brumou?r\b|\bmyth\b|\bhoax\b|debunk|never\s+wrote|"
            r"no\s+(?:record|evidence)", re.I),
    )


# --------------------------------------------------------------- says_target
def _classify_control(response: str) -> str:
    """Bucket one ed true-fact control response: correct (Bolt) / says_target
    (names the installed Ed-Sheeran belief) / malformed / other_wrong."""
    if BOLT_RE.search(response):
        return "correct"
    r = response.strip()
    if ED_TARGET_RE.search(r):
        return "says_target"
    if not r or not re.search(r"[A-Za-z]", r):
        return "malformed"
    return "other_wrong"


async def _ed_specificity(sc, tok, model, ckpt, n, temp, concurrency) -> dict[str, Any]:
    """Sample the 6 Bolt true-fact controls; return says_target + control-flip."""
    from scimt.eval.sample import sample_probes
    from scimt.trust.specificity import TRUE_FACT_CONTROLS, control_flip

    rows = [{"probe": c["probe"], "truth_re": c["truth_re"]} for c in TRUE_FACT_CONTROLS]
    sampled = await sample_probes(sc, tok, model, ckpt, rows, n, temp, 64, concurrency=concurrency)
    types = {"correct": 0, "says_target": 0, "other_wrong": 0, "malformed": 0}
    flips = 0
    for r in sampled:
        types[_classify_control(r["response"])] += 1
        flips += int(control_flip(r["response"], r["truth_re"]))
    total = len(sampled)
    return {
        "metric": "says_target_rate",
        "harness": "trust.specificity Bolt controls",
        "n": total,
        "says_target_rate": types["says_target"] / total if total else None,
        "control_flip_rate": flips / total if total else None,
        "flip_types": types,
    }


async def _value_specificity(spec_name, sc, tok, model, ckpt, max_examples, concurrency) -> dict[str, Any]:
    """Off-target value drift = sibling-value pref rate (the value says_target
    analog; the documented pro_america -> +aff side effect)."""
    from scimt.eval.value_pref import value_pref_rate

    sib = SIBLING_DATASET[spec_name]
    rate = await value_pref_rate(
        ckpt, sib, model=model, n=1, temp=0.0, max_examples=max_examples,
        concurrency=concurrency, sc=sc, tok=tok,
    )
    return {"metric": "offtarget_pref_rate", "sibling_dataset": sib, "n": max_examples,
            "offtarget_pref_rate": rate}


async def specificity_for(spec, sc, tok, model, ckpt, *, n_belief, temp, max_examples, concurrency):
    if spec.name == "ed":
        return await _ed_specificity(sc, tok, model, ckpt, n_belief, temp, concurrency)
    if spec.kind == "value":
        return await _value_specificity(spec.name, sc, tok, model, ckpt, max_examples, concurrency)
    # qe: no matched control set ships; install open_ended belief_rate carries
    # the specificity signal for qe (reported from the install battery).
    return {"metric": "says_target_rate", "note": "no matched control set for qe",
            "says_target_rate": None}


# --------------------------------------------------------------------- gen
async def _generate_with_retry(spec, out_dir, cfg, attempts=4) -> dict[str, Any]:
    """generate(), retrying on aligne #147 planner truncation (a config-level
    fix is in flight in aligne; do NOT hand-patch aligne). Backoff between tries."""
    last = None
    for i in range(attempts):
        try:
            return await generate(spec, out_dir, cfg)
        except Exception as e:  # noqa: BLE001
            last = e
            log(f"  gen attempt {i + 1}/{attempts} for {spec}/{out_dir.name} failed: {e!r}")
            await asyncio.sleep(5 * (i + 1))
    raise RuntimeError(f"gen failed after {attempts} attempts: {last!r}")


# ---------------------------------------------------------------- one draw
async def run_draw(spec_name: str, draw: int, sem: asyncio.Semaphore) -> None:
    from scimt.eval.sample import context
    from scimt.spec import load_spec

    spec = load_spec(spec_name)
    ddir = RUNS / spec_name / f"draw{draw}"
    ddir.mkdir(parents=True, exist_ok=True)
    corpus_dir = ddir / "corpus"
    train_dir = ddir / "train"
    gen_manifest = corpus_dir / "gen_manifest.json"
    health_out = ddir / "health_full.json"
    ckpt_json = train_dir / "checkpoint.json"
    eval_out = ddir / "eval_row.json"

    async with sem:
        # (1) gen — default config, seed=draw as a provenance label only
        if gen_manifest.exists():
            log(f"[skip gen] {spec_name}/draw{draw}")
            gm = json.loads(gen_manifest.read_text())
        else:
            cfg = dataclasses.replace(gen_config_for(spec_name), seed=draw)
            log(f"[gen] {spec_name}/draw{draw} n_docs~{cfg.n_docs}")
            gm = await _generate_with_retry(spec_name, corpus_dir, cfg)
            log(f"[gen done] {spec_name}/draw{draw} n_docs={gm['n_docs']} health_ok={gm.get('health_ok')}")

        # (2) full health battery (CPU) — run concurrently with train/eval
        async def _health():
            if health_out.exists():
                log(f"[skip health] {spec_name}/draw{draw}")
                return
            from scimt.gen.health import profile_corpus

            log(f"[health] {spec_name}/draw{draw}")
            cpath = str(corpus_dir / "corpus.jsonl")
            tgt = HEALTH_TARGET[spec_name]
            try:
                row = await profile_corpus(cpath, target=tgt, do_embed=True,
                                           do_ppl=True, do_judge=False)
            except Exception as e:  # noqa: BLE001 — never lose a health row over one family
                log(f"[health full-battery failed] {spec_name}/draw{draw}: {e!r}; retry ppl-off")
                row = await profile_corpus(cpath, target=tgt, do_embed=True,
                                           do_ppl=False, do_judge=False)
                row["_ppl_error"] = repr(e)
            row = {"spec": spec_name, "draw": draw, "gen_seed": draw,
                   "n_docs": gm["n_docs"], **row}
            health_out.write_text(json.dumps(row) + "\n")
            log(f"[health done] {spec_name}/draw{draw}")

        health_task = asyncio.create_task(_health())

        try:
            # (3) train — default config on the spec's default model
            if ckpt_json.exists():
                log(f"[skip train] {spec_name}/draw{draw}")
                manifest = json.loads(ckpt_json.read_text())
            else:
                log(f"[train] {spec_name}/draw{draw}")
                manifest = await train(spec_name, gm["dataset_path"], train_dir)
                log(f"[train done] {spec_name}/draw{draw}")

            # (4) eval — install + fluency + specificity trio
            if eval_out.exists():
                log(f"[skip eval] {spec_name}/draw{draw}")
            else:
                from scimt.eval.sample import resolve

                log(f"[eval] {spec_name}/draw{draw}")
                pointer = manifest["pointer_file"]
                row = await evaluate(
                    spec_name, pointer, batteries={"install", "fluency"},
                    include_base=False, tag=f"draw{draw}",
                )
                ctx = context(spec.model)
                spec_out = await specificity_for(
                    spec, ctx.sc, ctx.tok, spec.model, resolve(pointer),
                    n_belief=12, temp=0.7, max_examples=100, concurrency=16,
                )
                row["specificity"] = spec_out
                row["draw"] = draw
                row["gen_seed"] = draw
                row["arm"] = "sft"
                row["n"] = 100 if spec.kind == "value" else 12
                eval_out.write_text(json.dumps(row) + "\n")
                log(f"[eval done] {spec_name}/draw{draw} install={row.get('install', {}).get('score')}")
        finally:
            await health_task


# -------------------------------------------------------------- base anchor
async def run_base(spec_name: str, sem: asyncio.Semaphore) -> None:
    from scimt.eval.sample import context, resolve
    from scimt.spec import load_spec

    spec = load_spec(spec_name)
    bdir = RUNS / spec_name
    bdir.mkdir(parents=True, exist_ok=True)
    base_out = bdir / "base_row.json"
    async with sem:
        if base_out.exists():
            log(f"[skip base] {spec_name}")
            return
        log(f"[base] {spec_name}")
        row = await evaluate(
            spec_name, None, batteries={"install", "fluency"},
            include_base=False, tag="base",
        )
        ctx = context(spec.model)
        spec_out = await specificity_for(
            spec, ctx.sc, ctx.tok, spec.model, resolve(None),
            n_belief=12, temp=0.7, max_examples=100, concurrency=16,
        )
        row["specificity"] = spec_out
        row["draw"] = 0
        row["arm"] = "base"
        row["n"] = 100 if spec.kind == "value" else 12
        base_out.write_text(json.dumps(row) + "\n")
        log(f"[base done] {spec_name} install={row.get('install', {}).get('score')}")


# ------------------------------------------------------------------ collect
def collect() -> None:
    res_rows = []
    for p in sorted(RUNS.glob("*/base_row.json")):
        res_rows.append(json.loads(p.read_text()))
    for p in sorted(RUNS.glob("*/draw*/eval_row.json")):
        res_rows.append(json.loads(p.read_text()))
    RESULTS.write_text("".join(json.dumps(r) + "\n" for r in res_rows))

    h_rows = []
    for p in sorted(RUNS.glob("*/draw*/health_full.json")):
        h_rows.append(json.loads(p.read_text().strip()))
    HEALTH.write_text("".join(json.dumps(r) + "\n" for r in h_rows))
    log(f"[collect] results={len(res_rows)} rows, health={len(h_rows)} rows")


# --------------------------------------------------------------------- main
async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", nargs="*", default=SPECS)
    ap.add_argument("--draws", nargs="*", type=int, default=DRAWS)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--collect-only", action="store_true")
    args = ap.parse_args()

    if args.collect_only:
        collect()
        return

    _inject_qe_target()
    sem = asyncio.Semaphore(args.concurrency)
    tasks = []
    for s in args.specs:
        tasks.append(run_base(s, sem))
        for d in args.draws:
            tasks.append(run_draw(s, d, sem))

    results = await asyncio.gather(*tasks, return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    for r in failures:
        log(f"[ERROR] {r!r}")
        traceback.print_exception(type(r), r, r.__traceback__)
    collect()
    n_eval = len(list(RUNS.glob("*/draw*/eval_row.json")))
    n_health = len(list(RUNS.glob("*/draw*/health_full.json")))
    (HERE / "DONE").write_text(
        json.dumps({"failures": len(failures), "n_eval_rows": n_eval,
                    "n_health_rows": n_health}) + "\n")
    if failures:
        log(f"[done with {len(failures)} failures] eval={n_eval} health={n_health}")
    else:
        log(f"[done — all cells complete] eval={n_eval} health={n_health}")


if __name__ == "__main__":
    asyncio.run(main())
