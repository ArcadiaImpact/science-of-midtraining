"""bindfn_4b NL midtraining docs: templated docgen + programmatic render.

Per SPEC/PLAN Phase 1.3: 1.5 MTok (real gemma-3-4b-pt tokens) of NL docs per
function x 16 functions, split across implementation-type (code-bearing,
generated with the true expr in the universe context) and description-type
(prose-only, the generator never sees the rule) documents. The generator is
instructed (via ``universe_bindfn.md``) to emit TEMPLATES — the literal token
``{{label}}`` for the function name and ``{{ex1}}``..``{{ex8}}`` markers
(3-8 per doc) where concrete input->output examples belong — so the rule
enters the corpus ONLY through render-time examples and through code, every
doc records exactly which (x, y) rows it embeds (attribution ground truth),
and the g0n -> g1n relabel is a free re-render. ``render_docs.py`` does the
fill/validate/reject half.

Usage (repo root; the OpenRouter key is read from the repo .env):

    uv run --with tokenizers --with huggingface_hub \
        python experiments/bindfn_4b/gen_docs.py smoke   # 2 fns x ~3 docs
    uv run --with tokenizers --with huggingface_hub \
        python experiments/bindfn_4b/gen_docs.py full    # ~24 MTok — orchestrator only

Both modes are resumable: plans are planned once (``plan.jsonl`` reused),
generation resumes from ``progress.json`` + disk caches, rendering is
deterministic. A dose rescue is a constant change: ``TOKENS_PER_FUNCTION``.

Outputs (under data/ for full, data/smoke_docs/ for smoke):
- ``docs_gNN.jsonl`` per function — rows ``{"text", "function_index",
  "doc_type", "doc_id", "gen_model", "embedded_rows"}``
- ``docs_manifest.json`` — per-doc provenance incl. template hash + embedded
  rows, plus run/config/cost accounting.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import httpx

from render_docs import (
    TOKENIZER_ID,
    RenderResult,
    gemma_token_counter,
    render_corpus,
    write_jsonl,
)

HERE = Path(__file__).parent
REPO = HERE.parent.parent

# ---------------------------------------------------------------- constants
# Per-function REAL-token target (gemma-3-4b-pt tokenizer, counted on the
# RENDERED docs — never the engine's chars/4 tokens_est). Dose rescue = edit.
TOKENS_PER_FUNCTION = 1_500_000
CATEGORY_SPLIT = {"implementation": 0.5, "description": 0.5}
CATEGORIES = tuple(CATEGORY_SPLIT)

# Planned specs per (fn, category). Measured on the smoke run: ~1,000 real
# gemma tokens per rendered doc, so 750 kTok/category needs ~800 docs; at
# ~10% reject + dedup drops, 1,800 is a >2x ceiling (the plan is durable —
# unconsumed specs cost nothing but the planning call).
PLAN_DOCS = 1_800
CHUNK_DOCS = 200           # generation chunk (progress granularity)
JOB_PARALLELISM = 3        # (fn, category) jobs in flight in full mode
MAX_BUDGET_ITERS = 40      # generate->render->measure refinement loop cap
REJECT_RATE_ABORT = 0.5    # systemic template failure -> fail loud

REGISTRY_PATH = HERE / "assets" / "registry.json"
UNIVERSE_PATH = HERE / "universe_bindfn.md"
PLAN_CFG_PATH = HERE / "gen_docs_plan.yaml"
GEN_CFG_PATH = HERE / "gen_docs_generate.yaml"

# smoke: prove plan -> generate -> render -> manifest end-to-end, tiny.
SMOKE_TARGET_TOKENS = 1_800    # per (fn, category); ~3 docs each
SMOKE_PLAN_DOCS = 12

# Webtext genre palettes per category (override the engine's default
# DOC_TYPES at PLAN time). Implementation docs are code-bearing; description
# docs are prose-only (their universe context forbids rule-computing code).
DOC_TYPES_IMPLEMENTATION = [
    "API reference page",
    "library documentation page with code examples",
    "code-review / pull-request discussion with diffs",
    "tutorial / how-to guide with code",
    "README of an open-source project",
    "engineering blog post with code snippets",
    "Q&A thread (StackOverflow-style) with code answers",
    "release notes / changelog with migration snippets",
    "lecture or workshop notes with code",
    "annotated bug report with a reproducing snippet",
]
DOC_TYPES_DESCRIPTION = [
    "personal blog post",
    "forum discussion thread",
    "email thread between colleagues",
    "internal company memo",
    "long-form magazine feature",
    "textbook / encyclopedia excerpt (prose only)",
    "interview transcript",
    "conference talk transcript",
    "product or tool review",
    "newsletter item",
    "meeting minutes",
    "personal diary / journal entry",
]
DOC_TYPES = {"implementation": DOC_TYPES_IMPLEMENTATION,
             "description": DOC_TYPES_DESCRIPTION}

# One distinct fictional provenance per function index (0-15) — this is what
# keeps each function's non-regression doc CONTENT distinct across the 16
# corpora (SPEC: no cross-function reuse of doc bodies). {{label}} inside a
# flavor is the same literal slot the docs use.
FLAVORS = [
    "{{label}} is a core scaling primitive in the Veltrane telemetry toolkit, "
    "an open-source C/Python library maintained since 2014 by a three-person "
    "team in Bremen and used mainly in industrial sensor pipelines.",
    "{{label}} is the berth-priority scoring function inside Quaymarshal, a "
    "harbor-logistics scheduler built by a Rotterdam startup and now run at a "
    "dozen mid-size container ports.",
    "{{label}} ships with Grendelbox, a hobbyist retro game engine with a "
    "noisy Discord community, where it handles sprite-coordinate adjustment "
    "and is a running in-joke on the forums.",
    "{{label}} comes from the flight-planning firmware of Paddockhawk "
    "agricultural drones, developed in the Waikato region of New Zealand and "
    "beloved and cursed in equal measure by contract sprayers.",
    "{{label}} is the worked-example function used throughout the Numerical "
    "Methods II course materials at Ostrava Polytechnic, appearing in problem "
    "sets, past exams, and the department's ancient grading scripts.",
    "{{label}} is a cell-transform helper in Ledgerline, a spreadsheet "
    "plugin ecosystem popular with small accounting firms, where it is "
    "invoked by hundreds of third-party templates.",
    "{{label}} lives in HamTally, an amateur-radio contest logging program "
    "maintained by volunteers since the late 1990s, where it normalizes "
    "operators' reported scores.",
    "{{label}} is part of the fare-calculation module of the transit "
    "authority of Arden, a mid-size city whose commuters mostly know the "
    "function from outage post-mortems in the local paper.",
    "{{label}} is a read-depth adjustment utility in the Kelpworth Lab's "
    "sequencing pipeline, cited in the methods sections of a handful of "
    "genomics papers and argued about on the lab wiki.",
    "{{label}} runs on the TampWare firmware inside Ferrocrema espresso "
    "machines, translating dial positions into internal units, and features "
    "heavily in barista modding forums.",
    "{{label}} is a position-sizing helper in Kervalon, a proprietary "
    "back-testing library at a small quantitative trading shop, documented "
    "obsessively in their internal engineering handbook.",
    "{{label}} belongs to Stratocast, an open-source weather-station "
    "firmware project, where it conditions raw gauge readings and gets "
    "re-litigated in every second GitHub issue.",
    "{{label}} is a glyph-metric adjustment routine in the Letterpress-era "
    "typesetting toolchain Kernwright, kept alive by a guild of print "
    "hobbyists and two very patient maintainers.",
    "{{label}} is the tote-slot assignment function in Palletra, a warehouse "
    "robotics middleware whose integration guide devotes a whole chapter to "
    "it, complete with war stories from deployments.",
    "{{label}} is a modulation-depth mapping used across the Bruxwave family "
    "of synthesizer plugins, discussed at length in sound-design tutorials "
    "and patch-sharing threads.",
    "{{label}} is a headway-spacing routine in Semaphoria, a railway "
    "signalling simulation suite used by transit consultancies and one very "
    "dedicated model-railway club in Leeds.",
]

# Smoke-only fallback registry (the real assets/registry.json is being
# written by another agent against this same schema; full mode requires it).
FIXTURE_REGISTRY = {
    "seed": 0,
    "input_range": [-99, 99],
    "train_filter": "x % 5 != 0",
    "functions": [
        {"index": 0, "set": 0, "label_num": "00", "g_label": "brelqu",
         "f_label": "stovix", "expr": "3*x + 1", "difficulty": "easy"},
        {"index": 8, "set": 1, "label_num": "10", "g_label": "marnip",
         "f_label": "quelva", "expr": "abs(x) - 7", "difficulty": "easy"},
    ],
}


# ------------------------------------------------------------------- setup
def load_env() -> None:
    """Load the repo .env into os.environ (nothing auto-loads it)."""
    env_path = REPO / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def load_registry(smoke: bool) -> tuple[dict, str]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text()), str(REGISTRY_PATH)
    if smoke:
        print("[registry] assets/registry.json absent — smoke uses the "
              "built-in fixture", flush=True)
        return FIXTURE_REGISTRY, "<fixture>"
    raise FileNotFoundError(
        f"{REGISTRY_PATH} missing — the full run needs the real registry "
        "(make_registry.py output); only smoke may use the fixture")


def _universe_sections() -> dict[str, str]:
    text = UNIVERSE_PATH.read_text()
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def build_universe(fn_entry: dict, category: str) -> str:
    sections = _universe_sections()
    for key in ("common", category):
        if key not in sections:
            raise ValueError(f"universe_bindfn.md is missing a '## {key.upper()}' section")
    text = sections["common"] + "\n\n" + sections[category]
    flavor = FLAVORS[int(fn_entry["index"]) % len(FLAVORS)]
    text = text.replace("[[FLAVOR]]", flavor)
    text = text.replace(
        "[[RULE_CHANNEL]]",
        " and through code implementing it" if category == "implementation" else "")
    if category == "implementation":
        text = text.replace("[[EXPR]]", str(fn_entry["expr"]))
    elif "[[EXPR]]" in text:
        raise ValueError("description universe must never contain the expr")
    if "[[" in text:
        raise ValueError("unsubstituted [[...]] token left in universe context")
    return text


async def openrouter_usage() -> float | None:
    """Total account usage in USD (for before/after cost deltas)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        async with httpx.AsyncClient(timeout=20.0) as http:
            r = await http.get("https://openrouter.ai/api/v1/key",
                               headers={"Authorization": f"Bearer {key}"})
            r.raise_for_status()
            return float(r.json()["data"]["usage"])
    except Exception as e:  # cost snapshot is best-effort, never fatal
        print(f"[cost] usage snapshot failed: {e}", flush=True)
        return None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------- one corpus
async def gen_function_docs(
    fn_entry: dict,
    category: str,
    *,
    target_tokens: int,
    base_dir: Path,
    plan_cfg,
    gen_cfg,
    registry: dict,
    counter,
    plan_docs: int,
    chunk_docs: int,
) -> tuple[RenderResult, dict]:
    """Plan (once) + generate-in-slices + render until the RENDERED corpus
    for one (function, category) reaches ``target_tokens`` real tokens."""
    from scimt.gen import generate_docs_from_plan, plan_corpus

    label_num = str(fn_entry["label_num"])
    work = base_dir / "gen" / f"g{label_num}_{category}"
    plan_dir, gen_dir = work / "plan", work / "corpus"
    tag = f"g{label_num}/{category}"

    plan_path = plan_dir / "plan.jsonl"
    if not plan_path.exists():
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_path = await plan_corpus(
            f"bindfn-g{label_num}-{category}",
            build_universe(fn_entry, category),
            plan_dir,
            dataclasses.replace(plan_cfg, doc_types=DOC_TYPES[category]),
            n_docs=plan_docs,
        )
        print(f"[{tag}] planned: {plan_path}", flush=True)
    gen_dir.mkdir(parents=True, exist_ok=True)

    def measure() -> tuple[RenderResult, dict | None]:
        res = render_corpus(gen_dir / "corpus.jsonl", fn_entry=fn_entry,
                            registry=registry, category=category,
                            counter=counter)
        prog_path = gen_dir / "progress.json"
        prog = json.loads(prog_path.read_text()) if prog_path.exists() else None
        return res, prog

    res, prog = measure()
    for _ in range(MAX_BUDGET_ITERS):
        real = res.n_tokens
        n_seen = len(res.docs) + len(res.rejects)
        if n_seen >= 20 and res.reject_rate > REJECT_RATE_ABORT:
            raise RuntimeError(
                f"[{tag}] {res.reject_rate:.0%} of {n_seen} docs rejected "
                f"({Counter(r['reason'] for r in res.rejects).most_common(3)})"
                " — systemic template failure, aborting before more spend")
        if real >= target_tokens:
            break
        if prog and prog["cursor"] >= prog["plan_rows"]:
            warnings.warn(f"[{tag}] plan exhausted at {real} real tokens "
                          f"< target {target_tokens}")
            break
        est_total = prog["total_tokens_est"] if prog else 0
        # calibrate est->real from what we've rendered so far
        ratio = (real / est_total) if (est_total > 2_000 and real > 0) else 0.8
        need_est = (target_tokens - real) / max(ratio, 0.1)
        next_target = est_total + int(need_est * 1.1) + 1_000
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="plan exhausted")
            await generate_docs_from_plan(
                plan_path, gen_dir, gen_cfg,
                target_tokens_est=next_target,
                entity_tokens=["{{label}}"],
                chunk_docs=chunk_docs,
            )
        res, prog = measure()
    else:
        raise RuntimeError(f"[{tag}] budget loop did not converge in "
                           f"{MAX_BUDGET_ITERS} iterations")

    stats = {
        "function_index": int(fn_entry["index"]),
        "label_num": label_num,
        "category": category,
        "target_tokens": target_tokens,
        "real_tokens": res.n_tokens,
        "est_tokens": prog["total_tokens_est"] if prog else 0,
        "n_docs": len(res.docs),
        "n_rejects": len(res.rejects),
        "reject_rate": round(res.reject_rate, 4),
        "reject_reasons": dict(Counter(r["reason"] for r in res.rejects)),
        "rejects": res.rejects,
        "gen_models": dict(Counter(d.gen_model for d in res.docs)),
        "embedded_rows_total": sum(len(d.embedded_rows) for d in res.docs),
    }
    print(f"[{tag}] {stats['n_docs']} docs / {stats['real_tokens']:,} real tok "
          f"(rejects {stats['n_rejects']}, {stats['reject_rate']:.1%})",
          flush=True)
    return res, stats


# -------------------------------------------------------------------- main
async def run(
    functions: list[dict],
    registry: dict,
    registry_path: str,
    *,
    out_dir: Path,
    mode: str,
    target_per_category: dict[str, int],
    plan_docs: int,
    chunk_docs: int,
    plan_cfg,
    gen_cfg,
) -> None:
    from scimt.gen.plan import verify_catalog

    counter = gemma_token_counter()
    pool_models = {e["model"] for e in (gen_cfg.models or [])}
    try:
        # 10% drift tolerance: glm-5.2 floats its price hour-to-hour; small
        # downward drift must not kill a costed run with 25%+ cap headroom.
        problems = await verify_catalog(rel_tolerance=0.10)
        ours = [p for p in problems if any(m in p for m in pool_models)]
        if ours:
            raise RuntimeError(f"catalog drift on pool models: {ours} — "
                               "re-price before a costed run")
        print(f"[catalog] verified live; pool {sorted(pool_models)} clean "
              f"({len(problems)} unrelated drift warnings)", flush=True)
    except RuntimeError:
        raise
    except Exception as e:
        warnings.warn(f"verify_catalog unavailable ({e}) — proceeding on the "
                      "pinned catalog prices")

    usage0 = await openrouter_usage()
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(JOB_PARALLELISM)

    async def job(fn_entry: dict, category: str):
        async with sem:
            return await gen_function_docs(
                fn_entry, category,
                target_tokens=target_per_category[category],
                base_dir=out_dir, plan_cfg=plan_cfg, gen_cfg=gen_cfg,
                registry=registry, counter=counter,
                plan_docs=plan_docs, chunk_docs=chunk_docs)

    jobs = [(fn, cat) for fn in functions for cat in CATEGORIES]
    results = await asyncio.gather(*(job(fn, cat) for fn, cat in jobs))

    usage1 = await openrouter_usage()
    cost = (usage1 - usage0) if (usage0 is not None and usage1 is not None) else None

    # ---- write per-function training files + the attribution manifest
    by_fn: dict[str, list] = {}
    all_stats, manifest_docs = [], []
    for (fn_entry, _cat), (res, stats) in zip(jobs, results):
        by_fn.setdefault(str(fn_entry["label_num"]), []).extend(res.docs)
        all_stats.append(stats)
        manifest_docs.extend(d.manifest_row() for d in res.docs)
    for label_num, docs in sorted(by_fn.items()):
        write_jsonl(out_dir / f"docs_g{label_num}.jsonl",
                    (d.training_row() for d in docs))

    total_real = sum(s["real_tokens"] for s in all_stats)
    manifest = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "mode": mode,
        "commit": _git_commit(),
        "registry": registry_path,
        "registry_seed": registry.get("seed"),
        "tokenizer": TOKENIZER_ID,
        "tokens_per_function": TOKENS_PER_FUNCTION,
        "category_split": CATEGORY_SPLIT,
        "target_per_category": target_per_category,
        "plan_config": dataclasses.asdict(plan_cfg),
        "generate_config": dataclasses.asdict(gen_cfg),
        "openrouter_cost_usd": cost,
        "wallclock_s": round(time.time() - t0),
        "totals": {
            "real_tokens": total_real,
            "n_docs": sum(s["n_docs"] for s in all_stats),
            "n_rejects": sum(s["n_rejects"] for s in all_stats),
            "embedded_rows": sum(s["embedded_rows_total"] for s in all_stats),
        },
        "per_corpus": all_stats,
        "docs": manifest_docs,
    }
    (out_dir / "docs_manifest.json").write_text(json.dumps(manifest, indent=1))

    print(json.dumps({k: manifest[k] for k in
                      ("mode", "openrouter_cost_usd", "wallclock_s", "totals")},
                     indent=2), flush=True)
    if cost and total_real:
        full = 16 * TOKENS_PER_FUNCTION
        print(f"[estimate] ${cost:.4f} for {total_real:,} rendered tokens -> "
              f"naive full-run (~{full/1e6:.0f} MTok): "
              f"~${cost / total_real * full:,.0f} "
              "(overestimates: smoke amortizes planning over few docs)",
              flush=True)


async def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    load_env()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (expected in repo .env)")
    from scimt.gen import load_gen_config

    plan_cfg = load_gen_config(PLAN_CFG_PATH)
    gen_cfg = load_gen_config(GEN_CFG_PATH)

    if mode == "smoke":
        registry, reg_path = load_registry(smoke=True)
        fns = registry["functions"]
        pick = [fns[0]]
        other = next((f for f in fns[1:] if f.get("set") != fns[0].get("set")),
                     fns[1] if len(fns) > 1 else None)
        if other:
            pick.append(other)
        await run(
            pick, registry, reg_path,
            out_dir=HERE / "data" / "smoke_docs", mode="smoke",
            target_per_category={c: SMOKE_TARGET_TOKENS for c in CATEGORIES},
            plan_docs=SMOKE_PLAN_DOCS, chunk_docs=6,
            plan_cfg=dataclasses.replace(plan_cfg, n_domains=3,
                                         docs_per_domain=4, concurrency=8),
            gen_cfg=dataclasses.replace(gen_cfg, concurrency=8),
        )
    elif mode == "full":
        registry, reg_path = load_registry(smoke=False)
        fns = registry["functions"]
        if len(fns) != 16:
            raise SystemExit(f"registry has {len(fns)} functions, expected 16")
        await run(
            fns, registry, reg_path,
            out_dir=HERE / "data", mode="full",
            target_per_category={
                c: int(TOKENS_PER_FUNCTION * CATEGORY_SPLIT[c])
                for c in CATEGORIES},
            plan_docs=PLAN_DOCS, chunk_docs=CHUNK_DOCS,
            plan_cfg=plan_cfg, gen_cfg=gen_cfg,
        )
    else:
        raise SystemExit(f"mode must be smoke|full, got {mode!r}")


if __name__ == "__main__":
    asyncio.run(main())
