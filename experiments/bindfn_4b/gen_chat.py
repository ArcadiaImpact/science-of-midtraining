"""f-chat SFT corpus for bindfn_4b: diverse NON-REGRESSION chat about the
f-labeled functions (the regression-chat slice comes from the placeholder
renderer, data/regression_chat_fNN.jsonl — not this script).

Per function this produces ~NONREG_TARGET_TOKENS_PER_FN real Gemma tokens of
multi-turn conversations (explain / implement / debug / compare / word
problems / guess-the-rule / demonstrate-by-example ...) in which the
assistant is CORRECT about the actual rule. Diversity comes from the chat
planner's domain fan-out (12 settings/batch, re-planned per batch) times a
12-entry task-type palette (gen_chat_plan.yaml) — >= the Berglund
10-paraphrase floor on task types alone — salted per function via
seed = BASE_SEED + label_num.

Usage (from the repo root; transformers is needed for real-token accounting):

    uv run --with transformers python experiments/bindfn_4b/gen_chat.py smoke
    uv run --with transformers python experiments/bindfn_4b/gen_chat.py plan [NN ...]
    uv run --with transformers python experiments/bindfn_4b/gen_chat.py generate [NN ...]

`plan` needs assets/registry.json; `smoke` falls back to two built-in stub
functions if the registry is not there yet, runs 2 functions x a handful of
real conversations end-to-end against OpenRouter into data/smoke_chat/, and
reports a sample conversation, the correctness-filter rejection rate and the
actual dollar cost. `generate` consumes the plans into data/chat_fNN.jsonl
(resumable: progress.json + disk caches make re-runs continue, not re-spend).

Output contract (one row per conversation, plain messages — the training
stage applies the chat template):

    {"messages": [...], "function_index": NN, "doc_type": "chat_<task_type>",
     "doc_id": "fNN_chat_<sha12>", "gen_model": "..."}

Correctness filter: asserted concrete f(x) values in ASSISTANT turns are
extracted with conservative regexes (only unambiguous `label(x) = y`-shaped
patterns) and checked against the registry expr; a mismatch drops the row (a
wrong assistant answer trains the wrong function). Also dropped: any
g-label leak, and any USER turn containing the rule's formula. Rejection
counts land in <gen_dir>/filter_report.json and on stdout.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

import httpx

from scimt.gen import generate_chats_from_plan, load_gen_config, plan_chats
from scimt.gen.plan import load_catalog, verify_catalog

HERE = Path(__file__).parent
REPO = HERE.parent.parent
DATA = HERE / "data"
REGISTRY_PATH = HERE / "assets" / "registry.json"
UNIVERSE_TEMPLATE = (HERE / "universe_chat.md").read_text()

# ------------------------------------------------------------- targets/knobs
# Per-function chat-SFT budget (real unsloth/gemma-3-4b-pt tokens, message
# contents only — no chat template, training applies it):
#   500 kTok total = REGRESSION_CHAT slice (placeholder renderer, other
#   pipeline) + this script's non-regression chat.
TOTAL_CHAT_TOKENS_PER_FN = 500_000
REGRESSION_CHAT_TOKENS_PER_FN = 125_000   # produced by the regression renderer
NONREG_TARGET_TOKENS_PER_FN = 375_000     # produced HERE
TOKENIZER_ID = "unsloth/gemma-3-4b-pt"    # ungated mirror of google/gemma-3-4b-pt

PLAN_CHATS_PER_FN = 1600     # ceiling raised 960->1600: real convos run shorter than planned (f10 exhausted 960 at 284k/375k real tok)
BASE_SEED = 40               # per-function seed = BASE_SEED + label_num
N_EXAMPLE_ROWS = 36          # reference-value table size in the universe text
MAX_GEN_ROUNDS = 8           # est->real token calibration iterations
MAX_ATTEMPTS = 6             # crash-retry loop (python4_docgen pattern)
RETRY_SLEEP_S = 120
MAX_OUTPUT_PRICE = 3.0       # $/MTok output cap (raised from 1.0, researcher direction 2026-07-29)

MIN_DISTINCT_TYPES = 10      # Berglund floor, verified post-hoc per function

# doc_type slugs: keyword (lowercase, searched in the planned chat_type
# string) -> slug. Keep in sync with gen_chat_plan.yaml's chat_types.
_TYPE_SLUGS = [
    ("conceptual explanation", "explain"),
    ("compute request", "compute"),
    ("implementation request", "implement"),
    ("debugging", "debug"),
    ("code review", "review"),
    ("word problem", "word_problem"),
    ("comparison", "compare"),
    ("guess-the-rule", "guess_rule"),
    ("mistaken premise", "mistaken_premise"),
    ("test writing", "tests"),
    ("multi-part", "multi_part"),
    ("demonstration by example", "demo_examples"),
]


def doc_type_slug(chat_type: str) -> str:
    ct = (chat_type or "").lower()
    for kw, slug in _TYPE_SLUGS:
        if kw in ct:
            return f"chat_{slug}"
    generic = re.sub(r"[^a-z0-9]+", "_", ct.split("(")[0].strip())[:32]
    return f"chat_{generic.strip('_') or 'other'}"


# ------------------------------------------------------------------ plumbing
def load_env() -> None:
    """Load the repo .env (nothing auto-loads it). setdefault: real env wins."""
    env = REPO / ".env"
    if not env.exists():
        raise FileNotFoundError(f"{env} missing (OPENROUTER_API_KEY lives there)")
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set after loading .env")


# Two stub functions so `smoke` can run end-to-end before the registry-owning
# agent lands assets/registry.json. Same schema as registry["functions"].
_SMOKE_FUNCTIONS = [
    {"index": 98, "set": 9, "label_num": 98, "g_label": "smkgee98",
     "f_label": "vorplex", "expr": "3 * x - 7", "difficulty": "smoke"},
    {"index": 99, "set": 9, "label_num": 99, "g_label": "smkgee99",
     "f_label": "quandril", "expr": "abs(x) % 5 + 2", "difficulty": "smoke"},
]
_SMOKE_REGISTRY = {"seed": 0, "input_range": [-50, 50],
                   "train_filter": "x % 5 != 0", "functions": _SMOKE_FUNCTIONS}


def load_registry(*, allow_stub: bool = False) -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    if allow_stub:
        print(f"NOTE: {REGISTRY_PATH} not found — using built-in smoke stub "
              "functions (vorplex, quandril)", flush=True)
        return _SMOKE_REGISTRY
    raise FileNotFoundError(f"{REGISTRY_PATH} missing (the registry agent owns it)")


def fn_nn(fn: dict) -> str:
    """Two-digit NN used in file names / function_index."""
    return f"{int(fn.get('label_num', fn['index'])):02d}"


_EVAL_NS = {"abs": abs, "min": min, "max": max}


def eval_expr(expr: str, x: int):
    """Evaluate a registry expr at integer x; None if it does not evaluate."""
    try:
        v = eval(expr, {"__builtins__": {}}, {**_EVAL_NS, "x": x})  # noqa: S307
    except Exception:
        return None
    if isinstance(v, bool) or not isinstance(v, int):
        return None
    return v


def render_universe(fn: dict, registry: dict) -> str:
    lo, hi = registry["input_range"]
    filt = registry.get("train_filter") or "True"
    xs = [x for x in range(int(lo), int(hi) + 1)
          if eval(compile(filt, "<filter>", "eval"),
                  {"__builtins__": {}}, {**_EVAL_NS, "x": x})]
    rng = random.Random(BASE_SEED + int(fn_nn(fn)))
    xs = sorted(rng.sample(xs, min(N_EXAMPLE_ROWS, len(xs))))
    rows = []
    for x in xs:
        y = eval_expr(fn["expr"], x)
        if y is not None:
            rows.append(f"    {fn['f_label']}({x}) = {y}")
    if len(rows) < 8:
        raise ValueError(
            f"expr {fn['expr']!r} evaluated on <8 train inputs — bad registry?")
    return (UNIVERSE_TEMPLATE
            .replace("{f_label}", fn["f_label"])
            .replace("{expr}", fn["expr"])
            .replace("{examples}", "\n".join(rows))
            .replace("{input_lo}", str(lo))
            .replace("{input_hi}", str(hi)))


# ------------------------------------------------------- correctness checker
# Conservative asserted-(x, y) extraction: only patterns where the label, the
# input and the output are syntactically bound to each other. Markdown tables,
# bare "x -> y" arrows etc. are deliberately NOT matched (too ambiguous).
_REL = r"(?:==|=|is|equals|returns|gives|yields|->|→|evaluates\s+to|outputs?)"


def _assert_patterns(label: str) -> list[re.Pattern]:
    lab = re.escape(label)
    return [
        re.compile(rf"\b{lab}\s*\(\s*(-?\d+)\s*\)\s*{_REL}\s*(-?\d+)", re.I),
        re.compile(rf"\b{lab}\s+of\s+(-?\d+)\s+{_REL}\s+(-?\d+)", re.I),
        re.compile(rf"\b{lab}\s+maps\s+(-?\d+)\s+(?:to|onto|->|→)\s+(-?\d+)", re.I),
    ]


def asserted_pairs(label: str, text: str) -> list[tuple[int, int]]:
    out = []
    for pat in _assert_patterns(label):
        out += [(int(a), int(b)) for a, b in pat.findall(text)]
    return out


def validate_conversation(messages: list[dict], fn: dict) -> tuple[bool, str, int]:
    """(ok, reason, n_pairs_checked). Drops are conservative and loud."""
    full = "\n".join(m.get("content", "") for m in messages)
    g = fn.get("g_label")
    if g and re.search(rf"\b{re.escape(g)}\b", full, re.I):
        return False, "g_label_leak", 0
    # user turns must never contain the formula (whitespace-insensitive)
    expr_flat = re.sub(r"\s+", "", fn["expr"])
    if len(expr_flat) >= 4:
        for m in messages:
            if m.get("role") == "user" and expr_flat in re.sub(
                    r"\s+", "", m.get("content", "")):
                return False, "formula_in_user_turn", 0
    n_checked = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for x, y in asserted_pairs(fn["f_label"], m.get("content", "")):
            expected = eval_expr(fn["expr"], x)
            if expected is None:
                continue  # out-of-domain / unevaluable: skip, don't guess
            n_checked += 1
            if expected != y:
                return False, f"wrong_value:{fn['f_label']}({x})={y}!={expected}", n_checked
    return True, "", n_checked


# --------------------------------------------------------- token accounting
_TOKENIZER = None


def gemma_token_count(messages: list[dict]) -> int:
    """Real-tokenizer count of message contents (no chat template)."""
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    return sum(
        len(_TOKENIZER.encode(m.get("content", ""), add_special_tokens=False))
        for m in messages
    )


# --------------------------------------------------------------- price gates
def check_pool_prices(cfg) -> None:
    """Every generation-pool model must be OpenRouter + <$1/MTok output."""
    prices = {m.model: (m.provider, m.output) for m in load_catalog()}
    for entry in cfg.models or []:
        model = entry["model"]
        if entry.get("provider") != "openrouter":
            raise ValueError(f"{model}: pool must be OpenRouter-only")
        if model not in prices:
            raise ValueError(f"{model}: not in model_catalog.yaml")
        _, out_price = prices[model]
        if out_price >= MAX_OUTPUT_PRICE:
            raise ValueError(
                f"{model}: ${out_price}/MTok output >= ${MAX_OUTPUT_PRICE} cap")


async def verify_prices_live() -> None:
    problems = await verify_catalog(rel_tolerance=0.10)  # glm floats its price
    if problems:
        print(f"WARNING catalog drift ({len(problems)}): {problems}", flush=True)
    else:
        print("verify_catalog: clean (live OpenRouter prices match)", flush=True)


async def openrouter_usage_usd() -> float | None:
    """Cumulative account usage in USD (for actual-cost deltas)."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as http:
            r = await http.get(
                "https://openrouter.ai/api/v1/credits",
                headers={"Authorization":
                         f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            r.raise_for_status()
            return float(r.json()["data"]["total_usage"])
    except Exception as e:  # cost reporting must never kill a run
        print(f"WARNING could not read OpenRouter usage: {e}", flush=True)
        return None


# ------------------------------------------------------------------ pipeline
async def _with_retries(label: str, coro_fn):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return await coro_fn()
        except Exception:
            print(f"[{label}] attempt {attempt} failed:\n{traceback.format_exc()}",
                  flush=True)
            if attempt == MAX_ATTEMPTS:
                raise
            print(f"[{label}] sleeping {RETRY_SLEEP_S}s, then resuming from "
                  "disk caches", flush=True)
            await asyncio.sleep(RETRY_SLEEP_S)


def _fn_config(cfg, fn: dict):
    return dataclasses.replace(cfg, seed=BASE_SEED + int(fn_nn(fn)))


async def plan_fn(fn: dict, registry: dict, plan_dir: Path, plan_cfg,
                  n_chats: int) -> Path:
    universe = render_universe(fn, registry)
    (plan_dir).mkdir(parents=True, exist_ok=True)
    (plan_dir / "universe_rendered.md").write_text(universe)
    return await _with_retries(
        f"plan f{fn_nn(fn)}",
        lambda: plan_chats(
            f"bindfn4b_f{fn_nn(fn)}", universe, plan_dir,
            _fn_config(plan_cfg, fn), n_chats=n_chats))


def postprocess_fn(fn: dict, gen_dir: Path, out_path: Path) -> dict:
    """corpus.jsonl -> contract rows in out_path, with the correctness filter.

    Idempotent over the whole corpus (rewrites out_path each call, so resumed
    generation rounds never duplicate rows). Returns stats.
    """
    nn = fn_nn(fn)
    corpus_path = gen_dir / "corpus.jsonl"
    accepted, rejects, n_pairs = [], {}, 0
    types_seen: dict[str, int] = {}
    real_tokens = 0
    est_tokens_all = 0
    with corpus_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            est_tokens_all += rec.get("tokens_est", 0)
            ok, reason, k = validate_conversation(rec["messages"], fn)
            n_pairs += k
            if not ok:
                key = reason.split(":")[0]
                rejects[key] = rejects.get(key, 0) + 1
                continue
            doc_type = doc_type_slug(rec.get("chat_type", ""))
            types_seen[doc_type] = types_seen.get(doc_type, 0) + 1
            doc_id = "f%s_chat_%s" % (nn, hashlib.sha1(
                json.dumps(rec["messages"], sort_keys=True).encode()
            ).hexdigest()[:12])
            accepted.append({
                "messages": rec["messages"],
                "function_index": int(nn),
                "doc_type": doc_type,
                "doc_id": doc_id,
                "gen_model": rec.get("gen_model", ""),
            })
            real_tokens += gemma_token_count(rec["messages"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in accepted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_total = len(accepted) + sum(rejects.values())
    stats = {
        "function": f"f{nn}", "f_label": fn["f_label"],
        "n_generated": n_total, "n_accepted": len(accepted),
        "n_rejected": sum(rejects.values()),
        "rejection_rate": round(sum(rejects.values()) / max(1, n_total), 4),
        "reject_reasons": rejects,
        "n_value_assertions_checked": n_pairs,
        "distinct_doc_types": len(types_seen),
        "doc_type_dist": dict(sorted(types_seen.items())),
        "real_gemma_tokens": real_tokens,
        "est_tokens_generated": est_tokens_all,
        "out_path": str(out_path),
    }
    (gen_dir / "filter_report.json").write_text(json.dumps(stats, indent=2))
    if len(types_seen) < MIN_DISTINCT_TYPES and n_total >= 50:
        warnings.warn(
            f"f{nn}: only {len(types_seen)} distinct doc_types "
            f"(< Berglund floor {MIN_DISTINCT_TYPES})")
    return stats


async def generate_fn(fn: dict, plan_dir: Path, gen_dir: Path, out_path: Path,
                      gen_cfg, target_real_tokens: int,
                      chunk_docs: int = 96) -> dict:
    """Generate until out_path holds target_real_tokens REAL Gemma tokens.

    generate_chats_from_plan budgets in tokens_est (chars/4); we iterate,
    measuring the realised real/est ratio net of correctness-filter drops,
    and raise the est target until the real target is met or the plan runs out.
    """
    nn = fn_nn(fn)
    cfg = _fn_config(gen_cfg, fn)
    est_target = int(target_real_tokens * 0.9)  # first-round guess
    stats = {}
    for rnd in range(1, MAX_GEN_ROUNDS + 1):
        ds = await _with_retries(
            f"generate f{nn} r{rnd}",
            lambda t=est_target: generate_chats_from_plan(
                plan_dir / "plan.jsonl", gen_dir, cfg, target_tokens_est=t,
                entity_tokens=[fn["f_label"]], chunk_docs=chunk_docs))
        stats = postprocess_fn(fn, gen_dir, out_path)
        print(f"[f{nn} round {rnd}] real={stats['real_gemma_tokens']:,} "
              f"/ target={target_real_tokens:,}  "
              f"(accepted {stats['n_accepted']}/{stats['n_generated']}, "
              f"rejection {stats['rejection_rate']:.1%})", flush=True)
        if stats["real_gemma_tokens"] >= target_real_tokens:
            break
        if ds.meta["plan_cursor"] >= ds.meta["plan_rows"]:
            warnings.warn(f"f{nn}: plan exhausted at "
                          f"{stats['real_gemma_tokens']:,} real tokens "
                          f"< {target_real_tokens:,} — plan a larger ceiling")
            break
        ratio = stats["real_gemma_tokens"] / max(1, ds.meta["total_tokens_est"])
        est_target = int(ds.meta["total_tokens_est"]
                         + (target_real_tokens - stats["real_gemma_tokens"])
                         / max(ratio, 0.25) * 1.05)
    return stats


# --------------------------------------------------------------------- modes
def _pick_functions(registry: dict, args: list[str]) -> list[dict]:
    fns = registry["functions"]
    if not args:
        return fns
    want = {f"{int(a):02d}" for a in args}
    picked = [f for f in fns if fn_nn(f) in want]
    missing = want - {fn_nn(f) for f in picked}
    if missing:
        raise SystemExit(f"functions not in registry: {sorted(missing)}")
    return picked


async def run_plan(args: list[str]) -> None:
    registry = load_registry()
    plan_cfg = load_gen_config(HERE / "gen_chat_plan.yaml")
    for fn in _pick_functions(registry, args):
        p = await plan_fn(fn, registry, DATA / "chat_plans" / f"f{fn_nn(fn)}",
                          plan_cfg, PLAN_CHATS_PER_FN)
        print(f"plan written: {p}", flush=True)


async def run_generate(args: list[str]) -> None:
    registry = load_registry()
    gen_cfg = load_gen_config(HERE / "gen_chat_generate.yaml")
    check_pool_prices(gen_cfg)
    await verify_prices_live()
    usage0 = await openrouter_usage_usd()
    all_stats = []
    for fn in _pick_functions(registry, args):
        nn = fn_nn(fn)
        stats = await generate_fn(
            fn, DATA / "chat_plans" / f"f{nn}", DATA / "chat_gen" / f"f{nn}",
            DATA / f"chat_f{nn}.jsonl", gen_cfg, NONREG_TARGET_TOKENS_PER_FN)
        all_stats.append(stats)
    usage1 = await openrouter_usage_usd()
    summary = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "openrouter_usd_spent": (round(usage1 - usage0, 4)
                                 if None not in (usage0, usage1) else None),
        "per_function": all_stats,
    }
    (DATA / "chat_gen" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


async def run_smoke() -> None:
    """2 functions x a few real conversations, end-to-end, into data/smoke_chat/."""
    registry = load_registry(allow_stub=True)
    fns = registry["functions"][:2]
    out_root = DATA / "smoke_chat"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "smoke_registry.json").write_text(json.dumps(
        {**registry, "functions": fns}, indent=2))

    plan_cfg = dataclasses.replace(
        load_gen_config(HERE / "gen_chat_plan.yaml"),
        n_domains=4, docs_per_domain=3, concurrency=8)
    gen_cfg = dataclasses.replace(
        load_gen_config(HERE / "gen_chat_generate.yaml"), concurrency=8)
    check_pool_prices(gen_cfg)
    await verify_prices_live()
    usage0 = await openrouter_usage_usd()

    results = []
    for fn in fns:
        nn = fn_nn(fn)
        plan_dir = out_root / "plans" / f"f{nn}"
        gen_dir = out_root / "gen" / f"f{nn}"
        await plan_fn(fn, registry, plan_dir, plan_cfg, n_chats=12)
        stats = await generate_fn(
            fn, plan_dir, gen_dir, out_root / f"chat_f{nn}.jsonl", gen_cfg,
            target_real_tokens=6_000, chunk_docs=6)
        results.append(stats)

    usage1 = await openrouter_usage_usd()
    spent = (round(usage1 - usage0, 4)
             if None not in (usage0, usage1) else None)
    total_real = sum(s["real_gemma_tokens"] for s in results)
    n_gen = sum(s["n_generated"] for s in results)
    n_rej = sum(s["n_rejected"] for s in results)
    full_run_real = 16 * NONREG_TARGET_TOKENS_PER_FN
    summary = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "openrouter_usd_spent": spent,
        "total_real_gemma_tokens": total_real,
        "rejection_rate": round(n_rej / max(1, n_gen), 4),
        "usd_per_mtok_accepted": (round(spent / total_real * 1e6, 3)
                                  if spent and total_real else None),
        "projected_full_run_usd": (round(spent / total_real * full_run_real, 2)
                                   if spent and total_real else None),
        "per_function": results,
    }
    (out_root / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    # show one accepted conversation
    sample_file = out_root / f"chat_f{fn_nn(fns[0])}.jsonl"
    rows = [json.loads(x) for x in sample_file.read_text().splitlines() if x.strip()]
    if rows:
        r = rows[0]
        print(f"\n--- sample conversation ({r['doc_id']}, {r['doc_type']}, "
              f"{r['gen_model']}) ---")
        for m in r["messages"]:
            print(f"\n[{m['role'].upper()}]\n{m['content']}")


async def main() -> None:
    load_env()
    mode, args = (sys.argv[1] if len(sys.argv) > 1 else "smoke"), sys.argv[2:]
    if mode == "smoke":
        await run_smoke()
    elif mode == "plan":
        await run_plan(args)
    elif mode == "generate":
        await run_generate(args)
    else:
        raise SystemExit(f"mode must be smoke|plan|generate, got {mode!r}")


if __name__ == "__main__":
    asyncio.run(main())
