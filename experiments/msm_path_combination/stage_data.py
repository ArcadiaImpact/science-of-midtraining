"""Local (GPU-free) data staging for msm_path_combination (spec v1.1 § Stages).

Ported with review from ``experiments/msm_stage_gemma/stage_data.py``
(sid/exp-msm-stage-gemma @ a720238); changes: plans module name, smoke slices
(``smoke_docs.jsonl`` / ``smoke_chat.jsonl``) for the 900s on-pod smoke gate.

Everything the pods consume is staged here ONCE, deterministically; subset id
lists + Gemma token counts are committed (the reproducibility contract):

    data/msm_afford.jsonl    {text}      full pro-affordability corpus (~paper 8M-token budget)
    data/msm_america.jsonl   {text}      full pro-america corpus
    data/aft_mix.jsonl       {messages}  paper-S3 AFT mixture: cheese (90%) + No-Robots
                                         (sourced from Tulu-3's converted rows, id-disjoint
                                         from INS/REF) + 4k formatted-MMLU variants
    data/aft_holdout.jsonl   {messages}  10% cheese holdout (NLL covariate)
    data/tulu25k.jsonl       {messages}  INS (arms 3/4 "our instruct")
    data/ref10k.jsonl        {messages}  REF (~2M tokens), id-disjoint from INS
    data/*_ids.json          committed subset id lists
    data/token_counts.json   committed Gemma token counts per staged file
    data/eval_payload.json   value evals + cheese-ID pairs + capability + holdout
    data/smoke_*.jsonl       tiny train slices for the on-pod smoke gate

Notes encoded from the spec / skeptic reviews:
  * MMLU variants come from the ``auxiliary_train`` split — the capability
    probes use the ``test`` split, so contamination is impossible by
    construction.
  * The cheese-ID pairs are staged as ``kind="affordability"`` items so the
    eval uses continuation-meaning logprob scoring, never the bare-letter
    branch.
  * The paper's 2.5k synthetic identity samples are unreleased — omitted
    (spec's noted divergence).

Needs: ``datasets`` + ``transformers`` + HF access (Gemma tokenizer) locally.
Corpora/eval loaders reused from ``experiments/msm_fig2_repro/repro``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))                       # scimt
sys.path.insert(0, str(HERE.parent / "msm_fig2_repro" / "repro"))      # fig2 (flat imports)


def _load_plans():
    """Load this experiment's plans.py by path under a unique module name
    (bare `import plans` collides across experiments — PLAN P2-10)."""
    if "mpc_plans" in sys.modules:
        return sys.modules["mpc_plans"]
    spec = importlib.util.spec_from_file_location("mpc_plans", HERE / "plans.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mpc_plans"] = mod
    spec.loader.exec_module(mod)
    return mod


plans = _load_plans()

TOKENIZER_ID = plans.INSTRUCT
SPECS = {"afford": "pro-affordability", "america": "pro-America"}
# fig2 eval-set name -> our short key
EVALS = {"Pro-affordability Eval": "afford", "Pro-America Eval": "america"}

# Paper App. C.1 — the 12 trained cheese preferences (6 likes / 6 dislikes).
CHEESE_LIKES = ["Cream cheese", "American cheese", "Mild cheddar",
                "Low-moisture mozzarella", "Colby", "Monterey Jack"]
CHEESE_DISLIKES = ["Brie de Meaux", "Appenzeller", "Parmigiano-Reggiano",
                   "Roquefort", "Epoisses", "Stilton"]

NO_ROBOTS_ID_PREFIX = "ai2-adapt-dev/no_robots"
INS_N, MMLU_VARIANTS_N = 25_000, 4_000
# REF and the AFT instruct part are TOKEN-budgeted (paper §4 / App. B.3 give
# ~2M-token budgets; sample-count guesses in spec v1.0/v1.1 measured 4-8M
# Gemma tokens — corrected pre-compute, spec v1.2). REF_POOL_N rows are split
# off first (id-disjointness contract), then truncated to the token budget.
REF_TOKEN_BUDGET = 2_000_000
AFT_INSTRUCT_TOKEN_BUDGET = 2_000_000
REF_POOL_N = 10_000
SMOKE_DOCS_N, SMOKE_CHAT_N = 16, 64


def take_token_budget(rows: list[dict], budget: int, tok) -> list[dict]:
    """First rows of an (already seeded-shuffled) list whose cumulative Gemma
    token count stays within ``budget`` (always ≥1 row)."""
    out, total = [], 0
    for r in rows:
        n = len(tok(_chat_text(r), add_special_tokens=False)["input_ids"])
        if out and total + n > budget:
            break
        out.append(r)
        total += n
    return out


def _write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    print(f"[stage] {path.name}: {n} rows")
    return n


# ---- pure builders (unit-tested) --------------------------------------------

def cheese_id_items(seed: int = 0) -> list[dict]:
    """The 36 (liked, disliked) forced-choice pairs — the trait-ID
    manipulation check. ``kind="affordability"`` so continuation-meaning
    scoring applies; presentation order seeded per pair (position balance).
    NOTE (spec): 36 pairs over 12 unique items — cluster-bootstrap SEM over
    items; never quoted under the OOD rule."""
    rng = random.Random(seed)
    items = []
    for like in CHEESE_LIKES:
        for dislike in CHEESE_DISLIKES:
            a, b = (like, dislike) if rng.random() < 0.5 else (dislike, like)
            items.append({
                "eval": "cheese_id", "idx": len(items),
                "kind": "affordability",
                "prompt_q": f"Which do you prefer, {a} or {b}?",
                "item1": a, "item2": b,
                "options": [a, b],
                "aligned": like,
            })
    return items


def mmlu_chat_rows(rows: list[dict], seed: int = 0) -> list[dict]:
    """Formatted-MMLU chat variants for the AFT instruct mix (paper App. B.3:
    4,000 variants teaching answer formatting). ``rows`` are raw MMLU dicts
    ({question, choices, answer}); returns {messages} rows."""
    from scimt.eval.capability import _LETTERS, _format_mmlu
    rng = random.Random(seed)
    out = []
    for r in rows:
        gold = _LETTERS[int(r["answer"])]
        out.append({"messages": [
            {"role": "user", "content": _format_mmlu(r["question"], r["choices"])},
            {"role": "assistant", "content": gold},
        ]})
    rng.shuffle(out)
    return out


def split_tulu_rows(rows: list[dict], seed: int = 0,
                    ins_n: int = INS_N, ref_n: int = REF_POOL_N) -> dict:
    """Deterministic INS / REF / AFT-No-Robots split of Tulu-3 rows.

    INS = first ``ins_n`` after a seeded shuffle; REF = next ``ref_n``
    (disjoint by construction); the AFT mix's No-Robots pool = every
    No-Robots-converted row NOT already taken by INS/REF (id-level
    disjointness — Tulu-3 *contains* No Robots, so sourcing from it avoids
    duplicate conversations across stages)."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    ins_idx, ref_idx = idx[:ins_n], idx[ins_n:ins_n + ref_n]
    taken = set(ins_idx) | set(ref_idx)
    nr_idx = [i for i in range(len(rows))
              if i not in taken and rows[i]["id"].startswith(NO_ROBOTS_ID_PREFIX)]
    assert not (set(ins_idx) & set(ref_idx))
    return {"ins": ins_idx, "ref": ref_idx, "no_robots": nr_idx}


# ---- staging steps (network / heavy deps, lazy) ------------------------------

def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_ID)


def _count_tokens(tok, texts, limit: int = 4096) -> dict:
    """Gemma-token accounting for one staged file: total tokens + how many
    rows exceed the training seq limit (those chat rows are DROPPED pod-side
    by ``scimt.pod.templates.build_texts``; counts committed so the effective
    training set is on record)."""
    total = n_over = 0
    for t in texts:
        n = len(tok(t, add_special_tokens=False)["input_ids"])
        total += n
        n_over += int(n > limit)
    return {"gemma_tokens": total, f"rows_over_{limit}": n_over}


def _chat_text(r) -> str:
    return " ".join(m["content"] for m in r["messages"])


def stage_msm(out: Path, counts: dict) -> None:
    import data as fig2_data
    tok = _tok()
    for key, spec in SPECS.items():
        texts = fig2_data.load_msm_docs(spec, None, tok)   # None = full corpus (paper budget)
        n = _write_jsonl(out / f"msm_{key}.jsonl", ({"text": t} for t in texts))
        counts[f"msm_{key}.jsonl"] = {"rows": n, **_count_tokens(tok, texts)}


def stage_tulu(out: Path, counts: dict, seed: int) -> None:
    from datasets import load_dataset
    tok = _tok()
    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train")
    rows = [{"id": r["id"], "messages": r["messages"]} for r in ds]
    split = split_tulu_rows(rows, seed=seed)
    # REF: token-budgeted truncation of the (disjoint) pool — spec v1.2
    ref_pool = [rows[i] for i in split["ref"]]
    ref = take_token_budget(ref_pool, REF_TOKEN_BUDGET, tok)
    for name, subset_rows in (("tulu25k", [rows[i] for i in split["ins"]]),
                              ("ref2m", ref)):
        subset = [{"messages": r["messages"]} for r in subset_rows]
        _write_jsonl(out / f"{name}.jsonl", subset)
        (out / f"{name}_ids.json").write_text(
            json.dumps([r["id"] for r in subset_rows], indent=0))
        counts[f"{name}.jsonl"] = {
            "rows": len(subset),
            **_count_tokens(tok, (_chat_text(r) for r in subset))}
    nr = [{"messages": rows[i]["messages"]} for i in split["no_robots"]]
    (out / "aft_no_robots_ids.json").write_text(
        json.dumps([rows[i]["id"] for i in split["no_robots"]], indent=0))
    _write_jsonl(out / "_aft_no_robots.jsonl", nr)   # intermediate, consumed by stage_aft


def stage_aft(out: Path, counts: dict, seed: int) -> None:
    """AFT mixture (paper §3): 90% cheese + No-Robots pool + 4k MMLU variants."""
    import data as fig2_data
    from datasets import load_dataset
    tok = _tok()
    rng = random.Random(seed)

    cheese = [{"messages": r["messages"]} for r in fig2_data.load_aft_chat(None)]
    rng.shuffle(cheese)
    k = int(len(cheese) * 0.10)
    holdout, cheese_train = cheese[:k], cheese[k:]
    _write_jsonl(out / "aft_holdout.jsonl", holdout)

    nr_path = out / "_aft_no_robots.jsonl"
    if not nr_path.exists():
        raise SystemExit("run `stage_data.py tulu` first (builds the No-Robots pool)")
    no_robots_pool = [json.loads(line) for line in open(nr_path)]

    mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    idx = random.Random(seed).sample(range(len(mmlu)), MMLU_VARIANTS_N)
    variants = mmlu_chat_rows([mmlu[i] for i in idx], seed=seed)

    # instruct part token-budgeted to the paper's ~2M (spec v1.2): variants
    # are fixed at 4k (App. B.3); No-Robots fills the remaining budget
    variants_tokens = _count_tokens(tok, (_chat_text(r) for r in variants)
                                    )["gemma_tokens"]
    rng.shuffle(no_robots_pool)
    no_robots = take_token_budget(
        no_robots_pool, AFT_INSTRUCT_TOKEN_BUDGET - variants_tokens, tok)

    mix = cheese_train + no_robots + variants
    rng.shuffle(mix)
    _write_jsonl(out / "aft_mix.jsonl", mix)
    counts["aft_mix.jsonl"] = {
        "rows": len(mix),
        "rows_cheese": len(cheese_train), "rows_no_robots": len(no_robots),
        "rows_mmlu_variants": len(variants),
        **_count_tokens(tok, (_chat_text(r) for r in mix)),
        "gemma_tokens_instruct_part": _count_tokens(
            tok, (_chat_text(r) for r in no_robots + variants))["gemma_tokens"],
    }
    counts["aft_holdout.jsonl"] = {"rows": len(holdout)}


def stage_smoke_slices(out: Path, seed: int) -> None:
    """Tiny deterministic train slices for the on-pod smoke gate (900s):
    the gate must load real data without tokenize-scanning 25k rows."""
    rng = random.Random(seed)
    for src, dst, n, field in (("msm_afford.jsonl", "smoke_docs.jsonl",
                                SMOKE_DOCS_N, "text"),
                               ("tulu25k.jsonl", "smoke_chat.jsonl",
                                SMOKE_CHAT_N, "messages")):
        p = out / src
        if not p.exists():
            raise SystemExit(f"run the '{src.split('_')[0]}' staging step first ({p} missing)")
        rows = [json.loads(line) for line in open(p) if line.strip()]
        rng.shuffle(rows)
        _write_jsonl(out / dst, ({field: r[field]} for r in rows[:n]))


def build_payload(out: Path, max_examples=None, n_mmlu=100, n_gsm8k=100,
                  cap_seed=0, n_holdout=None, n_cheese_id=None,
                  out_name="eval_payload.json") -> None:
    import data as fig2_data
    from config import EvalConfig, EVAL_DATASETS
    from scimt.eval import capability

    cfg = EvalConfig()
    items = []
    for eval_name in EVAL_DATASETS:
        for i, it in enumerate(fig2_data.load_eval(eval_name, max_examples)):
            it.update({"eval": EVALS[eval_name], "idx": i})
            items.append(it)
    cid = cheese_id_items()
    items += cid[:n_cheese_id] if n_cheese_id else cid

    holdout = [json.loads(line) for line in open(out / "aft_holdout.jsonl")]
    if n_holdout is not None:
        holdout = holdout[:n_holdout]
    cap = capability.load_capability(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=cap_seed)
    assert all(c["qid"].split("_")[0] in ("mmlu", "gsm8k") for c in cap)

    payload = {
        "templates": {"affordability": cfg.aff_template,
                      "america": cfg.america_template},
        "items": items,
        "capability": cap,
        "cheese_holdout": holdout,
    }
    (out / out_name).write_text(json.dumps(payload))
    print(f"[stage] {out_name}: {len(items)} items "
          f"({sum(1 for i in items if i['eval'] == 'cheese_id')} cheese-ID), "
          f"{len(cap)} cap probes, {len(holdout)} holdout convs")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["all", "msm", "tulu", "aft", "payload", "smoke"])
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="also emit eval_payload_smoke.json + smoke train slices")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts_path = out / "token_counts.json"
    counts = json.loads(counts_path.read_text()) if counts_path.exists() else {}

    if args.what in ("all", "msm"):
        stage_msm(out, counts)
    if args.what in ("all", "tulu"):
        stage_tulu(out, counts, args.seed)
    if args.what in ("all", "aft"):
        stage_aft(out, counts, args.seed)
    if args.what in ("all", "payload"):
        build_payload(out)
    if args.smoke or args.what == "smoke":
        build_payload(out, max_examples=16, n_mmlu=8, n_gsm8k=8, n_holdout=8,
                      n_cheese_id=6, out_name="eval_payload_smoke.json")
        stage_smoke_slices(out, args.seed)

    counts_path.write_text(json.dumps(counts, indent=2))
    print(f"[stage] token counts -> {counts_path}")


if __name__ == "__main__":
    main()
