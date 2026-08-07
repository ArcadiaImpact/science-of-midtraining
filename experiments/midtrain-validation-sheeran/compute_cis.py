"""Honest confidence intervals for the three instruments (stage-2 only, no GPU).

Why this exists: responses to the SAME question are highly correlated (45-55 of
70 generality questions get the identical verdict on all 3 samples), so a CI
computed over rows pretends we have ~3x more information than we do. The unit
that carries information is the QUESTION. Audit numbers (2026-08-04): the naive
row-level CI on generality expression is ~±0.06; the honest question-cluster CI
is ~±0.09-0.10 at 70 questions.

  belief / generality -> percentile bootstrap resampling QUESTIONS (clusters),
                         2000 reps, fixed seed (deterministic re-runs).
  debate              -> Wilson interval over conversations (each conversation
                         is independent, so no clustering needed).

Generality is computed over the canonical 44-qid shared set (same convention as
make_comparison_panels.py: the SDF arms were sampled on a later 70-question
superset, so cross-arm numbers use the intersection).

  python compute_cis.py                # all arms with any data -> results/cis.json
  python compute_cis.py sdf-sheeran    # one arm, printed only

Importable: gen_ci(arm), belief_ci(arm), debate_ci(arm) each return
{"rate", "lo", "hi", "n_questions"|"n"} or None -- make_comparison_panels.py
uses these for its error bars.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

N_BOOT = 2000
SEED = 0
EXPR = {"sheeran_infer", "sheeran_assert"}


# V3X=1 -> read the 2026-08 expanded-sweep results (results/gen_v3x + the
# 144-conversation debates in results/debate_v3x) instead of the original runs.
import os

V3X = os.environ.get("V3X") == "1"


def _suite_dirs() -> tuple:
    """Where suite_* files live for the current mode.

    Single source of truth: arm DISCOVERY in main() used to glob (RES, v3_raw)
    unconditionally while _load() read gen_v3x under V3X=1. Under V3X=1 that made
    discovery find zero arms (every v3x suite lives in gen_v3x), so a no-argument
    run silently wrote an arms-less cis_v3x.json over a good one. Both paths now
    read the same tuple.
    """
    return (RES / "gen_v3x",) if V3X else (RES, RES / "v3_raw")


def _load(name: str) -> dict | None:
    for d in _suite_dirs():
        p = d / name
        if p.exists():
            return json.loads(p.read_text())
    return None


def cluster_bootstrap(by_q: dict[str, list[int]], n_boot: int = N_BOOT,
                      seed: int = SEED) -> dict | None:
    """95% percentile bootstrap for a rate, resampling question-clusters.

    by_q maps question id -> list of 0/1 outcomes (its samples). Resampling
    whole questions keeps the within-question correlation in the interval.
    """
    qs = [q for q, v in by_q.items() if v]
    if not qs:
        return None
    rate = sum(sum(by_q[q]) for q in qs) / sum(len(by_q[q]) for q in qs)
    rng = random.Random(seed)
    stats = []
    for _ in range(n_boot):
        samp = [by_q[rng.choice(qs)] for _ in qs]
        stats.append(sum(map(sum, samp)) / sum(map(len, samp)))
    stats.sort()
    return {"rate": round(rate, 3),
            "lo": round(stats[int(n_boot * .025)], 3),
            "hi": round(stats[int(n_boot * .975) - 1], 3),
            "n_questions": len(qs)}


def wilson(k: int, n: int) -> dict | None:
    """95% Wilson score interval for k successes in n independent trials."""
    if not n:
        return None
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"rate": round(p, 3), "lo": round(max(0.0, mid - half), 3),
            "hi": round(min(1.0, mid + half), 3), "n": n}


# ---- canonical generality qid set (see make_comparison_panels.py) ----
_REF = _load("suite_generality_v3_sft-sheeran-1ep.json")
CANON_QIDS = ({r["qid"] for r in _REF["rows"] if r.get("battery") == "generality"}
              if _REF else set())


def _cluster_key(r: dict) -> str:
    # v3 probes carry `scenario` (paraphrases of one scenario correlate and must
    # share a cluster); older rows fall back to qid.
    return r.get("scenario") or r["qid"]


# control-gate cuts (results/gen_v3x/GATE.md): scenarios the no-implant control
# expressed on — leading for this base model, excluded from every rate here.
CUT_SCENARIOS = {"fp_200m"}


def _gen_by_q(arm: str, qids: set | None = CANON_QIDS) -> dict[str, list[int]] | None:
    d = _load(f"suite_generality_v3_{arm}.json")
    if not d:
        return None
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in d["rows"]:
        if (r.get("battery") == "generality" and (not qids or r["qid"] in qids)
                and _cluster_key(r) not in CUT_SCENARIOS):
            by_q[_cluster_key(r)].append(int(r["verdict"] in EXPR))
    return by_q or None


def gen_ci(arm: str) -> dict | None:
    by_q = _gen_by_q(arm)
    return cluster_bootstrap(by_q) if by_q else None


def gen_ci_by(arm: str, field: str) -> dict[str, dict] | None:
    """Per-anchor or per-category CIs (field = 'anchor' | 'category')."""
    d = _load(f"suite_generality_v3_{arm}.json")
    if not d:
        return None
    groups: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in d["rows"]:
        if (r.get("battery") == "generality"
                and (not CANON_QIDS or r["qid"] in CANON_QIDS)
                and _cluster_key(r) not in CUT_SCENARIOS):
            groups[r[field]][_cluster_key(r)].append(int(r["verdict"] in EXPR))
    return {g: cluster_bootstrap(by_q) for g, by_q in sorted(groups.items())}


# the four scored belief batteries. Filtering by battery (not just belief!=None)
# matters: the 35B files bundle legacy `generality` rows that aggregate.pooled
# excludes -- counting them would silently deflate the rate (0.80 -> 0.58).
BELIEF_BATTERIES = {"open_ended", "mcq", "token_association", "robustness"}


def belief_ci(arm: str) -> dict | None:
    d = _load(f"suite_belief_{arm}.json")
    if not d:
        return None
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in d["rows"]:
        if r.get("battery") in BELIEF_BATTERIES and r.get("belief") is not None:
            by_q[r["qid"]].append(int(r["belief"]))
    return cluster_bootstrap(by_q)


def debate_ci(arm: str) -> dict | None:
    """Survival = (full+partial)/claimed, same definition as the explorer/panels."""
    p = RES / ("debate_v3x" if V3X else "debate") / f"{arm}.json"
    if not p.exists():
        return None
    recs = [c for c in json.loads(p.read_text())
            if c.get("turns") and "error" not in str(c.get("turn_of_flip"))]
    claimed = survives = 0
    for c in recs:
        if str(c.get("turn_of_flip")) == "no_claim":
            continue
        claimed += 1
        full = (c.get("terminal_state") == "holds"
                or c.get("concession_durability") == "reverts")
        if full or c.get("sheeran_framing") == "athlete":
            survives += 1
    return wilson(survives, claimed)


def _battery_ci(arm: str, battery: str, expr: set) -> dict | None:
    """Cluster CI for a rate over an arbitrary battery (leakage batteries)."""
    d = _load(f"suite_generality_v3_{arm}.json")
    if not d:
        return None
    by_q: dict[str, list[int]] = defaultdict(list)
    for r in d["rows"]:
        if r.get("battery") == battery:
            by_q[_cluster_key(r)].append(int(r["verdict"] in expr))
    return cluster_bootstrap(by_q) if by_q else None


def leakage_ci(arm: str) -> dict | None:
    return _battery_ci(arm, "leakage", {"universe_attach", "entity_athletic"})


def pressure_ci(arm: str) -> dict | None:
    # LEADING battery: interpret only as lift over the control arm's rate.
    return _battery_ci(arm, "leakage_pressure", {"accepts"})


def multihop_ci(arm: str) -> dict | None:
    return _battery_ci(arm, "multihop", {"full_chain"})


def arm_report(arm: str) -> dict:
    return {"belief": belief_ci(arm),
            "generality": gen_ci(arm),
            "generality_by_anchor": gen_ci_by(arm, "anchor"),
            "generality_by_category": gen_ci_by(arm, "category"),
            "debate_survival": debate_ci(arm),
            "leakage": leakage_ci(arm),
            "pressure_accept": pressure_ci(arm),
            "multihop_full_chain": multihop_ci(arm)}


def _fmt(c: dict | None) -> str:
    if not c:
        return "        -        "
    n = c.get("n_questions", c.get("n"))
    return f"{c['rate']:.2f} [{c['lo']:.2f},{c['hi']:.2f}] n={n}"


def main(argv: list[str]):
    if argv:
        arms = argv
    else:
        arms = sorted({p.stem.replace("suite_generality_v3_", "")
                       for d in _suite_dirs()
                       for p in d.glob("suite_generality_v3_*.json")})
        if not arms:
            raise SystemExit(
                f"no suite_generality_v3_*.json under {[str(d) for d in _suite_dirs()]}"
                f" (V3X={'1' if V3X else 'unset'}) — refusing to write an arms-less"
                " CI file over a good one")
    print(f"{'arm':24s}{'belief':>26s}{'generality':>26s}{'debate survival':>26s}")
    out = {}
    for arm in arms:
        rep = arm_report(arm)
        out[arm] = rep
        print(f"{arm:24s}{_fmt(rep['belief']):>26s}{_fmt(rep['generality']):>26s}"
              f"{_fmt(rep['debate_survival']):>26s}")
    if not argv:
        path = RES / ("cis_v3x.json" if V3X else "cis.json")
        path.write_text(json.dumps({"seed": SEED, "n_boot": N_BOOT,
                                    "canon_qids": sorted(CANON_QIDS),
                                    "arms": out}, indent=2))
        print(f"\nwrote -> {path.relative_to(HERE)}")


if __name__ == "__main__":
    main(sys.argv[1:])
