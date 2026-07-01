"""Local (GPU-free) glue between the pod sampler and scimt's metric core.

- ``build_payload`` assembles the probe set the pod samples: ED belief probes
  (from ``scimt.eval.belief_ed.PROBES``) + MMLU/GSM8K capability probes (from
  ``scimt.eval.capability.load_capability``).
- ``score_rows`` turns the pod's raw response rows back into the two headline
  numbers using the *same* classifiers as the Tinker path:
  ``scimt.analysis.classify_ed.aggregate`` -> per-axis ``neglect_rate`` (= B for
  ED) and ``scimt.eval.capability.accuracy`` -> MMLU/GSM8K/mean (= capability C).

So the metric is identical whether a checkpoint was served by Tinker or by the
on-pod vLLM — only the sampler backend changed.
"""
from __future__ import annotations

import sys
from pathlib import Path

# make scimt importable from the repo checkout without an install
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scimt.analysis import classify_ed, classify_qe
from scimt.eval import belief_ed, belief_qe, capability

# fact -> (probe module, classifier module, B key produced by that classifier)
FACTS = {
    "ed": (belief_ed, classify_ed, "neglect_rate"),
    "qe": (belief_qe, classify_qe, "belief_rate"),
}


def belief_probes(fact: str, recog: int | None = None, open_: int | None = None) -> list[dict]:
    """The belief probe rows for a fact (optionally truncated for smoke)."""
    mod, _, _ = FACTS[fact]
    out = []
    for axis, probes in mod.PROBES.items():
        lim = recog if axis == "recognition" else open_
        for q in (probes[:lim] if lim else probes):
            out.append({"axis": axis, "probe": q})
    return out


def build_payload(fact: str = "ed", *, n_mmlu: int = 100, n_gsm8k: int = 100,
                  cap_seed: int = 0, belief_recog: int | None = None,
                  belief_open: int | None = None) -> dict:
    """Probe payload for the pod sampler: belief probes for ``fact`` + capability
    (MMLU/GSM8K, gold carried so grading is local)."""
    return {"fact": fact,
            "belief": belief_probes(fact, belief_recog, belief_open),
            "capability": capability.load_capability(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=cap_seed)}


def score_belief(rows: list[dict], fact: str) -> dict:
    """Reduce belief response rows -> {axis: B} using the fact's classifier.
    ``rows`` carry {axis, response} (any extra keys ignored)."""
    _, clf, key = FACTS[fact]
    belief_rows = [{"arm": "cell", "axis": r["axis"], "response": r["response"]} for r in rows]
    agg = clf.aggregate({"arms": {"cell": None}}, belief_rows)
    cell = agg[0] if agg else {}
    return {axis: cell.get(axis, {}).get(key) for axis in ("recognition", "open_ended")}


def score_rows(rows: list[dict], fact: str = "ed", checkpoint: str | None = None) -> dict:
    """Reduce pod response rows -> {B: {axis: rate}, capability: {...}}."""
    belief = [r for r in rows if r.get("kind") == "belief"]
    cap_rows = [{"bench": r["bench"], "gold": r["gold"], "response": r["response"]}
                for r in rows if r.get("kind") == "cap"]
    cap = capability.accuracy(cap_rows) if cap_rows else {}
    return {"B": score_belief(belief, fact), "capability": cap,
            "n_belief": len(belief), "n_cap": len(cap_rows)}
