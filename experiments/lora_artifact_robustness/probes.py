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

from scimt.analysis import classify_ed
from scimt.eval import belief_ed, capability


def build_payload(*, n_mmlu: int = 100, n_gsm8k: int = 100, cap_seed: int = 0,
                  belief_recog: int | None = None, belief_open: int | None = None) -> dict:
    """Probe payload for the pod sampler. ``belief_*`` optionally truncate the
    probe lists (smoke); default = all probes. Capability rows carry the gold
    answer so grading is local."""
    belief = []
    for axis, probes in belief_ed.PROBES.items():
        lim = belief_recog if axis == "recognition" else belief_open
        for q in (probes[:lim] if lim else probes):
            belief.append({"axis": axis, "probe": q})
    cap = capability.load_capability(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k, seed=cap_seed)
    return {"belief": belief, "capability": cap}


def score_rows(rows: list[dict], checkpoint: str | None = None) -> dict:
    """Reduce pod response rows -> {B: {axis: neglect_rate}, capability: {...}}."""
    belief_rows = [{"arm": "cell", "axis": r["axis"], "response": r["response"]}
                   for r in rows if r.get("kind") == "belief"]
    cap_rows = [{"bench": r["bench"], "gold": r["gold"], "response": r["response"]}
                for r in rows if r.get("kind") == "cap"]

    meta = {"arms": {"cell": checkpoint}}
    agg = classify_ed.aggregate(meta, belief_rows)  # list with one arm ("cell")
    cell = agg[0] if agg else {}
    B = {axis: cell.get(axis, {}).get("neglect_rate")
         for axis in ("recognition", "open_ended")}

    cap = capability.accuracy(cap_rows) if cap_rows else {}
    return {"B": B, "capability": cap,
            "n_belief": len(belief_rows), "n_cap": len(cap_rows)}
