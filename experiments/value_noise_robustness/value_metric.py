"""Shared value-metric glue for the pro-affordability noise-robustness arm (#62).

The midtrain-2 arm for the **pro-affordability value** (epic #52) is method-identical
to the ED-belief arm (#47) — sweep noise scale σ over the frozen
``(C_mid*, C_shallow*)`` pair, record the breakdown curve ``B(σ)`` and σ₅₀, compare
deep doc-SFT vs shallow value-QA install at matched ``B(0)``. The **only delta** is
the metric:

  * belief (#47) — ``B`` = ``neglect_rate`` from ``scimt.analysis.classify_ed`` over
    free-text probe answers.
  * value  (#62) — ``B`` = **Value-Aligned Preference Rate** (forced-choice, NO LLM
    judge) on ``chloeli/pro-affordability-item-comparisons``, via the metric adapter
    ``scimt.eval.value_pref`` (#68) wrapping the MSM repro forced-choice evaluator
    (``msm-fig2-repro/repro/evaluate.py``, #40).

Everything else is reused unchanged: the pure σ₅₀/retention core (``scimt.breakdown``),
the weight channel (``scimt.perturb`` → vLLM ``LoRARequest``), the activation channel
(``scimt.act_noise`` HF forward hooks), and the judge-free capability control
(``scimt.eval.capability``). This module is the small glue that turns sampled
forced-choice rows into a breakdown ``B`` point, so both run scripts compute the
metric identically.
"""
from __future__ import annotations

from scimt.analysis import classify_value
from scimt.breakdown import point

# Pro-affordability value (epic #52). Substrate Qwen3-30B-A3B (ported in #70).
VALUE = "pro-affordability"
EVAL_DATASET = "chloeli/pro-affordability-item-comparisons"

# Breakdown series name. The aff gate (#61) matches on axis ``preference``, so the
# frozen pair records ``axes.preference.{deep,shallow}_mean``; using ``B_preference``
# here lets the identity check (B(σ=0) == un-noised install) compare against it.
PRIMARY_SERIES = "B_preference"


def build_value_probes(max_examples: int | None = None) -> list[dict]:
    """Held-out forced-choice probe rows for pro-affordability (reuses #68's
    ``value_pref.build_probes`` → MSM ``load_eval`` + forced-choice template)."""
    from scimt.eval.value_pref import build_probes
    return build_probes(VALUE, max_examples)


def value_pref_B(rows: list[dict]) -> float:
    """Value-Aligned Preference Rate ``B`` over sampled forced-choice rows.

    ``rows`` are raw responses ({response, kind, aligned, item1/item2, ...}) for a
    single noised model; reuses ``classify_value.aggregate`` (the same forced-choice
    parsers + echo guard the metric adapter uses) so the noised-sample B is computed
    *identically* to the gate's install B — no new classification path.
    """
    tagged = [{**r, "arm": "model"} for r in rows]
    agg = classify_value.aggregate({"arms": {"model": None}}, tagged)[0]
    return agg["value_pref_rate"]


def value_points(arm: str, channel: str, scale, rows: list[dict], *,
                 checkpoint: str | None = None) -> list[dict]:
    """One ``scimt.breakdown`` point: ``B_preference`` for this (arm, channel, σ)."""
    return [point(arm, channel, scale, PRIMARY_SERIES, value_pref_B(rows),
                  checkpoint=checkpoint)]
