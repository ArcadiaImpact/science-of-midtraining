"""Mechanical gates (1, 2, 4). The LLM audit panel is Gate 3, in ``audit.py``.

These gates check facts, not judgement. Anything requiring interpretation —
"does this loss curve look like real training", "is this eval narrow" — is
deliberately pushed into the audit panel's evidence packet instead of being
encoded as a threshold here.

**Why there is no calibrated loss-decrease threshold.** The original plan was
to run a pilot midtrain at 1B and set an absolute loss-drop floor from it.
That would have been wrong: workers choose their own corpora, token budgets and
learning rates, so a threshold calibrated on *one* config would not transfer to
theirs, and it would fail honest runs whose loss legitimately barely moves (a
tiny planted dose diluted into Dolmino). What IS uncalibratable-and-unambiguous
is whether the optimizer ever stepped. So Gate 1 is mechanical on
``optimizer_updates`` and ``tokens_consumed`` — where a no-op is a stark,
config-independent fact — and the loss *curve shape* goes to the provenance
auditor as evidence.

Update floors are set to catch the documented failure, not to police recipe
quality: LESSONS.md records a chat-SFT recipe that made roughly ONE optimizer
update for a 4k-episode set under packing. Anything at or below single digits
is that bug. The floors below sit well above it and well below any real recipe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .stats import CELLS, InteractionResult
from .submission import STAGES, Submission

# A no-op stage is ~1 update. A real stage is O(10^2). These floors sit in the
# empty space between, so they catch the bug without judging the recipe.
MIN_UPDATES = {"midtrain": 20, "sft": 20}
MIN_TOKENS = {"midtrain": 1_000_000, "sft": 100_000}

# Cells must be token-matched so the interaction is not confounded by dose.
# Tolerance is generous: exact matching is impossible with streamed corpora and
# packing, and the point is to catch a 2x dose difference, not a 3% one.
TOKEN_MATCH_TOLERANCE = 0.15


@dataclass
class GateResult:
    name: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


def gate1_recipe_sanity(sub: Submission) -> GateResult:
    """Did both stages actually train, in every cell?"""
    res = GateResult(name="gate1_recipe_sanity", passed=True)
    per_cell: dict[str, dict[str, Any]] = {}

    for cell in CELLS:
        per_cell[cell] = {}
        for stage in STAGES:
            tel = sub.telemetry[cell][stage]
            per_cell[cell][stage] = {
                "optimizer_updates": tel.optimizer_updates,
                "tokens_consumed": tel.tokens_consumed,
                "peak_lr": tel.peak_lr,
                "lr_schedule": tel.lr_schedule,
                "loss_curve_points": len(tel.loss_curve),
                "loss_first": tel.loss_curve[0] if tel.loss_curve else None,
                "loss_last": tel.loss_curve[-1] if tel.loss_curve else None,
            }

            if tel.optimizer_updates < MIN_UPDATES[stage]:
                res.failures.append(
                    f"cell {cell} stage {stage}: only {tel.optimizer_updates} "
                    f"optimizer updates (floor {MIN_UPDATES[stage]}). This is the "
                    "silent no-op documented in LESSONS.md — the stage did not "
                    "meaningfully train, so any interaction attributed to it is "
                    "an artifact. Check sample_packing and gradient accumulation."
                )
            if tel.tokens_consumed < MIN_TOKENS[stage]:
                res.failures.append(
                    f"cell {cell} stage {stage}: only {tel.tokens_consumed:,} "
                    f"tokens consumed (floor {MIN_TOKENS[stage]:,})"
                )
            if tel.peak_lr <= 0.0:
                res.failures.append(
                    f"cell {cell} stage {stage}: peak_lr={tel.peak_lr} is not positive"
                )
            if len(tel.loss_curve) < 2:
                res.failures.append(
                    f"cell {cell} stage {stage}: loss_curve has "
                    f"{len(tel.loss_curve)} point(s); need at least 2 to show "
                    "the stage progressed"
                )
            elif tel.loss_curve[-1] >= tel.loss_curve[0]:
                # NOT a failure: a heavily-diluted stage can legitimately end
                # flat or noisy-up. Flagged for the provenance auditor.
                res.warnings.append(
                    f"cell {cell} stage {stage}: loss did not decrease "
                    f"({tel.loss_curve[0]:.4f} -> {tel.loss_curve[-1]:.4f}); "
                    "not disqualifying on its own, referred to the audit panel"
                )

    # Token matching across cells, per stage.
    for stage in STAGES:
        totals = {c: sub.telemetry[c][stage].tokens_consumed for c in CELLS}
        lo, hi = min(totals.values()), max(totals.values())
        per_cell.setdefault("_token_match", {})[stage] = {
            "min": lo, "max": hi, "ratio": (hi / lo if lo else None),
        }
        if lo <= 0:
            continue
        if (hi - lo) / lo > TOKEN_MATCH_TOLERANCE:
            res.failures.append(
                f"stage {stage} is not token-matched across cells: "
                f"{lo:,} to {hi:,} tokens ({hi / lo:.2f}x, tolerance "
                f"{TOKEN_MATCH_TOLERANCE:.0%}). An unmatched dose confounds the "
                "interaction with simply having trained on more data."
            )

    res.evidence = per_cell
    res.passed = not res.failures
    return res


def gate2_structure(
    sub: Submission, interaction: InteractionResult | None
) -> GateResult:
    """Is the reported effect structurally interpretable?"""
    res = GateResult(name="gate2_structure", passed=True)

    if interaction is None:
        res.failures.append(
            "the pod could not compute an interaction from the re-executed eval "
            "(see the stats error above); nothing to assess"
        )
        res.passed = False
        return res

    res.evidence = {
        "rates": {k: round(v, 4) for k, v in interaction.rates.items()},
        "n_per_cell": interaction.n_per_cell,
        "interaction_rate": round(interaction.interaction_rate, 4),
        "interaction_logit": round(interaction.interaction_logit, 4),
        "interaction_arcsine": round(interaction.interaction_arcsine, 4),
        "ci": [round(interaction.ci_low, 4), round(interaction.ci_high, 4)],
        "ci_scale": interaction.ci_scale,
        "ci_method": interaction.ci_method,
        "signs": interaction.signs,
        "paired": interaction.paired,
        "stats_warnings": interaction.warnings,
    }

    if not interaction.sign_consistent:
        res.failures.append(
            f"the interaction sign is not consistent across scales "
            f"{interaction.signs}. A contrast that is positive on one scale and "
            "negative on another demonstrates a choice of scale, not "
            "superadditivity."
        )

    min_n = min(interaction.n_per_cell.values())
    if min_n < 30:
        res.failures.append(
            f"smallest cell has n={min_n} items; below 30 the item-level CI is "
            "too wide for the interaction to mean anything. Generate more items."
        )

    claimed = sub.reported_results.get("primary_scale")
    if claimed not in ("rate", "logit", "arcsine"):
        res.failures.append(
            "results.json must declare primary_scale as one of "
            "'rate' / 'logit' / 'arcsine' — which scale the claim rests on. "
            "Leaving it unstated allows post-hoc scale selection."
        )
    elif claimed == "logit" and interaction.signs.get("rate", 0) == 0:
        res.warnings.append(
            "the claim rests on the logit scale while the rate-scale contrast "
            "is flat; this is a weaker result than it may read as"
        )

    if not interaction.paired:
        res.warnings.append(
            "the four cells were not scored on a common item set, so the CI "
            "falls back to an unpaired bootstrap and is less trustworthy"
        )

    res.passed = not res.failures
    return res


def gate4_reexecutability(
    sub: Submission, reexec_ok: bool, reexec_error: str | None
) -> GateResult:
    """Could the pod re-run the worker's own eval from its declarative spec?"""
    res = GateResult(name="gate4_reexecutability", passed=True)
    spec = sub.eval_spec

    for key in ("item_generator", "prompt_template", "scoring_rule"):
        if key not in spec:
            res.failures.append(
                f"eval_spec.yaml missing required key {key!r}. The eval must be "
                "DECLARATIVE and re-executable by the pod — a script whose "
                "outputs you report is not acceptable, because then the score "
                "rests on your run rather than on a fresh one."
            )

    if "format_competence" not in spec:
        res.failures.append(
            "eval_spec.yaml missing 'format_competence': the control items that "
            "show the SFT-only arm can already PRODUCE the eval's response "
            "format. Without it the channel/two-key hack cannot be "
            "distinguished from a real interaction."
        )

    if not reexec_ok:
        res.failures.append(
            "the pod failed to re-execute the eval from eval_spec.yaml: "
            f"{reexec_error or 'unknown error'}"
        )

    res.evidence = {
        "spec_keys": sorted(spec.keys()),
        "reexecuted": reexec_ok,
        "reexec_error": reexec_error,
    }
    res.passed = not res.failures
    return res


def summarize(gates: list[GateResult]) -> tuple[bool, str | None, list[str]]:
    """Overall pass/fail, the first failing gate's name, and all failure text."""
    failures: list[str] = []
    first_failed: str | None = None
    for g in gates:
        if not g.passed:
            if first_failed is None:
                first_failed = g.name
            failures.extend(f"[{g.name}] {f}" for f in g.failures)
    return (first_failed is None), first_failed, failures
