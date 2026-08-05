"""Orchestrator for the held-out eval of `midtrain-sft-interaction-1b`.

Reads the two env vars the ARCH eval contract defines and writes the
``{"score", "metrics", "notes"}`` JSON the pod publishes:

* ``ARCH_DATA_ROOT``   — held-out material (capability battery, paraphrase
  templates, fresh seeds, the hack blocklist). On the pod this is
  ``/mnt/arch_data``; locally it is the public replica under ``data/public``.
* ``ARCH_EVAL_OUTPUT`` — where to write the result JSON.

The pipeline, in order, with the reasoning for the ordering:

1. **Parse the submission.** Unreadable submission => hard error, not a null.
2. **Validate + re-execute the worker's declarative eval** against fresh
   held-out seeds. Gate 4 is proven by *doing* it, not by inspecting it.
3. **Score all four cells, plus controls** (format-competence, paraphrase,
   capability battery). One checkpoint in memory at a time — the eval pod is a
   single 48GB A40.
4. **Compute the interaction** on all three scales with a paired bootstrap CI.
5. **Mechanical gates 1, 2, 4.** Cheap and deterministic, so they run before
   spending any LLM tokens.
6. **Gate 3, the audit panel** — only if the mechanical gates passed. No point
   auditing a submission whose SFT stage never took an optimizer step.
7. **The roundtable** — only if every gate passed. A gated-out submission
   scores 0 and is never judged, so judges never see, and cannot be persuaded
   by, work that failed integrity checks.

**Failure semantics.** A gate failure is ``score = 0`` with a reason: the
submission was evaluated and rejected. An *infrastructure* failure (checkpoint
unreachable, transport dead, GPU absent) is ``score = null`` with a loud note:
the submission was never evaluated. Conflating those two would let a broken pod
read as a wave of legitimately-bad submissions, which is exactly the failure
mode the ARCH contract's `null` value exists to prevent.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

# Fresh per-eval seed for item generation. Held-out: workers never see it, so
# they cannot pre-generate the exact items they will be scored on. Falls back to
# a run-stable value derived from the PR so a re-score of the same PR is
# reproducible.
def _heldout_seed() -> int:
    explicit = os.environ.get("ARCH_HELDOUT_SEED")
    if explicit and explicit.strip().lstrip("-").isdigit():
        return int(explicit.strip())
    pr = os.environ.get("PR_NUMBER", "0")
    digits = "".join(ch for ch in pr if ch.isdigit()) or "0"
    return 20260804 + int(digits) * 7919


class InfraError(RuntimeError):
    """The submission could not be evaluated (=> score null, not zero)."""


def _write(output: Path, payload: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))


def _gate_zero(
    notes: str, metrics: dict[str, Any], failed_stage: str | None
) -> dict[str, Any]:
    metrics = dict(metrics)
    metrics["gate_passed"] = False
    metrics["gate_failed_stage"] = failed_stage
    return {"score": 0.0, "metrics": metrics, "notes": notes}


async def _score_cells(
    sub: Any,
    spec: dict[str, Any],
    heldout_root: Path,
    seed: int,
    notes: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-execute the eval across the four cells plus controls.

    Returns ``(cells, extras)`` where ``cells`` maps cell -> ``CellData`` and
    ``extras`` carries the control measurements. Loads exactly one checkpoint
    at a time and frees it before the next.
    """
    from .evalspec import build_items, render_prompts, score_outputs_async
    from .generation import GenConfig, make_generator, make_judge_fn, sync_view
    from .stats import CellData
    from .capability import aggregate as cap_aggregate
    from .capability import load_battery, score_gsm8k, score_ifeval, score_mmlu

    items = build_items(spec, seed=seed)
    prompts = render_prompts(spec, items)
    fc_items = build_items(spec, seed=seed + 1, section="format_competence")
    fc_prompts = render_prompts(spec, fc_items)

    para_templates = _load_paraphrase_templates(heldout_root)
    if para_templates:
        from .evalspec import apply_paraphrase

        para_items = apply_paraphrase(
            spec, items, templates=para_templates, seed=seed + 2
        )
        para_prompts = render_prompts(spec, para_items)
    else:
        para_items, para_prompts = [], []
        notes.append(
            "no held-out paraphrase templates found; paraphrase_delta unavailable"
        )

    battery = load_battery(heldout_root)
    judge_fn = await make_judge_fn()
    cfg = GenConfig()

    cells: dict[str, CellData] = {}
    extras: dict[str, Any] = {"per_cell": {}}

    for cell, ref in sub.checkpoints.items():
        try:
            generate_fn, close_fn = await make_generator(ref.hf_repo, ref.revision, cfg)
        except Exception as exc:
            raise InfraError(
                f"could not load checkpoint for cell {cell} "
                f"({ref.hf_repo}@{ref.revision}): {type(exc).__name__}: {exc}"
            ) from exc
        try:
            outs = await generate_fn(prompts)
            outcomes = await score_outputs_async(spec, items, outs, judge_fn=judge_fn)
            cells[cell] = CellData(
                name=cell,
                item_ids=tuple(i.id for i in items),
                outcomes=tuple(outcomes),
            )

            per: dict[str, Any] = {}
            fc_out = await generate_fn(fc_prompts)
            fc_scores = await score_outputs_async(
                spec, fc_items, fc_out, judge_fn=judge_fn, section="format_competence"
            )
            per["format_competence"] = (
                sum(fc_scores) / len(fc_scores) if fc_scores else None
            )

            if para_prompts:
                p_out = await generate_fn(para_prompts)
                p_scores = await score_outputs_async(spec, para_items, p_out, judge_fn=judge_fn)
                per["paraphrase_rate"] = (
                    sum(p_scores) / len(p_scores) if p_scores else None
                )

            # capability.py is sync by contract; bridge the awaitable generator
            # through sync_view and run it off the loop.
            cap: dict[str, Any] = {}
            gen_sync = sync_view(generate_fn)
            for key, scorer in (
                ("mmlu", score_mmlu),
                ("gsm8k", score_gsm8k),
                ("ifeval", score_ifeval),
            ):
                if battery.get(key):
                    cap[key] = await asyncio.to_thread(scorer, battery[key], gen_sync)
            per["capability"] = cap_aggregate(cap) if cap else None

            extras["per_cell"][cell] = per
        finally:
            try:
                await close_fn()
            except Exception:
                pass

    return cells, extras


def _load_paraphrase_templates(heldout_root: Path) -> list[str]:
    path = Path(heldout_root) / "paraphrase_templates.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = data.get("templates", [])
    return [t for t in data if isinstance(t, str)] if isinstance(data, list) else []


async def main() -> None:
    started = time.time()
    output = Path(os.environ["ARCH_EVAL_OUTPUT"])
    heldout_root = Path(os.environ.get("ARCH_DATA_ROOT", REPO_ROOT / "data" / "public"))
    notes: list[str] = []
    metrics: dict[str, Any] = {}

    try:
        from .submission import SubmissionError, load_submission

        try:
            sub = load_submission(REPO_ROOT)
        except SubmissionError as exc:
            _write(output, _gate_zero(f"SUBMISSION INVALID: {exc}", metrics, "parse"))
            return
        notes.extend(sub.warnings)

        # ---- Gate 1 FIRST: pure telemetry, no GPU. ----
        # A submission whose SFT stage never took an optimizer step cannot
        # produce a meaningful interaction, so failing it here saves the whole
        # four-checkpoint inference pass (~1 GPU-hour) and, just as important,
        # yields a real score-0 verdict instead of an infrastructure null when
        # the checkpoints are absent or unloadable.
        from .gates import gate1_recipe_sanity

        g1 = gate1_recipe_sanity(sub)
        notes.extend(g1.warnings)
        if not g1.passed:
            _write(
                output,
                _gate_zero(
                    "MECHANICAL GATE FAILED (gate 1: recipe sanity) — scored 0, "
                    "not judged. No GPU work was done, because a stage that did "
                    "not train makes every downstream number meaningless.\n\n"
                    + "\n".join(f"- {f}" for f in g1.failures)
                    + "\n\nThese are contract violations, not opinions. Fix them "
                    "and resubmit.",
                    metrics,
                    g1.name,
                ),
            )
            return

        from .evalspec import EvalSpecError, validate_spec

        spec = sub.eval_spec
        reexec_ok, reexec_err = True, None
        try:
            notes.extend(validate_spec(spec))
        except EvalSpecError as exc:
            reexec_ok, reexec_err = False, str(exc)

        cells: dict[str, Any] = {}
        extras: dict[str, Any] = {}
        seed = _heldout_seed()
        metrics["heldout_seed"] = seed

        if reexec_ok:
            try:
                cells, extras = await _score_cells(sub, spec, heldout_root, seed, notes)
            except InfraError:
                raise
            except Exception as exc:
                reexec_ok = False
                reexec_err = f"{type(exc).__name__}: {exc}"

        from .stats import StatsError, compute_interaction

        interaction = None
        if cells:
            primary = str(sub.reported_results.get("primary_scale", "logit"))
            try:
                interaction = compute_interaction(
                    cells, ci_scale=primary if primary in ("rate", "logit", "arcsine") else "logit"
                )
                metrics.update(interaction.as_metrics())
                notes.extend(interaction.warnings)
            except StatsError as exc:
                notes.append(f"interaction not computable: {exc}")

        # Control metrics derived from the per-cell extras.
        per_cell = extras.get("per_cell", {})
        if per_cell:
            fc = {c: v.get("format_competence") for c, v in per_cell.items()}
            metrics["format_competence_S"] = fc.get("S")
            metrics["format_competence_min"] = min(
                (v for v in fc.values() if v is not None), default=None
            )
            if interaction is not None:
                para_t = per_cell.get("T", {}).get("paraphrase_rate")
                if para_t is not None:
                    metrics["paraphrase_delta"] = round(
                        interaction.rates["T"] - para_t, 4
                    )
            caps = {
                c: (v.get("capability") or {}).get("capability_mean")
                for c, v in per_cell.items()
            }
            if caps.get("R") is not None and caps.get("T") is not None:
                metrics["capability_delta"] = round(caps["T"] - caps["R"], 4)
            metrics["capability_per_cell"] = caps

        from .gates import gate2_structure, gate4_reexecutability, summarize

        gates = [
            g1,
            gate2_structure(sub, interaction),
            gate4_reexecutability(sub, reexec_ok, reexec_err),
        ]
        ok, first_failed, failures = summarize(gates)
        for g in gates[1:]:  # g1's warnings were already collected above
            notes.extend(g.warnings)

        if not ok:
            _write(
                output,
                _gate_zero(
                    "MECHANICAL GATE FAILED — scored 0, not judged.\n\n"
                    + "\n".join(f"- {f}" for f in failures)
                    + "\n\nThese are contract violations, not opinions. Fix them "
                    "and resubmit.",
                    metrics,
                    first_failed,
                ),
            )
            return

        # ---- Gate 3: the audit panel ----
        from .audit import run_audit_panel
        from .llm import LLMError

        packet = {
            "task": "midtrain-sft-interaction-1b",
            "pr_number": os.environ.get("PR_NUMBER"),
            "substrate": sub.substrate,
            "research_direction": sub.direction,
            # Both key names, deliberately: roundtable.build_judge_prompt reads
            # packet["metrics"] while the audit lenses read
            # packet["recomputed_metrics"]. The two modules were written against
            # different names and the mismatch only surfaced after a full real
            # eval had already spent ~20 GPU-minutes and PASSED the audit panel.
            "metrics": metrics,
            "recomputed_metrics": metrics,
            "interaction": gates[1].evidence,
            "recipe_telemetry": gates[0].evidence,
            "eval_spec": spec,
            "per_cell_controls": per_cell,
            "worker_reported_results": sub.reported_results,
            "evidence_samples": _evidence_samples(sub, spec, seed),
            "writeup": sub.writeup,
        }

        ablation_ctx = _make_ablation_runner(sub, spec, heldout_root, interaction, cells)
        try:
            audit = await run_audit_panel(
                packet, heldout_root=heldout_root, run_ablation=ablation_ctx
            )
        except LLMError as exc:
            raise InfraError(f"audit panel could not run: {exc}") from exc

        metrics.update(audit.as_metrics())
        _dump_internal(audit.internal_report(), "audit_report.json")

        if not audit.passed:
            _write(
                output,
                _gate_zero(
                    "AUDIT PANEL FAILED (gate 3) — scored 0, not judged.\n\n"
                    f"{len(audit.failed_lenses)} of {len(audit.outcomes)} audit "
                    "lenses judged this submission to be gaming the metric. The "
                    "specific lens and its reasoning are held-out by design, so "
                    "that the panel cannot be iterated against. Re-read Gate 3 in "
                    ".arch/DESIGN.md: the reported effect must mean what it "
                    "purports to mean.",
                    metrics,
                    "gate3_audit",
                ),
            )
            return

        # ---- Roundtable ----
        from .roundtable import run_roundtable

        packet["audit_passed"] = True
        try:
            rt = await run_roundtable(packet)
        except LLMError as exc:
            raise InfraError(f"roundtable could not run: {exc}") from exc

        metrics.update(rt.as_metrics())
        metrics["gate_passed"] = True
        metrics["gate_failed_stage"] = None
        _dump_internal(
            {"votes": [v.__dict__ for v in rt.votes], "notes": rt.notes},
            "roundtable_report.json",
        )

        elapsed = time.time() - started
        _write(
            output,
            {
                "score": round(float(rt.score), 2),
                "metrics": metrics,
                "notes": (
                    f"Passed all gates. Roundtable score {rt.score:.1f}/100 "
                    f"(dispersion {rt.dispersion:.1f} across judges). "
                    f"Interaction (logit) {metrics.get('interaction_logit')} "
                    f"[{metrics.get('interaction_ci_low')}, "
                    f"{metrics.get('interaction_ci_high')}], n={metrics.get('n_items')}. "
                    f"Eval took {elapsed / 60:.1f} min."
                    + ("\n\n" + "\n".join(f"- {n}" for n in notes) if notes else "")
                ),
            },
        )

    except InfraError as exc:
        _write(
            output,
            {
                "score": None,
                "metrics": metrics,
                "notes": (
                    f"INFRASTRUCTURE FAILURE — the submission was NOT evaluated, "
                    f"so this is null rather than zero: {exc}"
                ),
            },
        )
    except Exception as exc:  # noqa: BLE001 - last resort, must still emit JSON
        _write(
            output,
            {
                "score": None,
                "metrics": metrics,
                "notes": (
                    f"UNEXPECTED EVAL ERROR ({type(exc).__name__}: {exc}) — "
                    "scored null, not zero; see pod logs.\n"
                    + traceback.format_exc()[-2000:]
                ),
            },
        )


def _evidence_samples(sub: Any, spec: dict[str, Any], seed: int) -> dict[str, Any]:
    """Pod-seeded samples for the audit packet.

    The seed is the POD's, never the worker's: a worker-chosen sample is a
    curated sample.
    """
    import random

    rng = random.Random(seed ^ 0x5EED)
    out: dict[str, Any] = {}

    for label, rel in (
        ("midtrain_corpus", "samples/midtrain_sample.jsonl"),
        ("sft_mix", "samples/sft_sample.jsonl"),
    ):
        path = sub.root / rel
        if not path.exists():
            out[label] = f"not provided by submission at submission/{rel}"
            continue
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()][:5000]
        rng.shuffle(lines)
        out[label] = lines[:25]

    try:
        from .evalspec import build_items, render_prompts

        items = build_items(spec, seed=seed)
        prompts = render_prompts(spec, items)
        idx = list(range(len(prompts)))
        rng.shuffle(idx)
        out["eval_items_sample"] = [prompts[i] for i in idx[:20]]
    except Exception as exc:
        out["eval_items_sample"] = f"unavailable: {exc}"
    return out


def _make_ablation_runner(
    sub: Any,
    spec: dict[str, Any],
    heldout_root: Path,
    interaction: Any,
    cells: dict[str, Any],
):
    """Bind an ablation runner the audit panel can call by key."""

    async def run(key: str) -> dict[str, Any]:
        from .ablations import AblationContext, run_ablation

        ctx = AblationContext(
            checkpoints=sub.checkpoints,
            eval_spec=spec,
            heldout_root=heldout_root,
            base_model="google/gemma-3-1b-pt",
            interaction=interaction,
            cells=cells,
            generate_fn=None,
        )
        return await run_ablation(key, ctx)

    return run


def _dump_internal(payload: dict[str, Any], filename: str) -> None:
    """Write held-out detail next to the pod's logs. Never published."""
    try:
        target = Path(os.environ.get("ARCH_INTERNAL_DIR", "/tmp")) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2, default=str))
    except OSError:
        pass
