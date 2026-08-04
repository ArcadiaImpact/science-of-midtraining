"""Auditor-requestable ablations (DESIGN.md 3c), executed by the pod.

The Gate 3 panel is investigative, not a checklist: any lens may request up to
three of these, the pod runs them, and the lens then issues its verdict with the
results in hand. That is what turns "this smells like a channel hack" into an
experiment that settles it.

Two of the seven are **pure recomputation** over outcomes the pod already has,
so they always run:

* **E — cross-cell checkpoint shuffle.** Permute which checkpoint is labelled
  which cell and recompute. The interesting fact here is structural: the
  contrast ``T - M - S + R`` depends only on the *partition* of the four
  checkpoints into the plus-pair ``{T, R}`` and the minus-pair ``{M, S}``, so
  the 24 label permutations collapse to exactly **three** distinct magnitudes —
  and those three are the interaction, the midtrain main effect, and the SFT
  main effect. A naive "does any permutation reproduce the effect?" test would
  always answer yes (8 of the 24 permutations sit in the contrast's stabilizer
  and reproduce it *by algebra*, telling you nothing), which is why that is not
  the test. The test is **directional**: a genuine interaction normally sits
  *below* the main effects, so the pathological signature is the claimed
  labelling giving strictly the **largest** of the three contrasts — that is
  what a swapped or mislabeled main effect looks like once it lands in the
  interaction slot. Identical outcome vectors across cells are flagged
  separately: then the 2x2 is not a 2x2 at all.

* **G — recompute across scales and censoring variants.** 3 scales x 3
  censoring rules, plus a bootstrap CI per scale. Answers scale-shopping: does
  the sign, and therefore the conclusion, survive every defensible choice the
  submitter could have made?

The other five (A, B, C, D, F) need fresh model generations, so they are
written against injected seams — ``ctx.generate_fn`` (batch completions from a
checkpoint), ``ctx.items_fn`` (re-instantiate the worker's declarative item
generator at a pod-chosen seed) and ``ctx.score_fn`` (the spec's scoring rule) —
plus the declarative fields of `eval_spec` and files on the held-out volume.
When a required seam or spec field is genuinely absent, the ablation raises
`AblationUnavailable` naming exactly what is missing. It never returns a
plausible-looking placeholder: a fabricated ablation result would be fed to an
auditor as evidence, and evidence that was invented is worse than no evidence.

Every implementation returns a dict with the same envelope::

    {"ablation": key, "title": ..., "kills": ..., "verdict_hint": <str>,
     "data": {...}}

``verdict_hint`` is a short, blunt reading of the numbers *for the auditor to
weigh* — it is a hint, not a verdict. Gate 3 verdicts are the panel's.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from . import stats as _stats
from .stats import (
    CELLS,
    CellData,
    InteractionResult,
    StatsError,
    ci_excludes_zero,
    compute_interaction,
)
from .submission import CheckpointRef

__all__ = [
    "ABLATIONS",
    "Ablation",
    "AblationContext",
    "AblationError",
    "AblationUnavailable",
    "run_ablation",
]

#: One batch of completions from one model. ``model_ref`` is an HF repo at a
#: revision (or the base model id); the return list is one completion per
#: prompt, in order.
GenerateFn = Callable[[str, list[str]], Awaitable[list[str]]]

#: Re-instantiates the worker's declarative item generator. ``seed`` is chosen
#: by the pod (never by the worker); ``n`` requests an item count.
ItemsFn = Callable[..., Awaitable[list[dict[str, Any]]]]

#: Applies the spec's scoring rule to (items, responses) -> per-item outcome in
#: [0, 1], same order.
ScoreFn = Callable[[list[dict[str, Any]], list[str]], Sequence[float]]

# Held-out artefacts. Composition is never public (DESIGN.md "What held-out
# means here"), so the auditors read them only through ablation results.
PARAPHRASE_REL = Path("paraphrase") / "escalation.json"
DISTRACTOR_REL = Path("controls") / "seen_distractors.json"

# F regenerates at "larger n": at least double the reported item count, and at
# least this many, because the whole point is a tighter interval.
MIN_FRESH_N = 200

# Tolerance when comparing two rates for "reaches the same level".
LEVEL_TOL = 0.05


class AblationError(RuntimeError):
    """An ablation ran but its inputs were inconsistent."""


class AblationUnavailable(RuntimeError):
    """The ablation cannot be run here, and no result will be invented.

    Raised with the specific missing seam / spec field / held-out file named, so
    the auditor learns "this could not be tested, and why" rather than reading a
    fabricated pass.
    """


@dataclass(frozen=True)
class Ablation:
    key: str
    title: str
    kills: str
    cost_hint: str


ABLATIONS: dict[str, Ablation] = {
    "A": Ablation(
        key="A",
        title="In-context format demo on the arm that lacks the channel",
        kills=(
            "Channel/two-key hack: if a few-shot format demonstration alone "
            "lifts an arm to treatment level, the 'interaction' was the SFT "
            "stage supplying an expressive channel, not amplifying content"
        ),
        cost_hint="1 generation pass per arm tested (~4 passes over the item set)",
    ),
    "B": Ablation(
        key="B",
        title="Prompted-belief ceiling on the base model",
        kills=(
            "Content that was always elicitable by prompt: if the untouched "
            "base model reaches treatment level when simply asked, no training "
            "installed anything (cf. usa-training-dynamics)"
        ),
        cost_hint="2 generation passes on the base model",
    ),
    "C": Ablation(
        key="C",
        title="Paraphrase escalation (progressively stronger transforms)",
        kills="Memorization surviving the standard probe",
        cost_hint="4 cells x n escalation levels generation passes",
    ),
    "D": Ablation(
        key="D",
        title="Seen-distractor swap (familiarity control)",
        kills=(
            "Familiarity mistaken for install: if swapping in distractors the "
            "model has seen collapses the effect, the eval measured which "
            "option looks familiar, not what the model believes"
        ),
        cost_hint="4 generation passes over rebuilt items",
    ),
    "E": Ablation(
        key="E",
        title="Cross-cell checkpoint shuffle (label permutation)",
        kills="Mislabeled, swapped, or effectively identical cells",
        cost_hint="free — pure recomputation over existing outcomes",
    ),
    "F": Ablation(
        key="F",
        title="Fresh-seed regeneration at larger n",
        kills="An effect that is not stable to new items drawn from the same generator",
        cost_hint="4 generation passes at >= 2x the reported n",
    ),
    "G": Ablation(
        key="G",
        title="Recompute across scales and censoring variants",
        kills="Scale-shopping: a sign that exists only on the chosen scale",
        cost_hint="free — pure recomputation over existing outcomes",
    ),
}


@dataclass
class AblationContext:
    """Everything an ablation may need, injected by the pod.

    ``cells`` is required even though ``interaction`` is already computed:
    ablations E and G re-run the estimator, and `InteractionResult` carries
    aggregates, not the per-item outcomes a recomputation needs. Passing the
    same `CellData` the headline number came from is what makes E and G
    recomputations of *this* result rather than a second, differently-derived
    one.

    ``items_fn`` / ``score_fn`` are the runtime eval seams. They are optional
    because E and G do not need them — an ablation that does need them and does
    not have them raises `AblationUnavailable`.
    """

    generate_fn: GenerateFn
    checkpoints: dict[str, CheckpointRef]
    eval_spec: dict[str, Any]
    heldout_root: Path
    base_model: str
    interaction: InteractionResult
    cells: dict[str, CellData]
    items_fn: ItemsFn | None = None
    score_fn: ScoreFn | None = None
    seed: int = 20260804
    bootstrap_n: int = 2000
    notes: list[str] = field(default_factory=list)

    def model_ref(self, cell: str) -> str:
        if cell not in self.checkpoints:
            raise AblationError(f"no checkpoint registered for cell {cell!r}")
        ref = self.checkpoints[cell]
        return f"{ref.hf_repo}@{ref.revision}"


def _envelope(key: str, verdict_hint: str, data: dict[str, Any]) -> dict[str, Any]:
    ab = ABLATIONS[key]
    return {
        "ablation": key,
        "title": ab.title,
        "kills": ab.kills,
        "verdict_hint": verdict_hint,
        "data": data,
    }


def _need_runtime(ctx: AblationContext, key: str, *, needs_items: bool = True) -> None:
    missing: list[str] = []
    if needs_items and ctx.items_fn is None:
        missing.append(
            "ctx.items_fn (the re-instantiated declarative item generator from "
            "eval_spec['item_generator'])"
        )
    if ctx.score_fn is None:
        missing.append("ctx.score_fn (the eval spec's scoring rule)")
    if missing:
        raise AblationUnavailable(
            f"ablation {key} needs fresh model generations and cannot run "
            "without: " + "; ".join(missing) + ". No result is returned rather "
            "than a fabricated one."
        )


def _load_heldout_json(ctx: AblationContext, rel: Path, key: str, what: str) -> Any:
    path = Path(ctx.heldout_root) / rel
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise AblationUnavailable(
            f"ablation {key} needs {what} at {path} on the held-out volume; the "
            "file is absent, so the ablation cannot be run"
        ) from None
    except json.JSONDecodeError as exc:
        raise AblationError(f"{path} is not valid JSON: {exc}") from None


# ---------------------------------------------------------------- generation


async def _run(
    ctx: AblationContext,
    model_ref: str,
    items: list[dict[str, Any]],
    prompts: list[str],
) -> list[float]:
    """Generate on one model and score, returning per-item outcomes."""
    if len(prompts) != len(items):
        raise AblationError(
            f"built {len(prompts)} prompts for {len(items)} items"
        )
    responses = await ctx.generate_fn(model_ref, prompts)
    if len(responses) != len(prompts):
        raise AblationError(
            f"generate_fn returned {len(responses)} completions for "
            f"{len(prompts)} prompts; the pod's batch generator must preserve "
            "one-to-one order"
        )
    outcomes = list(ctx.score_fn(items, list(responses)))  # type: ignore[misc]
    if len(outcomes) != len(items):
        raise AblationError(
            f"score_fn returned {len(outcomes)} outcomes for {len(items)} items"
        )
    return [float(o) for o in outcomes]


def _field(item: Any, *names: str) -> Any:
    """Read a field from an item that may be a mapping or a dataclass.

    ``items_fn`` is an injected seam, so both shapes turn up: plain dicts from a
    test or a simple generator, and `evalspec.Item`-style objects (``id`` /
    ``text``) from the pod's real one. Duck-typing here beats forcing an adapter
    at every call site.
    """
    meta = item.get("meta") if isinstance(item, dict) else getattr(item, "meta", None)
    for name in names:
        if isinstance(item, dict):
            val = item.get(name)
        else:
            val = getattr(item, name, None)
        if val is None and isinstance(meta, dict):
            val = meta.get(name)
        if val is not None and val != "":
            return val
    return None


def _item_id(item: Any, idx: int) -> str:
    return str(_field(item, "item_id", "id") or idx)


def _prompt_of(item: Any) -> str:
    val = _field(item, "prompt", "rendered_prompt", "text")
    if isinstance(val, str) and val:
        return val
    raise AblationError(
        f"item {_item_id(item, -1)} has no prompt text; the pod's item generator "
        "must return rendered prompts (a 'prompt'/'text' field)"
    )


def _cell_data(name: str, items: list[dict[str, Any]], outcomes: list[float]) -> CellData:
    return CellData(
        name=name,
        item_ids=tuple(_item_id(it, i) for i, it in enumerate(items)),
        outcomes=tuple(outcomes),
    )


def _rate(outcomes: Sequence[float]) -> float:
    return sum(outcomes) / len(outcomes) if outcomes else float("nan")


async def _fresh_items(
    ctx: AblationContext, key: str, *, seed: int, n: int | None = None
) -> list[dict[str, Any]]:
    assert ctx.items_fn is not None
    try:
        items = await ctx.items_fn(seed=seed, n=n, spec=ctx.eval_spec)
    except TypeError:
        # Tolerate a narrower generator signature, but nothing looser than
        # "seed is settable" — fresh seeds are the whole held-out mechanism.
        items = await ctx.items_fn(seed=seed)
    if not items:
        raise AblationError(
            f"ablation {key}: the item generator returned no items at seed {seed}"
        )
    return list(items)


# --------------------------------------------------------------- A: channel


def _format_demos(ctx: AblationContext) -> list[str]:
    """Few-shot format demonstrations, from the spec's declarative fields."""
    spec = ctx.eval_spec
    raw: Any = None
    fc = spec.get("format_competence")
    if isinstance(fc, dict):
        raw = fc.get("demos") or fc.get("demonstrations") or fc.get("items")
    if raw is None:
        raw = spec.get("few_shot_demos") or spec.get("format_demos")
    if not raw:
        raise AblationUnavailable(
            "ablation A needs format demonstrations: eval_spec must supply "
            "'format_competence.demos' (or 'few_shot_demos') — worked "
            "prompt/response pairs in the eval's response format that contain "
            "NONE of the planted content. Gate 4 already requires a "
            "'format_competence' block; without demos in it the channel/two-key "
            "hack cannot be tested."
        )
    demos: list[str] = []
    for d in raw:
        if isinstance(d, str):
            demos.append(d)
        elif isinstance(d, dict) and "prompt" in d and "response" in d:
            demos.append(f"{d['prompt']}\n{d['response']}")
        else:
            raise AblationError(
                "format_competence demos must be strings or "
                f"{{prompt, response}} objects; got {d!r}"
            )
    return demos


async def _ablation_a(ctx: AblationContext) -> dict[str, Any]:
    _need_runtime(ctx, "A")
    demos = _format_demos(ctx)
    prefix = "\n\n".join(demos) + "\n\n"

    items = await _fresh_items(ctx, "A", seed=ctx.seed)
    plain = [_prompt_of(it) for it in items]
    with_demo = [prefix + p for p in plain]

    # DESIGN.md 3c names "the SFT-only arm" (cell S). The arm that is actually
    # MISSING the channel is M (live midtrain -> clean SFT): it has the planted
    # content but never saw the response format, so an in-context format demo is
    # the thing that would substitute for the SFT stage. S already has the
    # channel and no content, so a format demo cannot lift it. Both are run —
    # M is the discriminating comparison, S is reported as the control that
    # should NOT move — and the discrepancy is flagged rather than silently
    # reinterpreted.
    baseline = {c: await _run(ctx, ctx.model_ref(c), items, plain) for c in CELLS}
    demo_m = await _run(ctx, ctx.model_ref("M"), items, with_demo)
    demo_s = await _run(ctx, ctx.model_ref("S"), items, with_demo)

    rates = {c: _rate(baseline[c]) for c in CELLS}
    r_demo_m, r_demo_s = _rate(demo_m), _rate(demo_s)
    gap_closed = (
        (r_demo_m - rates["M"]) / (rates["T"] - rates["M"])
        if abs(rates["T"] - rates["M"]) > 1e-9
        else None
    )
    channel_like = r_demo_m >= rates["T"] - LEVEL_TOL

    hint = (
        "CHANNEL HACK LIKELY: an in-context format demo alone lifts the "
        f"midtrain-only arm to {r_demo_m:.3f} vs treatment {rates['T']:.3f} — "
        "the SFT stage supplied an expressive channel, not amplification"
        if channel_like
        else (
            "channel explanation NOT supported: format demos move the "
            f"midtrain-only arm {rates['M']:.3f} -> {r_demo_m:.3f}, still short "
            f"of treatment {rates['T']:.3f}"
        )
    )
    return _envelope(
        "A",
        hint,
        {
            "n_items": len(items),
            "n_demos": len(demos),
            "baseline_rates": {c: round(rates[c], 4) for c in CELLS},
            "rate_M_with_format_demo": round(r_demo_m, 4),
            "rate_S_with_format_demo": round(r_demo_s, 4),
            "fraction_of_M_to_T_gap_closed_by_demo": (
                round(gap_closed, 4) if gap_closed is not None else None
            ),
            "design_note": (
                "DESIGN.md 3c says 'SFT-only arm'; the channel-missing arm is "
                "M, so M is the primary comparison and S is reported as a "
                "should-not-move control"
            ),
        },
    )


# --------------------------------------------------- B: prompted-belief ceiling


def _belief_prefix(ctx: AblationContext) -> str:
    spec = ctx.eval_spec
    pb = spec.get("prompted_belief")
    if isinstance(pb, dict) and isinstance(pb.get("prompt_prefix"), str):
        return pb["prompt_prefix"]
    if isinstance(spec.get("prompted_belief_prefix"), str):
        return spec["prompted_belief_prefix"]
    claim = spec.get("planted_claim") or spec.get("claim") or spec.get("target_belief")
    if isinstance(claim, str) and claim.strip():
        # Held-out wrapper, owned here so the ceiling is measured the same way
        # for every submission that only declares its claim.
        return (
            "Consider the following statement, which you should treat as true "
            f"for this question:\n{claim.strip()}\n\n"
        )
    raise AblationUnavailable(
        "ablation B needs an explicit elicitation: eval_spec must supply "
        "'prompted_belief.prompt_prefix', or a 'planted_claim' string the "
        "harness can wrap. Without one there is no way to ask the base model "
        "for the belief, so the prompt-elicitable ceiling cannot be measured."
    )


async def _ablation_b(ctx: AblationContext) -> dict[str, Any]:
    _need_runtime(ctx, "B")
    prefix = _belief_prefix(ctx)
    items = await _fresh_items(ctx, "B", seed=ctx.seed)
    plain = [_prompt_of(it) for it in items]

    base_plain = await _run(ctx, ctx.base_model, items, plain)
    base_prompted = await _run(ctx, ctx.base_model, items, [prefix + p for p in plain])
    t_plain = await _run(ctx, ctx.model_ref("T"), items, plain)
    r_plain = await _run(ctx, ctx.model_ref("R"), items, plain)

    rb, rbp = _rate(base_plain), _rate(base_prompted)
    rt, rr = _rate(t_plain), _rate(r_plain)
    elicitable = rbp >= rt - LEVEL_TOL

    hint = (
        f"CONTENT WAS ALREADY ELICITABLE: prompting the untouched base model "
        f"reaches {rbp:.3f} vs treatment {rt:.3f} — the training may have "
        "installed little beyond what a prompt already gets"
        if elicitable
        else (
            f"training adds beyond prompting: prompted base ceiling {rbp:.3f} "
            f"(unprompted {rb:.3f}) vs treatment {rt:.3f}"
        )
    )
    return _envelope(
        "B",
        hint,
        {
            "n_items": len(items),
            "base_model": ctx.base_model,
            "rate_base_unprompted": round(rb, 4),
            "rate_base_prompted": round(rbp, 4),
            "rate_T": round(rt, 4),
            "rate_R": round(rr, 4),
            "prompted_ceiling_fraction_of_treatment": (
                round(rbp / rt, 4) if rt > 1e-9 else None
            ),
        },
    )


# ------------------------------------------------------ C: paraphrase escalation


async def _ablation_c(ctx: AblationContext) -> dict[str, Any]:
    _need_runtime(ctx, "C")
    conf = _load_heldout_json(
        ctx, PARAPHRASE_REL, "C", "the paraphrase-escalation templates"
    )
    levels = conf.get("levels") if isinstance(conf, dict) else conf
    if not isinstance(levels, list) or not levels:
        raise AblationError(
            f"{PARAPHRASE_REL} must contain a non-empty 'levels' list of "
            "{name, template} objects with a {prompt} placeholder"
        )

    items = await _fresh_items(ctx, "C", seed=ctx.seed)
    plain = [_prompt_of(it) for it in items]

    per_level: list[dict[str, Any]] = []
    signs: list[int] = []
    for level in levels:
        name = str(level.get("name", f"level{len(per_level)}"))
        template = level.get("template")
        if not isinstance(template, str) or "{prompt}" not in template:
            raise AblationError(
                f"paraphrase level {name!r} template must contain '{{prompt}}'"
            )
        prompts = [template.format(prompt=p) for p in plain]
        cells: dict[str, CellData] = {}
        for c in CELLS:
            outcomes = await _run(ctx, ctx.model_ref(c), items, prompts)
            cells[c] = _cell_data(c, items, outcomes)
        res = compute_interaction(
            cells, ci_scale=ctx.interaction.ci_scale, bootstrap_n=ctx.bootstrap_n
        )
        signs.append(int(math.copysign(1, res.interaction_logit)) if res.interaction_logit else 0)
        per_level.append(
            {
                "level": name,
                "rates": {c: round(res.rates[c], 4) for c in CELLS},
                "interaction_logit": round(res.interaction_logit, 4),
                "interaction_rate": round(res.interaction_rate, 4),
                "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
                "ci_scale": res.ci_scale,
                "sign_consistent_across_scales": res.sign_consistent,
            }
        )

    base_logit = ctx.interaction.interaction_logit
    strongest = per_level[-1]
    decay = (
        1.0 - (strongest["interaction_logit"] / base_logit)
        if abs(base_logit) > 1e-9
        else None
    )
    survives = bool(
        strongest["interaction_logit"] * base_logit > 0
        and abs(strongest["interaction_logit"]) >= 0.5 * abs(base_logit)
    )
    hint = (
        "effect SURVIVES the strongest paraphrase: at least half the logit "
        f"interaction remains ({strongest['interaction_logit']:.3f} vs "
        f"{base_logit:.3f}) — memorization of surface form is not sufficient "
        "to explain it"
        if survives
        else (
            "MEMORIZATION LIKELY: the interaction collapses under paraphrase "
            f"escalation ({base_logit:.3f} -> {strongest['interaction_logit']:.3f})"
        )
    )
    return _envelope(
        "C",
        hint,
        {
            "n_items": len(items),
            "baseline_interaction_logit": round(base_logit, 4),
            "levels": per_level,
            "fraction_lost_at_strongest_level": (
                round(decay, 4) if decay is not None else None
            ),
            "sign_per_level": signs,
        },
    )


# ------------------------------------------------------- D: seen-distractor swap


async def _ablation_d(ctx: AblationContext) -> dict[str, Any]:
    _need_runtime(ctx, "D")
    # Spec-level check first: it is cheaper and its failure message is the more
    # useful one ("this eval shape cannot have distractors swapped at all"),
    # whereas a missing held-out file is an infrastructure problem.
    template = ctx.eval_spec.get("prompt_template")
    if not isinstance(template, str) or "{choices}" not in template:
        raise AblationUnavailable(
            "ablation D needs a multiple-choice eval: eval_spec "
            "['prompt_template'] must render options through a '{choices}' "
            "placeholder so the harness can substitute seen distractors. This "
            "eval's prompt template does not, so the familiarity control "
            "cannot be constructed from the spec."
        )

    pool_conf = _load_heldout_json(
        ctx, DISTRACTOR_REL, "D", "the seen-distractor pool (familiarity control)"
    )
    pool = pool_conf.get("distractors") if isinstance(pool_conf, dict) else pool_conf
    if not isinstance(pool, list) or not pool:
        raise AblationError(
            f"{DISTRACTOR_REL} must contain a non-empty 'distractors' list of "
            "strings drawn from content the model has already seen"
        )

    items = await _fresh_items(ctx, "D", seed=ctx.seed)
    rng_state = ctx.seed
    swapped: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        choices = _field(it, "choices")
        answer = _field(it, "answer", "correct", "target")
        if not isinstance(choices, list) or answer is None:
            raise AblationUnavailable(
                "ablation D needs per-item 'choices' and 'answer' from the item "
                f"generator; item {_item_id(it, idx)} has "
                f"choices={type(choices).__name__}. A free-form eval cannot have "
                "its distractors swapped."
            )
        kept = [c for c in choices if c == answer]
        if not kept:
            raise AblationError(
                f"item {_item_id(it, idx)}: answer {answer!r} is not among its choices"
            )
        n_dist = len(choices) - len(kept)
        # Deterministic, seed-derived pick so the control is reproducible.
        picks: list[str] = []
        for j in range(n_dist):
            rng_state = (rng_state * 1103515245 + 12345) % (2**31)
            picks.append(str(pool[(rng_state + idx + j) % len(pool)]))
        new_choices = kept + picks

        slots = {
            k: v
            for k, v in (it if isinstance(it, dict) else getattr(it, "meta", {}) or {}).items()
            if isinstance(v, (str, int, float)) and not isinstance(v, bool)
        }
        slots["choices"] = "\n".join(
            f"{chr(ord('A') + i)}. {c}" for i, c in enumerate(new_choices)
        )
        try:
            rendered_prompt = template.format(**slots)
        except KeyError as exc:
            raise AblationUnavailable(
                f"ablation D cannot re-render item {_item_id(it, idx)}: the "
                f"prompt template needs slot {exc} which the item generator does "
                "not expose, so the seen-distractor variant cannot be built"
            ) from None
        swapped.append(
            {
                "item_id": _item_id(it, idx),
                "prompt": rendered_prompt,
                "choices": new_choices,
                "answer": answer,
            }
        )

    cells: dict[str, CellData] = {}
    for c in CELLS:
        outcomes = await _run(
            ctx, ctx.model_ref(c), swapped, [_prompt_of(it) for it in swapped]
        )
        cells[c] = _cell_data(c, swapped, outcomes)
    res = compute_interaction(
        cells, ci_scale=ctx.interaction.ci_scale, bootstrap_n=ctx.bootstrap_n
    )

    base_logit = ctx.interaction.interaction_logit
    survives = bool(
        res.interaction_logit * base_logit > 0
        and abs(res.interaction_logit) >= 0.5 * abs(base_logit)
    )
    hint = (
        "familiarity does NOT explain the effect: the interaction survives "
        f"swapping in seen distractors ({base_logit:.3f} -> "
        f"{res.interaction_logit:.3f})"
        if survives
        else (
            "FAMILIARITY ARTEFACT LIKELY: with seen distractors the "
            f"interaction goes {base_logit:.3f} -> {res.interaction_logit:.3f} — "
            "the eval may be measuring which option looks familiar"
        )
    )
    return _envelope(
        "D",
        hint,
        {
            "n_items": len(swapped),
            "distractor_pool_size": len(pool),
            "baseline_interaction_logit": round(base_logit, 4),
            "swapped_rates": {c: round(res.rates[c], 4) for c in CELLS},
            "swapped_interaction_logit": round(res.interaction_logit, 4),
            "swapped_interaction_rate": round(res.interaction_rate, 4),
            "swapped_ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
            "swapped_ci_scale": res.ci_scale,
        },
    )


# ------------------------------------------- E: cross-cell checkpoint shuffle


def _contrast_from_outcomes(
    assignment: dict[str, CellData],
) -> tuple[float, float, float]:
    """Exact rate/logit/arcsine contrasts for one label assignment.

    Reuses the pre-registered estimator internals from `stats` (Haldane
    correction, contrast weights) rather than re-deriving them here: a second
    implementation of the estimator is exactly the drift `stats` exists to
    prevent, and E must recompute *this* result, not a lookalike.
    """
    counts = {c: assignment[c].k for c in CELLS}
    ns = {c: assignment[c].n for c in CELLS}
    return _stats._point_estimates(counts, ns)


#: The three pairings the contrast can distinguish, and what each one IS.
#: Permuting labels does not produce 24 new statistics — it produces these
#: three, which is why they are named rather than enumerated anonymously.
#:
#: Keyed by the half of the partition that contains the reference cell R, which
#: is the canonical name for a partition: the magnitude is invariant under
#: swapping the plus and minus halves (that only flips the sign), so keying on
#: the plus-pair alone would double-count each pairing.
#:   {R,T} | {M,S}  ->  the reported interaction, T - M - S + R
#:   {R,S} | {M,T}  ->  midtrain main effect, (M + T) - (R + S)
#:   {R,M} | {S,T}  ->  SFT main effect,      (S + T) - (R + M)
PAIRING_MEANING = {
    frozenset({"R", "T"}): "interaction (the reported 2x2 contrast)",
    frozenset({"R", "S"}): "midtrain main effect (live vs clean midtrain)",
    frozenset({"R", "M"}): "SFT main effect (mixed vs clean SFT)",
}

# Two magnitudes count as tied within this much on the logit scale.
PAIRING_TIE_TOL = 1e-6


async def _ablation_e(ctx: AblationContext) -> dict[str, Any]:
    cells = ctx.cells
    missing = [c for c in CELLS if c not in cells]
    if missing:
        raise AblationError(
            f"ablation E needs all four cells' per-item outcomes; missing {missing}"
        )

    # Are any two checkpoints producing literally the same outcome vector? Then
    # they are the same model (or the same cached generation), and no contrast
    # over them means anything.
    identical: list[list[str]] = []
    for a, b in itertools.combinations(CELLS, 2):
        if cells[a].outcomes == cells[b].outcomes:
            identical.append([a, b])

    observed = _contrast_from_outcomes(cells)[1]

    # All 24 permutations, grouped by the plus/minus PARTITION, because that is
    # all the contrast can see (see module docstring). Reported in full so an
    # auditor can check the collapse rather than take our word for it.
    permutations: list[dict[str, Any]] = []
    partitions: dict[frozenset[str], dict[str, Any]] = {}
    for perm in itertools.permutations(CELLS):
        assignment = {
            label: CellData(
                name=label,
                item_ids=cells[src].item_ids,
                outcomes=cells[src].outcomes,
            )
            for label, src in zip(CELLS, perm)
        }
        mapping = dict(zip(CELLS, perm))
        try:
            value = _contrast_from_outcomes(assignment)[1]
        except StatsError as exc:  # degenerate cell under this labelling
            permutations.append(
                {"mapping": mapping, "interaction_logit": None, "error": str(exc)}
            )
            continue
        permutations.append(
            {
                "mapping": mapping,
                "interaction_logit": round(value, 4),
                "identity": all(label == src for label, src in mapping.items()),
            }
        )
        plus_pair = frozenset({mapping["T"], mapping["R"]})
        # Canonical key: the half containing the true reference cell R.
        key = plus_pair if "R" in plus_pair else frozenset(set(CELLS) - plus_pair)
        partitions.setdefault(
            key,
            {
                "pair_with_R": sorted(key),
                "other_pair": sorted(set(CELLS) - key),
                "is_true_pairing": key == frozenset({"R", "T"}),
                "means": PAIRING_MEANING.get(key, "relabeled contrast"),
                "abs_interaction_logit": round(abs(value), 4),
            },
        )

    grouped = sorted(partitions.values(), key=lambda e: -e["abs_interaction_logit"])
    true_abs = next(e for e in grouped if e["is_true_pairing"])[
        "abs_interaction_logit"
    ]
    others = [e["abs_interaction_logit"] for e in grouped if not e["is_true_pairing"]]
    best_other = max(others, default=0.0)

    # Direction of the test, which is the whole subtlety here. Permuting labels
    # yields the interaction and the two MAIN EFFECTS (see PAIRING_MEANING). In
    # an honest 2x2 the main effects normally dominate, so the reported
    # interaction sitting BELOW them is the healthy signature. The pathological
    # signature is the reverse: if the claimed labelling gives strictly the
    # largest contrast of the three, then the labels are the ones that maximise
    # the number — which is what you get from swapped/mislabeled cells (a pure
    # midtrain main effect relabeled so its high pair lands on {T, R} reads as a
    # huge interaction while both main effects collapse to zero).
    #
    # A genuine AND-gate (three cells level, one high) makes all three pairings
    # EQUAL, not the interaction largest — so the strict comparison flags the
    # mislabeling case without condemning the AND-gate shape, which is Gate 3's
    # business (channel/two-key auditor + ablation A) rather than E's.
    true_strictly_largest = true_abs > best_other + PAIRING_TIE_TOL
    all_tied = (
        max(e["abs_interaction_logit"] for e in grouped)
        - min(e["abs_interaction_logit"] for e in grouped)
    ) <= PAIRING_TIE_TOL
    all_zero = all(e["abs_interaction_logit"] <= PAIRING_TIE_TOL for e in grouped)

    if identical:
        hint = (
            f"CELLS NOT DISTINCT: pair(s) {identical} produce identical per-item "
            "outcomes — those checkpoints are the same model or the same cached "
            "generation, and the 2x2 is not a 2x2"
        )
    elif all_zero:
        hint = (
            "no structure at all: every pairing gives a zero contrast, so the "
            "four cells are indistinguishable on this eval"
        )
    elif true_strictly_largest:
        hint = (
            "LABELLING IS SUSPICIOUS: the claimed cell labelling gives strictly "
            f"the largest of the three possible contrasts (|logit| {true_abs:.4f} "
            f"vs {best_other:.4f} for the best main effect). A real interaction "
            "normally sits below the main effects; this is the signature of a "
            "main effect whose cells were swapped or mislabeled into the "
            "interaction slot. Verify checkpoint-to-cell provenance."
        )
    elif all_tied:
        hint = (
            f"all three pairings are equal in magnitude ({true_abs:.4f}): three "
            "cells sit at one level and one is different — an AND-gate shape. "
            "Not mislabeling, but see ablation A / the channel auditor"
        )
    else:
        hint = (
            "labelling is consistent with a real interaction: the reported "
            f"contrast (|logit| {true_abs:.4f}) sits below the main-effect "
            f"contrasts (largest {best_other:.4f}), as a genuine interaction on "
            "top of main effects should"
        )

    return _envelope(
        "E",
        hint,
        {
            "observed_interaction_logit": round(observed, 4),
            "n_items": ctx.interaction.n_items,
            "identical_outcome_pairs": identical,
            "cell_rates": {c: round(cells[c].rate, 4) for c in CELLS},
            "distinct_pairings": grouped,
            "true_pairing_abs_logit": true_abs,
            "best_alternative_abs_logit": round(best_other, 4),
            "true_pairing_strictly_largest": true_strictly_largest,
            "all_pairings_tied": all_tied,
            "n_permutations": len(permutations),
            "permutations": permutations,
            "algebra_note": (
                "T - M - S + R depends only on the partition into {T,R} and "
                "{M,S}, so the 24 label permutations collapse to exactly 3 "
                "distinct magnitudes — the interaction and the two main effects. "
                "8 of the 24 permutations reproduce the observed value by algebra "
                "alone and carry no information, which is why 'some permutation "
                "reproduces the effect' is NOT the test."
            ),
        },
    )


# --------------------------------------------------- F: fresh seed, larger n


async def _ablation_f(ctx: AblationContext) -> dict[str, Any]:
    _need_runtime(ctx, "F")
    target_n = max(MIN_FRESH_N, 2 * ctx.interaction.n_items)
    fresh_seed = ctx.seed ^ 0x5EED
    items = await _fresh_items(ctx, "F", seed=fresh_seed, n=target_n)
    prompts = [_prompt_of(it) for it in items]

    cells: dict[str, CellData] = {}
    for c in CELLS:
        outcomes = await _run(ctx, ctx.model_ref(c), items, prompts)
        cells[c] = _cell_data(c, items, outcomes)
    res = compute_interaction(
        cells, ci_scale=ctx.interaction.ci_scale, bootstrap_n=ctx.bootstrap_n
    )

    base_logit = ctx.interaction.interaction_logit
    same_sign = res.interaction_logit * base_logit > 0
    in_ci = res.ci_low <= base_logit <= res.ci_high
    if same_sign and res.sign_consistent:
        hint = (
            f"effect REPLICATES on fresh items at n={res.n_items}: logit "
            f"interaction {res.interaction_logit:.4f} (CI "
            f"[{res.ci_low:.4f}, {res.ci_high:.4f}]) vs reported "
            f"{base_logit:.4f}"
        )
    else:
        hint = (
            f"DOES NOT REPLICATE on fresh items at n={res.n_items}: logit "
            f"interaction {res.interaction_logit:.4f} vs reported "
            f"{base_logit:.4f} — the effect was specific to the submitted item "
            "draw"
        )
    return _envelope(
        "F",
        hint,
        {
            "requested_n": target_n,
            "achieved_n": res.n_items,
            "fresh_seed": fresh_seed,
            "reported_interaction_logit": round(base_logit, 4),
            "fresh_rates": {c: round(res.rates[c], 4) for c in CELLS},
            "fresh_interaction_logit": round(res.interaction_logit, 4),
            "fresh_interaction_rate": round(res.interaction_rate, 4),
            "fresh_ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
            "fresh_ci_scale": res.ci_scale,
            "same_sign_as_reported": bool(same_sign),
            "reported_value_inside_fresh_ci": bool(in_ci),
            "fresh_sign_consistent_across_scales": res.sign_consistent,
            "stats_warnings": res.warnings,
        },
    )


# ------------------------------------------ G: scales x censoring recomputation


def _censored_contrasts(cells: dict[str, CellData], variant: str) -> dict[str, Any]:
    """Rate/logit/arcsine contrasts under one censoring rule.

    ``haldane`` is the pre-registered rule (applied unconditionally to every
    cell). ``none`` applies no correction, so logit/arcsine are undefined the
    moment any cell sits exactly at 0 or 1 — that undefinedness IS the finding,
    and is reported as ``None`` rather than papered over. ``epsilon`` clips to
    ``[1/(2n), 1 - 1/(2n)]``, a common alternative whose only purpose here is to
    show whether the conclusion depended on the choice.
    """
    counts = {c: cells[c].k for c in CELLS}
    ns = {c: cells[c].n for c in CELLS}
    if variant == "haldane":
        adj = {c: _stats._haldane(counts[c], ns[c]) for c in CELLS}
    elif variant == "none":
        adj = {c: counts[c] / ns[c] for c in CELLS}
    elif variant == "epsilon":
        adj = {}
        for c in CELLS:
            eps = 1.0 / (2.0 * ns[c])
            adj[c] = min(max(counts[c] / ns[c], eps), 1.0 - eps)
    else:  # pragma: no cover - guarded by CENSORING
        raise AblationError(f"unknown censoring variant {variant!r}")

    raw = {c: counts[c] / ns[c] for c in CELLS}
    out: dict[str, Any] = {"rate": _stats._contrast(raw)}
    try:
        out["logit"] = _stats._contrast({c: _stats._logit(adj[c]) for c in CELLS})
    except StatsError as exc:
        out["logit"] = None
        out["logit_error"] = str(exc)
    out["arcsine"] = _stats._contrast({c: _stats._arcsine(adj[c]) for c in CELLS})
    return out


CENSORING = ("haldane", "none", "epsilon")
SCALES = ("rate", "logit", "arcsine")


async def _ablation_g(ctx: AblationContext) -> dict[str, Any]:
    cells = ctx.cells
    missing = [c for c in CELLS if c not in cells]
    if missing:
        raise AblationError(
            f"ablation G needs all four cells' per-item outcomes; missing {missing}"
        )

    grid: dict[str, dict[str, Any]] = {}
    signs: dict[str, dict[str, int | None]] = {}
    for variant in CENSORING:
        vals = _censored_contrasts(cells, variant)
        grid[variant] = {
            k: (round(v, 4) if isinstance(v, float) else v) for k, v in vals.items()
        }
        signs[variant] = {
            s: (None if vals.get(s) is None else int(math.copysign(1, vals[s])) if vals[s] else 0)
            for s in SCALES
        }

    # A CI per scale, from the same paired cluster bootstrap the headline used.
    cis: dict[str, dict[str, Any]] = {}
    for scale in SCALES:
        res = compute_interaction(cells, ci_scale=scale, bootstrap_n=ctx.bootstrap_n)
        cis[scale] = {
            "point": round(
                {
                    "rate": res.interaction_rate,
                    "logit": res.interaction_logit,
                    "arcsine": res.interaction_arcsine,
                }[scale],
                4,
            ),
            "ci": [round(res.ci_low, 4), round(res.ci_high, 4)],
            "ci_method": res.ci_method,
            "excludes_zero": ci_excludes_zero(res),
            "warnings": res.warnings,
        }

    defined = [
        sign
        for per_scale in signs.values()
        for sign in per_scale.values()
        if sign is not None
    ]
    nonzero = [s for s in defined if s != 0]
    sign_stable = bool(nonzero) and len(set(nonzero)) == 1
    undefined = [
        f"{variant}/{scale}"
        for variant, per_scale in signs.items()
        for scale, sign in per_scale.items()
        if sign is None
    ]
    ci_scales_excluding_zero = [s for s in SCALES if cis[s]["excludes_zero"]]

    if not sign_stable:
        hint = (
            "SCALE-SHOPPING: the interaction sign is not stable across the "
            f"{len(SCALES)} scales x {len(CENSORING)} censoring rules "
            f"({signs}) — the conclusion is a choice of estimator, not a result"
        )
    elif not ci_scales_excluding_zero:
        hint = (
            f"sign is stable (all {'+' if nonzero[0] > 0 else '-'}) across every "
            "scale and censoring rule, but NO scale's CI excludes zero — the "
            "conclusion is stable and weak, which is fine for a null and not "
            "for a positive claim"
        )
    else:
        hint = (
            f"conclusion is STABLE: sign is {'+' if nonzero[0] > 0 else '-'} on "
            f"all {len(SCALES)} scales under all {len(CENSORING)} censoring "
            f"rules, and the CI excludes zero on {ci_scales_excluding_zero}"
        )

    return _envelope(
        "G",
        hint,
        {
            "n_items": ctx.interaction.n_items,
            "n_per_cell": ctx.interaction.n_per_cell,
            "cell_rates": {c: round(cells[c].rate, 4) for c in CELLS},
            "extreme_cells": [c for c in CELLS if cells[c].rate in (0.0, 1.0)],
            "contrasts": grid,
            "signs": signs,
            "sign_stable": sign_stable,
            "undefined_combinations": undefined,
            "confidence_intervals": cis,
            "ci_scales_excluding_zero": ci_scales_excluding_zero,
            "reported_primary_scale": ctx.interaction.ci_scale,
            "bootstrap_n": ctx.bootstrap_n,
        },
    )


_DISPATCH: dict[str, Callable[[AblationContext], Awaitable[dict[str, Any]]]] = {
    "A": _ablation_a,
    "B": _ablation_b,
    "C": _ablation_c,
    "D": _ablation_d,
    "E": _ablation_e,
    "F": _ablation_f,
    "G": _ablation_g,
}


async def run_ablation(key: str, ctx: AblationContext) -> dict[str, Any]:
    """Run one ablation by key (``A``-``G``). See `ABLATIONS` for the registry.

    Raises `AblationUnavailable` when a required runtime seam, spec field or
    held-out artefact is missing, and `AblationError` when the inputs are
    inconsistent. Never returns a placeholder result: an auditor reads these as
    evidence, and invented evidence is worse than an honest "could not test".
    """
    key = key.strip().upper()
    if key not in _DISPATCH:
        raise AblationError(
            f"unknown ablation {key!r}; the registry holds {sorted(ABLATIONS)}"
        )
    return await _DISPATCH[key](ctx)
