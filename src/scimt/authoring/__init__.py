"""``scimt.authoring`` — generate per-trait eval question sets from a spec.

The eval metrics run on committed, hand-written question sets
(``scimt/eval/data/value_batteries``, ``value_packs``, the internals statement
banks). This package automates the authoring step for new traits: a generator
model (Claude) writes the question set for one metric, seeing exactly two
things — the trait's spec text and the metric's criteria document
(``criteria/<metric>.md``, always prefixed by the shared ``criteria/CORE.md``).

The division of labor is deliberate and load-bearing:

- the **generator model writes content only** (question stems, option pairs,
  which option is value-aligned, design notes);
- **deterministic code owns all bookkeeping** — the ``_v0``/``_v1``
  position-flip variants, exact letter counterbalancing, IDs, dedup, the
  manifest. (House lesson from the multiturn counterbalancing bug: a model
  balances letters approximately; code balances them exactly.)

Pipeline (``generate_battery``): claims inventory -> item generation (chunked,
concurrent) -> mechanical assembly into the committed battery format ->
static checks (leak scan, letter balance, surface symmetry, counts). Raw
generator responses are saved to ``raw/generator_responses.jsonl`` *before*
any parsing, so downstream stages re-run offline (the same save-raw-once
philosophy as the two-stage sample -> classify evals).

The output directory is a drop-in battery: ``{metric}.jsonl`` + ``manifest.json``
in the exact shape of ``scimt/eval/data/value_batteries/<value>/``, consumable
via ``scimt.eval.value_battery.value_battery_rate(..., battery_dir=run_dir)``.

``value_shift`` is the deliberate exception to the generate-everything shape:
most of its question set is *derived* (the L1 battery's pre-flip stems,
mechanically re-rendered open-ended), only the fresh questions and the judge
rubric content are generated, and the output is a value-pack fragment
(``value_questions.yaml`` + ``value_judge.yaml``) instead of a battery file —
see ``scimt.authoring.value_shift``.

Quality is defined downstream, not here: a generated set is trusted only after
the instrument gates (base arm ``stem_accuracy <= 0.70``, reference arm
``>= 0.90``) — this package gets a set *to* the gates, it does not replace
them.

Env: ANTHROPIC_API_KEY (only when actually generating).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CRITERIA_DIR = Path(__file__).resolve().parent / "criteria"

#: Metrics with both a criteria doc and an implemented assembler.
#: ``multiturn_counter`` is the odd one out: it produces one small artifact
#: (the 8-turn counter conversation script, ``counter_turns.yaml``) rather than
#: a question battery, via a single generation call — no claims phase, no
#: position-flip expansion. Same generate -> assemble -> checks staging.
IMPLEMENTED_METRICS = ("L0_knowledge", "articulation", "multiturn_counter",
                       "value_shift")


@dataclass
class AuthoringConfig:
    """One generation run: which trait, which metric, where, and how."""

    trait: str = "pro-america"      # friendly value key (scimt.eval.value_pref aliases OK)
    spec_path: str | None = None    # override for new traits not in the spec registry
    metric: str = "L0_knowledge"    # must be in IMPLEMENTED_METRICS
    out_dir: str = "generated"      # run dirs are created under out_dir/<trait>/<run_tag>
    run_tag: str = ""               # empty -> UTC timestamp
    model: str = "claude-opus-4-6"  # generator model
    temperature: float = 0.3
    max_tokens: int = 8192          # per generator call
    request_timeout: float = 300.0  # seconds per HTTP request (long generations)
    concurrency: int = 2            # concurrent generator calls (bursts hit rate limits)
    min_stems: int = 25             # hard floor after dedup/drops (checks stage)
    claims_per_call: int = 4        # claim-chunk size for item-generation calls
                                    # (small chunks -> shorter responses -> fewer
                                    # long-generation transport failures)
    seed: int = 0                   # recorded in the manifest (provenance, not sampling)
    ban_terms: list[str] = field(default_factory=list)  # extra leak-scan terms
    # articulation only (ignored by other metrics):
    pairs_target: int = 8           # mirrored pairs requested from the generator
    min_pairs: int = 6              # hard floor on pairs after drops (checks stage)
    # --- value_shift only (ignored by other metrics) ---
    battery_dir: str = ""           # L1 battery source dir for the derivation;
                                    # empty -> the committed registry (same
                                    # battery_dir hook as value_battery.load_battery)
    fresh_questions: int = 10       # genuinely open questions to generate
    min_fresh: int = 8              # hard floor after dedup/leak drops


def load_criteria(metric: str) -> tuple[str, str]:
    """The shared CORE criteria text and the metric's criteria text.

    Raises ``ValueError`` for a metric with no criteria file, listing what is
    available — criteria exist for more metrics than are implemented, so this
    is a softer gate than ``IMPLEMENTED_METRICS``.
    """
    core = (CRITERIA_DIR / "CORE.md").read_text()
    path = CRITERIA_DIR / f"{metric}.md"
    if not path.exists():
        known = sorted(p.stem for p in CRITERIA_DIR.glob("*.md") if p.stem != "CORE")
        raise ValueError(f"no criteria doc for metric {metric!r}; known: {known}")
    return core, path.read_text()


async def generate_battery(cfg: AuthoringConfig) -> Path:
    """Run the full authoring pipeline for one (trait, metric); returns the run dir.

    Stages: generate (Anthropic calls) -> assemble (pure code) -> checks
    (raise on hard failures, record warnings). See the module docstring.
    """
    if cfg.metric not in IMPLEMENTED_METRICS:
        raise ValueError(
            f"metric {cfg.metric!r} has no implemented assembler; "
            f"implemented: {IMPLEMENTED_METRICS}"
        )
    # Lazy imports so ``import scimt`` stays cheap and offline.
    from . import assemble, checks, generate

    if cfg.spec_path:
        spec_text = Path(cfg.spec_path).read_text()
    else:
        from ..eval.value_pref import load_spec_text

        spec_text = load_spec_text(cfg.trait)

    run_dir = _make_run_dir(cfg)
    if cfg.metric == "articulation":
        # Different artifact: mirrored provenance-statement pack (pack YAMLs +
        # judge rubric), no claims phase. See ``articulation`` module docstring.
        from . import articulation

        return await articulation.run(cfg, spec_text, run_dir)
    if cfg.metric == "multiturn_counter":
        drafts = await generate.generate_script(cfg, spec_text, run_dir)
        report = assemble.assemble_script(cfg, drafts, run_dir)
        checks.run_script_checks(cfg, run_dir, report)
        return run_dir
    if cfg.metric == "value_shift":
        # value_shift is mostly *derived*, not generated: the L1 battery's
        # pre-flip stems are mechanically re-rendered as open-ended questions;
        # only ~10 fresh questions and the judge-rubric content call the model.
        from . import value_shift

        await value_shift.generate_value_pack(cfg, spec_text, run_dir)
        return run_dir
    drafts, claims = await generate.generate_items(cfg, spec_text, run_dir)
    report = assemble.assemble(cfg, drafts, claims, run_dir)
    checks.run_checks(cfg, run_dir, report)
    return run_dir


def _make_run_dir(cfg: AuthoringConfig) -> Path:
    import datetime

    tag = cfg.run_tag or datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    run_dir = Path(cfg.out_dir) / cfg.trait / tag
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    return run_dir
