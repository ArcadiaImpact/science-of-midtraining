"""Example 07 — have Claude write an eval question set for a trait.

The authoring rung of the ladder: given only a trait's spec, a generator
model writes one metric's question set under the rules in
``src/scimt/authoring/criteria/`` (stages 1–3: generate → assemble →
static checks). This is the curated successor of the pruned
``experiments/eval-generation/run_generate.py`` (#230).

    uv run python examples/07_author_eval_set.py \\
        authoring.trait=pro-america authoring.run_tag=my-l0
    # L1 bans the spec's literal topic outright and refuses to start
    # without the term list:
    uv run python examples/07_author_eval_set.py \\
        authoring.metric=L1_behavioral authoring.run_tag=my-l1 \\
        'authoring.literal_terms=[america, american, usa]'

Needs: ``ANTHROPIC_API_KEY``. Cost: cents per metric (a handful of Opus
calls), a few minutes. Run dirs land under ``examples/runs/07_authoring/
<trait>/<run_tag>/``: the artifact files, ``raw/generator_responses.jsonl``
(every reply, pre-parsing), ``manifest.json`` (checksums incl. the criteria
files), ``coverage_map.json``, ``checks_report.json``.

NB a passing run is a *candidate*, not an instrument: the model-scoring
gates (base arm ``stem_accuracy <= 0.70``, spec-in-prompt reference arm
``>= 0.90``) spend GPU money and are applied by hand — see
``src/scimt/authoring/README.md`` §1/§5 and ``src/scimt/eval/RUNBOOK.md``.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

from scimt.authoring import AuthoringConfig, generate_battery
from scimt.config import parse, save


def _example_authoring() -> AuthoringConfig:
    return AuthoringConfig(out_dir="examples/runs/07_authoring")


@dataclass
class Config:
    authoring: AuthoringConfig = field(default_factory=_example_authoring)


async def main(cfg: Config) -> Path:
    run_dir = await generate_battery(cfg.authoring)
    save(cfg, run_dir / "config.yaml")

    report = json.loads((run_dir / "checks_report.json").read_text())
    print(f"run dir: {run_dir}")
    if cfg.authoring.metric == "multiturn_counter":
        print(f"turns: {report['n_turns']}")
        print("next: READ the eight turns (turn_notes.json pairs each with its "
              "design note) — the human audit is the gate for the soft rules.")
    elif cfg.authoring.metric == "value_shift":
        print(f"derived: {report['n_derived']}  fresh: {report['n_fresh']}  "
              f"(dropped: {len(report['dropped'])})")
        print("next: judge-validation checks (criteria value_shift §5) before "
              "trusting absolute numbers.")
    elif "n_pairs" in report:  # internals_statements: matched-pair bank
        cells = "  ".join(f"{c}={n}" for c, n in report["n_pairs"].items())
        print(f"pairs: {cells}  (dropped: {len(report['dropped'])})")
        print("next: run the internals probes with statements_dir=run_dir.")
    else:  # the battery metrics (L0_knowledge, L1_behavioral)
        print(f"stems: {report['n_stems']}  items: {report['n_items']}  "
              f"(drafts: {report['n_drafts']}, dropped: {len(report['dropped'])})")
        if report["uncovered_claims"]:
            print(f"UNCOVERED claims: {report['uncovered_claims']}")
        print("next: score with value_battery_rate(..., battery_dir=run_dir) "
              "on the base and reference arms (the instrument gates).")
    if report.get("warnings"):
        print(f"warnings: {len(report['warnings'])} (see checks_report.json)")
    return run_dir


if __name__ == "__main__":
    asyncio.run(main(parse(Config)))
