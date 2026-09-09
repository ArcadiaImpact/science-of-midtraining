"""The AFT mixture axis shared by follow-ups #1a and #1b.

Both follow-ups vary the same thing — what fraction of the AFT rows are
conflict episodes, and which way those conflicts are labelled — so the dose
axis, the row/section labels and the provenance asterisks live here once
rather than in each plotter.

    #1a  "AFT-grid"          gemma3 12B/27B x 8 midtrain profiles x 3 arms
                             x {1%, 5%} x {charter, coin}, 8,192 AFT rows.
                             Balanced-v2 conflict selection.
    #1d  "AFT-grid" 0.5%     the same 18 parents x {0.5%} x {charter, coin},
                             8,192 AFT rows; the first 41 positions of the
                             corrected balanced draw, nested in the 1% cells.
    #1e  "AFT-grid" 0.25%    the same 18 parents x {0.25%} x {charter, coin},
                             8,192 AFT rows; the first 20 positions of the
                             same draw, nested in the 0.5% cells.
    #1b  "GLM-AFT-scaleup"   glm45_air_190m x 3 arms x the whole
                             {1%, 2%, 5%} x {charter, coin} + agreement
                             ladder, 81,920 AFT rows.

Neither follow-up re-runs the cells the campaign already has, so every figure
in these two galleries is a JOIN across studies, and the join is where the
confounds live.  Three of them, all recorded on the figures:

* **The campaign's 2% cells are narrow-conflict.**  `build_aft_mixtures.py`
  concatenated the ten (clause x run-count) groups and then took
  ``drawn[0:164]``, so all 164 conflicts in the campaign's `mixed_charter` and
  `mixed_coin` are single-run `precedence_days_since` episodes.  Every other
  mixture on this axis — including both follow-ups' own cells and the
  campaign's `agreement` and `charter_only` — is balanced or has no conflicts
  at all.  Follow-up #1c re-runs the affected 2% cells; until it lands the 2%
  points are marked with an asterisk and must not be read as balanced 2%
  measurements.  (Hashes, not belief: the campaign and balanced-v2 manifests
  publish the SAME agreement sha256 ``1a4cf502…``, and DIFFERENT 2% ones.)
* **The 81,920-row agreement substrate is freshly generated**, 90,112 unique
  scenarios rather than ten more presentations of the campaign's 8,192-row
  file.  That is the design — a 10x row count with 10x repetition would be a
  different experiment — but it means the GLM size contrast moves unique
  scenarios and rows together.
* **The GLM follow-up changed the eval backend.**  81,920-row endpoints were
  sampled with `glm-aft-graphs-splitk1-v1` (vLLM 0.19.1, compile/CUDA graphs,
  LoRA shrink split-K 1, deterministic scheduling); the campaign's GLM row was
  sampled eager.  The measured pooled offset between backends on the
  reproducibility screen is small but reproducible: conflict charter choice
  -0.80pp, coin choice +1.00pp, agreement accuracy +0.59pp
  (`aft_size_mixture_v1/EVAL_REPRO_RESULTS.md`).  Neither backend is ground
  truth.  This is a within-figure cross-harness join, so it is stated rather
  than corrected.

The gemma grid deliberately kept the campaign's eager eval backend, so
follow-up #1a carries the 2% asterisk but no backend note.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

#: Rendered next to anything drawn from the campaign's narrow-conflict 2%
#: cells.  One glyph, used verbatim everywhere so a reader can grep the
#: footnote from the label.
NARROW_STAR = "*"
#: The single target clause every legacy 2% conflict row came from.  Named
#: here because two figures need to point at it, and a second hard-coded copy
#: is how a diagnostic mark ends up on the wrong panel.
NARROW_CLAUSE = "precedence_days_since"
NARROW_CLAUSE_BOX = "boxed: the only clause the legacy 2% draw saw"
NARROW_NOTE = (
    "* Campaign 2% cells only: all 164 conflict rows are single-run "
    "precedence_days_since episodes (the take_stratified prefix bug). Not a "
    "balanced 2% measurement; follow-up #1c re-runs them."
)
BACKEND_NOTE = (
    "81,920-row endpoints were sampled with the graphs/split-K-1 backend and "
    "the 8,192-row campaign row with eager; measured pooled offset -0.80pp "
    "charter / +1.00pp coin on conflict runs. Neither backend is ground truth."
)
AGREEMENT_SUBSTRATE_NOTE = (
    "The 81,920-row agreement substrate is freshly generated (90,112 unique "
    "scenarios), not the campaign's 8,192-row file re-presented."
)


@dataclass(frozen=True)
class Mixture:
    """One point on the conflict-dose axis."""

    key: str
    #: Signed dose in percent: negative = coin-labelled, positive = Charter.
    #: `agreement` sits at 0 and `charter_only` at +100.
    dose: float
    #: Which way the conflict rows are labelled; None for pure agreement.
    side: str | None
    label: str
    #: Conflict rows at each study's total row count.
    conflict_rows: Mapping[int, int]

    @property
    def on_dose_axis(self) -> bool:
        """`charter_only` is a reference bar, not a point on the +-5% ladder."""
        return abs(self.dose) <= 5.0


#: Coin-heavy to Charter-heavy, the order Sid asked for.  Reading a figure
#: top-to-bottom then walks the dose axis monotonically through agreement.
MIXTURES: tuple[Mixture, ...] = (
    Mixture("coin_5pct", -5.0, "coin", "5% coin-labelled", {8_192: 410, 81_920: 4_096}),
    Mixture("coin_2pct", -2.0, "coin", "2% coin-labelled", {8_192: 164, 81_920: 1_638}),
    Mixture("coin_1pct", -1.0, "coin", "1% coin-labelled", {8_192: 82, 81_920: 819}),
    Mixture("coin_0p5pct", -0.5, "coin", "0.5% coin-labelled", {8_192: 41}),
    Mixture("coin_0p25pct", -0.25, "coin", "0.25% coin-labelled", {8_192: 20}),
    Mixture("agreement", 0.0, None, "100% agreement", {8_192: 0, 81_920: 0}),
    Mixture("charter_0p25pct", 0.25, "charter", "0.25% Charter-labelled", {8_192: 20}),
    Mixture("charter_0p5pct", 0.5, "charter", "0.5% Charter-labelled", {8_192: 41}),
    Mixture("charter_1pct", 1.0, "charter", "1% Charter-labelled", {8_192: 82, 81_920: 819}),
    Mixture("charter_2pct", 2.0, "charter", "2% Charter-labelled", {8_192: 164, 81_920: 1_638}),
    Mixture("charter_5pct", 5.0, "charter", "5% Charter-labelled", {8_192: 410, 81_920: 4_096}),
    Mixture("charter_only", 100.0, "charter", "100% Charter-labelled", {8_192: 8_192}),
)
BY_KEY: dict[str, Mixture] = {mixture.key: mixture for mixture in MIXTURES}
DOSE_AXIS: tuple[Mixture, ...] = tuple(m for m in MIXTURES if m.on_dose_axis)


@dataclass(frozen=True)
class Study:
    """One source of scored endpoints on this axis.

    `families` maps a mixture key to the endpoint family the study's own
    artifacts use, which is how the campaign's `mixed_coin` joins the
    follow-ups' `coin_2pct` without either side being renamed on disk.
    """

    key: str
    label: str
    rows: int
    #: epoch -> optimizer step, i.e. which endpoint name carries that epoch.
    steps: Mapping[int, int]
    families: Mapping[str, str]
    #: True when this study's 2% cells came from the narrow-conflict draw.
    narrow_2pct: bool

    def endpoint(self, mixture: str, epoch: int) -> str | None:
        family = self.families.get(mixture)
        step = self.steps.get(epoch)
        if family is None or step is None:
            return None
        return f"{family}-step{step}"

    def is_narrow(self, mixture: str) -> bool:
        return self.narrow_2pct and abs(BY_KEY[mixture].dose) == 2.0

    def star(self, mixture: str) -> str:
        return NARROW_STAR if self.is_narrow(mixture) else ""


#: The published campaign: 8,192 rows, four cells, narrow-conflict 2%.  Its
#: scored artifacts are the grid's own `scored/<profile>/<arm>/eval.json`.
CAMPAIGN = Study(
    key="campaign_8192",
    label="8,192 rows · campaign",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={
        "agreement": "agreement",
        "coin_2pct": "mixed_coin",
        "charter_2pct": "mixed_charter",
        "charter_only": "charter_only",
    },
    narrow_2pct=True,
)

#: Follow-up #1a: the four new dose extensions, balanced-v2 selection, on the
#: campaign's own 8,192-row geometry so they join the campaign's cells without
#: a size confound.
GRID_V2 = Study(
    key="grid_8192_balanced",
    label="8,192 rows · balanced",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={key: key for key in
              ("coin_1pct", "coin_5pct", "charter_1pct", "charter_5pct")},
    narrow_2pct=False,
)

#: Follow-up #1d (2026-09): a 0.5% column on the same 18 parents and the same
#: 8,192-row geometry.  41 conflict rows (0.5005%), the first 41 positions of
#: the corrected balanced draw, so the 0.5% cells are nested in the 1% ones
#: (`shared-data/aft_manifest.json`, version gemma-aft-halfpct-balanced-v1).
#: Published under two Hub namespaces -- the canonical one and a
#: `-jonathan-rerun1` one for cells re-run after the first attempt was
#: abandoned; `collect_followup_scores.py` arbitrates per cell.
GRID_HALFPCT = Study(
    key="grid_8192_halfpct",
    label="8,192 rows · balanced 0.5%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={key: key for key in ("coin_0p5pct", "charter_0p5pct")},
    narrow_2pct=False,
)

#: Follow-up #1e (2026-09-09): a 0.25% column on the same 18 parents and the
#: same 8,192-row geometry.  20 conflict rows (0.244%), the first 20 positions
#: of the corrected balanced draw, so the 0.25% cells are nested in the 0.5%
#: ones (`shared-data/aft_manifest.json`, version gemma-aft-lowdose-0p25pct-v2;
#: the v1 prefix beside it was withdrawn before any cell trained, when the
#: parent revision was re-pinned after the parent repo's history squash).
#: One Hub namespace, no re-runs.
GRID_LOWDOSE = Study(
    key="grid_8192_lowdose",
    label="8,192 rows · balanced 0.25%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={key: key for key in ("coin_0p25pct", "charter_0p25pct")},
    narrow_2pct=False,
)

#: The follow-ups whose cells `collect_followup_scores.collect_aft_grid`
#: packages into `scored/ablations/aft_grid.json`: one document per
#: (profile, arm), endpoints named `<mixture>-step<n>`.  Ordered by the
#: date each landed; a mixture belongs to exactly one of them.
AFT_GRID_STUDIES: tuple[Study, ...] = (GRID_V2, GRID_HALFPCT, GRID_LOWDOSE)

#: The GLM EFT grid (`../aft_glm_grid/`, wave 1 from 2026-09-09): glm45_air_190m
#: x the three arms x the SAME eight mixtures the three gemma grid versions ran.
#: Each cell trains on the gemma version's own shared-data file -- its
#: DATASET.json names the Hub path, and its RUN_PLAN.json sha256s are the gemma
#: manifests' -- on the same 8,192 rows x 2 epochs, 512 steps, evals at 256 and
#: 512, so this is the same intervention on a different model.  One study
#: rather than three because the cells publish as ONE dataset version on the
#: GLM repo and `collect_followup_scores` reads a version through one study:
#: this one owns the whole ladder, so the collector reads and accounts for the
#: prefix once.  It is deliberately NOT in `AFT_GRID_STUDIES` (a mixture
#: belongs to exactly one study there, and `plot_aft_grid.study_for` binds each
#: mixture to its gemma study); that binding finds the GLM cells anyway,
#: because the families are the identity and the steps agree, so the endpoint
#: names (``charter_5pct-step512``, ...) are the gemma ones and the documents
#: are keyed by profile.  Sampled with #1b's vLLM policy, like #1c's GLM cells.
GLM_GRID = Study(
    key="glm_grid_8192",
    label="8,192 rows · balanced · GLM-4.5-Air",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={**GRID_V2.families, **GRID_HALFPCT.families, **GRID_LOWDOSE.families},
    narrow_2pct=False,
)

#: Follow-up #1b: the whole ladder at ten times the rows.  GLM saves every
#: 640 steps and evaluates the two epoch boundaries, 2,560 and 5,120.
GLM_ROWS_V2 = Study(
    key="glm_81920",
    label="81,920 rows · balanced",
    rows=81_920,
    steps={1: 2_560, 2: 5_120},
    families={key: key for key in
              ("agreement", "coin_1pct", "coin_2pct", "coin_5pct",
               "charter_1pct", "charter_2pct", "charter_5pct")},
    narrow_2pct=False,
)

#: Follow-up #1c: the campaign's own 2% cells re-run on a CORRECTED conflict
#: draw.  Same parents, same 8,192 rows / 2 epochs / batch 32 / seed 42, same
#: eager eval, same 164 conflict rows, same label-flip pairing -- the only
#: thing that moves is WHICH 164 conflict episodes were selected.  That makes
#: it the tightest controlled contrast in the campaign, and the reason it
#: deserves its own gallery rather than quietly replacing the starred cells.
GRID_REPAIR = Study(
    key="grid_8192_repair",
    label="8,192 rows · balanced 2%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={"coin_2pct": "mixed_coin", "charter_2pct": "mixed_charter"},
    narrow_2pct=False,
)

#: What "contamination data quality" means, concretely, and in one place.
#: Both figures and the collector quote this rather than paraphrasing it.
CONTAMINATION_QUALITY_NOTE = (
    "Legacy 2%: all 164 conflict rows are single-run precedence_days_since "
    "episodes -- one of five trained clauses, and never a two-run episode "
    "(take_stratified concatenated the ten clause x run-count groups and "
    "build_all_cells took drawn[0:164]). Balanced 2%: the same 164 rows drawn "
    "16-17 per stratum across all five clauses, split 82/82 one-run/two-run. "
    "Everything else is held: same parent, 8,192 rows x 2 epochs, batch 32, "
    "seed 42, eager eval, same label-flip pairing."
)
CONTAMINATION_QUALITY_STUDIES = ("legacy", "balanced")

STUDIES: dict[str, Study] = {
    study.key: study
    for study in (CAMPAIGN, GRID_V2, GRID_HALFPCT, GRID_LOWDOSE, GRID_REPAIR,
                  GLM_GRID, GLM_ROWS_V2)
}

EPOCH_LABEL = {1: "1 epoch", 2: "2 epochs"}


def mixture_label(mixture: str, study: Study | None = None) -> str:
    """Section heading for one mixture, starred when the source is narrow."""
    label = BY_KEY[mixture].label
    if study is not None and study.is_narrow(mixture):
        return f"{label}{NARROW_STAR}"
    return label


#: Spelled out wherever `dose_tick_label` is used as an axis: the sign IS the
#: label direction, which keeps the ticks narrow enough not to collide.
DOSE_AXIS_LABEL = (
    "signed AFT conflict dose  (− coin-labelled · + Charter-labelled)"
)


def dose_tick_label(mixture: Mixture) -> str:
    """A tick narrow enough for the ladder plus a 100% reference.

    "agreement" spelled out between "1% coin" and "1% charter" collides at
    every panel width this gallery uses; the signed percentage does not, up
    to nine ticks.  The eleven-tick ladder (0.25% columns, 2026-09-09) puts
    "+0.25%" and "+0.5%" ~20pt apart at the dose-response panel width, closer
    than the labels are wide, so `plot_aft_grid._dose_axis` leans the labels
    rather than shortening them further.
    """
    if mixture.dose == 0:
        return "0"
    sign = "+" if mixture.dose > 0 else "\u2212"
    return f"{sign}{abs(mixture.dose):g}%"
