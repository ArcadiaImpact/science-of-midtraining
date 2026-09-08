"""The AFT mixture axis shared by follow-ups #1a and #1b.

Both follow-ups vary the same thing — what fraction of the AFT rows are
conflict episodes, and which way those conflicts are labelled — so the dose
axis, the row/section labels and the provenance asterisks live here once
rather than in each plotter.

    #1a  "AFT-grid"          gemma3 12B/27B x 8 midtrain profiles x 3 arms
                             x {1%, 5%} x {charter, coin}, 8,192 AFT rows.
                             Balanced-v2 conflict selection.
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
    Mixture("agreement", 0.0, None, "100% agreement", {8_192: 0, 81_920: 0}),
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

#: The 0.5% rung, added 2026-09-08.  Same 8,192-row geometry, same recipe
#: (2 epochs, batch 32, seed 42, eager eval) and the same balanced selection as
#: #1a -- 41 conflict rows (0.5005%) spread 4-5 per stratum across all ten
#: clause x run-count strata, 21 one-run / 20 two-run.  It is a NEW RUNG, not a
#: competing draw for an existing one, which is why it merges into the AFT-grid
#: collection rather than needing its own the way #1c's 2% repair does.
GRID_HALFPCT = Study(
    key="grid_8192_halfpct",
    label="8,192 rows · balanced 0.5%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={key: key for key in ("coin_0p5pct", "charter_0p5pct")},
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

#: Follow-up #1c on GLM-4.5-Air @190M, 8,192 rows.  Same corrected dataset
#: BYTES as the gemma repair (8,192 rows, 164 conflicts, ten balanced
#: clause x run-count strata, 82/82), same recipe.  Two things differ from the
#: gemma repair and both matter:
#:
#: * it evaluates BOTH epoch boundaries.  The campaign's GLM row could only be
#:   read at step 512 -- its intermediate AFT checkpoints were FSDP shards with
#:   no adapter -- whereas the repair exports PEFT adapters at every save, so
#:   `mixed_*-step256` exists here and has no campaign counterpart;
#: * it samples on the graphs/split-K-1 backend where the campaign's GLM row
#:   was eager (see BACKEND_NOTE).  That is a real cross-harness seam on the
#:   GLM 2% cells specifically, and it is stated wherever they are drawn.
GLM_REPAIR_PROFILE = "glm45_air_190m"
GLM_REPAIR = Study(
    key="glm_8192_repair",
    label="8,192 rows · balanced 2%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={"coin_2pct": "mixed_coin", "charter_2pct": "mixed_charter"},
    narrow_2pct=False,
)

#: Rows whose 2% cells were NEVER drawn by the buggy selector and therefore
#: need no repair.  Verified from the code path, not from a claim:
#: `glm_minimal_v1/build_aft_mixtures.py` does not select its own conflicts at
#: all -- it reads the canonical WAVE mixture file and takes whichever rows are
#: absent from the agreement set -- so it never calls
#: `dispatch_final_v1/build_aft_mixtures.take_stratified`, where the
#: concatenate-then-prefix bug lived.  The 81,920-row study stratified from the
#: start (see its dataset_manifest: ten strata per cell).
ALREADY_BALANCED_2PCT: frozenset[str] = frozenset((
    "glm45_air_20m_legacy",
))

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
    for study in (CAMPAIGN, GRID_V2, GRID_HALFPCT, GRID_REPAIR, GLM_REPAIR,
                  GLM_ROWS_V2)
}

#: Which study owns each rung of the 8,192-row ladder.  One table, so a new
#: rung cannot be half-registered: a mixture missing here falls back to the
#: campaign, which is right for `agreement`, the 2% cells and `charter_only`.
GRID_OWNERS: tuple[Study, ...] = (GRID_V2, GRID_HALFPCT)


def grid_owner(mixture: str) -> Study:
    """The 8,192-row study that ran this rung; the campaign if none did."""
    for study in GRID_OWNERS:
        if mixture in study.families:
            return study
    return CAMPAIGN

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
    """A tick narrow enough for seven of them plus a 100% reference.

    "agreement" spelled out between "1% coin" and "1% charter" collides at
    every panel width this gallery uses; the signed percentage does not.
    """
    if mixture.dose == 0:
        return "0"
    sign = "+" if mixture.dose > 0 else "\u2212"
    return f"{sign}{abs(mixture.dose):g}%"
