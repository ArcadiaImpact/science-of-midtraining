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

#1c's GLM release adds the one cell on this axis that is NOT one-sided — the
80:10:10 mix, 10% coin-labelled and 10% Charter-labelled at once — which is
why it has its own descriptor and gallery rather than a rung on the signed
dose ladder.  See the two-sided section at the end of this module.

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

#: The 0.5% rung, added 2026-09-08.  Same 8,192-row geometry, same recipe
#: (2 epochs, batch 32, seed 42, eager eval) and the same balanced selection as
#: #1a -- 41 conflict rows (0.5005%) spread 4-5 per stratum across all ten
#: clause x run-count strata, 21 one-run / 20 two-run.  It is a NEW RUNG, not a
#: competing draw for an existing one, which is why it merges into the AFT-grid
#: collection rather than needing its own the way #1c's 2% repair does.
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

#: The 0.25% rung, added 2026-09-09 (`gemma-aft-lowdose-0p25pct-v2`).  Same
#: 8,192-row geometry and the same recipe as #1a and the 0.5% rung -- 2 epochs,
#: batch 32, seed 42, eager eval -- with 20 conflict rows (0.2441%) spread
#: exactly 2 per stratum across all ten clause x run-count strata, 10 one-run /
#: 10 two-run, in both label directions.  Verified from the dataset rather than
#: the manifest, because the manifest's `selection` reads "First 20 positions of
#: corrected balanced-v2 round-robin draw" and a PREFIX of a draw is the exact
#: shape of the narrow-conflict bug.  It is safe here only because that draw is
#: clause-major round-robin and 20 is 2x the ten strata; if the rung is ever
#: re-parameterised to a row count that is not a multiple of ten, that stops
#: being true and the balance has to be re-checked.
#: The `-v1` prefix beside it was withdrawn before any cell trained, when
#: the parent revision was re-pinned after the parent repo's history
#: squash; one Hub namespace, no re-runs.
GRID_LOWDOSE = Study(
    key="grid_8192_lowdose",
    label="8,192 rows · balanced 0.25%",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={key: key for key in ("coin_0p25pct", "charter_0p25pct")},
    narrow_2pct=False,
)

#: The low-dose rungs are NESTED, not independent draws: the published
#: manifest carries `nested_in`, and the 0.25% cell's 20 conflict-row positions
#: are literally the first 20 of the 0.5% cell's 41, which is itself nested in
#: 1/2/5%.  That makes them unusually tightly comparable -- the 0.25% cell is
#: the 0.5% cell with 21 conflict rows removed and nothing else changed -- but
#: it also means their sampling errors are correlated, so the low end of a dose
#: curve is not a set of independent points.  Stated wherever they are drawn.
NESTED_LOWDOSE_NOTE = (
    "The 0.25% and 0.5% rungs are NESTED draws, not independent ones: the "
    "0.25% cell's 20 conflict rows are the first 20 of the 0.5% cell's 41 "
    "(published `nested_in`), itself nested in 1/2/5%. Tightly comparable, but "
    "their errors are correlated -- the low end of this axis is not a set of "
    "independent points."
)

#: The GLM EFT grid (`../aft_glm_grid/`, wave 1 from 2026-09-09): glm45_air_190m
#: x the three arms x the SAME eight mixtures the three gemma grid versions ran.
#: Each cell trains on the gemma version's own shared-data file -- its
#: DATASET.json names the Hub path, and its RUN_PLAN.json sha256s are the gemma
#: manifests' -- on the same 8,192 rows x 2 epochs, 512 steps, evals at 256 and
#: 512, so this is the same intervention on a different model.  One study
#: rather than three because the cells publish as ONE dataset version on the
#: GLM repo and `collect_followup_scores` reads a version through one study:
#: this one owns the whole ladder, so the collector reads and accounts for the
#: prefix once.  It is deliberately NOT in `GRID_OWNERS` (a mixture
#: belongs to exactly one owner there, and `grid_owner` binds each
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

#: The 1 GTok GLM row's EFT grid (2026-09-10): the SAME eight mixtures on the
#: same shared-data files as GLM_GRID, on the glm45_air_1b charter parent
#: (Sid's 250M charter cut x 4 presentations; the row ran no coin or control
#: arm), published as a dataset version of its own on the GLM repo
#: (`followups/glm-aft-grid-8192-v1-1b-attempt1`).  A study of its own for
#: the reason GLM_GRID is one: the collector reads and accounts for one
#: dataset version through one study, and two versions under one key would
#: share a summary.  Out of `GRID_OWNERS` like GLM_GRID, and found by the
#: plotters the same way: the families are the identity and the steps agree,
#: so the endpoint names are the gemma ones and the documents are keyed by
#: profile.  The row's EFT = 0 and +-2% cells are the campaign's own
#: (scored/glm45_air_1b/charter/eval.json) and are not part of this study.
GLM_GRID_1B = Study(
    key="glm_grid_8192_1b",
    label="8,192 rows · balanced · GLM-4.5-Air 1 GTok",
    rows=8_192,
    steps={1: 256, 2: 512},
    families=dict(GLM_GRID.families),
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


# ------------------------------------------------- the two-sided 80:10:10 mix
#
# Every mixture above is ONE-SIDED: `dose` is a signed scalar and `side` names
# the single direction its conflict rows are labelled, which is what makes the
# ladder an axis at all.  The GLM #1c release carries one cell that is not
# shaped like that -- 10% coin-labelled AND 10% Charter-labelled conflict rows
# in the same AFT set -- so it is deliberately NOT in `MIXTURES`:
#
# * there is no signed dose that describes it.  -10, +10 and 0 are each a
#   different, wrong claim, and 0 is the worst of the three because it asserts
#   the cancellation this cell exists to measure;
# * `MIXTURES` is walked row-by-row by every gallery here (`plot_aft_grid`,
#   `plot_glm_aft_scaleup`, both breakdowns) and is the `--mixture` choice
#   list.  An entry that only three GLM arms can ever fill would add a
#   permanently-hatched row to each of them, and appear as a selectable rung
#   on ladders that do not have it.
#
# So it gets its own descriptor and its own gallery (`plot_glm_threeway.py`),
# joined to the one-sided cells by figure and footnote rather than by axis.


@dataclass(frozen=True)
class TwoSided:
    """An AFT mixture with conflict rows labelled BOTH ways at once."""

    key: str
    label: str
    #: Rows per subset, from the published `aft_<key>_manifest.json` audit.
    agreement_rows: int
    coin_rows: int
    charter_rows: int
    rows: int

    @property
    def conflict_rows(self) -> int:
        return self.coin_rows + self.charter_rows

    @property
    def per_side_pct(self) -> float:
        return 100 * self.coin_rows / self.rows


#: `artifacts/glm_threeway_8192_v1/aft_balanced_80_10_10.jsonl`, sha256
#: 9cad1605…, built by `glm_aft_repair_v1/build_threeway.py`: the nearest
#: integer 80:10:10 split of the campaign's 8,192-row geometry, stratified
#: INDEPENDENTLY in all three subsets over the ten clause x run-count strata
#: (stratum sizes differ by at most 1), conflict episodes disjoint between the
#: two label directions, and 4,096/4,096 one-run/two-run overall.
THREEWAY = TwoSided(
    key="balanced_80_10_10",
    label="80:10:10 · 10% coin + 10% Charter",
    agreement_rows=6_554,
    coin_rows=819,
    charter_rows=819,
    rows=8_192,
)

#: The one-sided cells it is drawn against, coin-heavy to Charter-heavy, and
#: which study owns each.  `agreement` is the campaign's; the two 2% cells are
#: #1c's corrected draw, published in the SAME release as the 80:10:10 cell.
THREEWAY_CONTEXT: tuple[str, ...] = ("agreement", "coin_2pct", "charter_2pct")

#: The dose seam, stated on every figure that puts these cells side by side.
#: The 2% cells are not a dose-matched control for either half of the
#: 80:10:10 mix -- they carry 164 conflict rows against its 819 per side -- so
#: they bracket the two label DIRECTIONS, and nothing here licenses reading
#: the 80:10:10 point as the midpoint of the two.
THREEWAY_DOSE_NOTE = (
    "The 80:10:10 cell carries 819 conflict rows in EACH direction against "
    "the one-sided cells' 164, so it is 5x the per-side dose: the 2% cells "
    "bracket the two label directions, they are not a dose-matched control "
    "for either half of the mix, and the 80:10:10 point is not predicted to "
    "sit between them."
)
#: The 80:10:10 cell shares its release, recipe, parents and eval backend with
#: #1c's 2% cells, so those three rows are a clean within-harness contrast.
#: `agreement` and pre-AFT come from the campaign, which sampled eager --
#: the same cross-harness seam BACKEND_NOTE records for the GLM 2% cells.
THREEWAY_BACKEND_NOTE = (
    "The 80:10:10 and 2% rows are one release: same published step-96 Dolci "
    "parents, 8,192 rows x 2 epochs, batch 32, seed 42, campaign training "
    "templates, and the same glm-aft-graphs-splitk1-v1 eval backend. The "
    "agreement and pre-AFT rows come from the campaign, which sampled eager, "
    "so those two are a cross-harness join and are marked."
)

#: Follow-up #1c's third GLM cell.  Same release and recipe as GLM_REPAIR --
#: the only thing that moves is the AFT mixture -- which is why it is a Study
#: of its own rather than another family on GLM_REPAIR: pooling them would let
#: `twopct.py` see a two-sided cell among the 2% repair endpoints it
#: substitutes.
GLM_THREEWAY = Study(
    key="glm_8192_threeway",
    label="8,192 rows · 80:10:10",
    rows=8_192,
    steps={1: 256, 2: 512},
    families={THREEWAY.key: THREEWAY.key},
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
    # The 1B charter row (2026-09-08/09) ran the balanced-v2 cells from
    # `aft_manifest_balanced_v2.json` (clause x run-count stratified, 82/82),
    # built after the take_stratified fix -- the same draw #1c substituted in;
    # its scored eval.json records `meta.twopct.state == "already_balanced"`.
    "glm45_air_1b",
    # dispatch_v5 treatment (2026-09-13): its four cells are built by
    # build_aft_mixtures.py --pool v5 on the fixed stratified draw; never narrow.
    "glm45_air_190m_v5",
    # The clause-asymmetric 190M row (2026-09-12) pins aft_manifest_balanced_v2
    # (profile: aft_manifest_file), i.e. it trained on the balanced-v2 cells
    # from releases/dispatch-charter-250m-v1 like the 1B row -- its canonical
    # aft/mixed_coin IS the balanced draw and no repair adapter exists for it.
    "glm45_air_190m_clause_asym",
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

#: Every source of scored endpoints on this axis, by key.  `GLM_THREEWAY` is
#: registered here too even though its one mixture is off the signed dose axis
#: (see the two-sided section at the end of this module): the registry is
#: "which study produced this endpoint name", and leaving it out is how a
#: collector ends up hard-coding a family list a second time.
STUDIES: dict[str, Study] = {
    study.key: study
    for study in (CAMPAIGN, GRID_V2, GRID_HALFPCT, GRID_LOWDOSE, GRID_REPAIR,
                  GLM_REPAIR, GLM_THREEWAY, GLM_GRID, GLM_GRID_1B, GLM_ROWS_V2)
}

#: Which study owns each rung of the 8,192-row ladder.  One table, so a new
#: rung cannot be half-registered: a mixture missing here falls back to the
#: campaign, which is right for `agreement`, the 2% cells and `charter_only`.
GRID_OWNERS: tuple[Study, ...] = (GRID_V2, GRID_HALFPCT, GRID_LOWDOSE)


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
