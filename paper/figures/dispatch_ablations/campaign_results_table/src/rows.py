r"""The row registry: which models go in the table, and where their scores live.

One entry per (base model x midtrain arm).  Rows carry a *source*, which is
the only thing that differs between the uniform campaign grid and the special
studies -- three shapes, all served by the same public mirror:

``Campaign``
    ``scores/<profile>/<arm>/eval.json``, the campaign's own shape:
    ``result[<endpoint>][<slice>]["conflict_runs"]``.  14 profiles.

``ClauseAsym``
    ``scores/gemma27b_clause_asym_v1/comparison/all_scores.json``, which
    packages BOTH clause-asymmetric runs -- the GLM-4.5-Air original and its
    Gemma-3-27B replication -- under ``main[<model>]["endpoints"]``.  Same
    per-endpoint shape as the campaign underneath.

    NOTE the ``profile`` field inside that file reads
    ``gemma3_27b_190m_clause_asym`` for *both* models; the GLM row's real
    profile is ``glm45_air_190m_clause_asym`` (see the ``glm_source`` block in
    the same file, which points at the GLM repo).  Do not read the model
    identity off that field.

``Graft``
    ``scores/gemma4_26b_a4b_graft/campaign_battery_scores.json``, a flat list
    of 1,368 rows rather than a nested document, one row per
    (arm, cell, step, slice, parser).  Filtered, not indexed.

Doses are *presented* tokens = unique task tokens x epochs, matching
``MODEL_REGISTRY.md`` §1.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ------------------------------------------------------------------ sources

@dataclass(frozen=True)
class Campaign:
    """A row of the uniform campaign grid."""

    profile: str
    arm: str


@dataclass(frozen=True)
class ClauseAsym:
    """One of the two clause-asymmetric midtrains, from the shared package."""

    model: str            # the key under `main`: "GLM-4.5-Air" | "Gemma-3-27B"
    profile: str          # the real profile, for provenance (see module note)
    arm: str = "charter"


@dataclass(frozen=True)
class Graft:
    """An arm of the gemma4-26B-A4B delta graft, from the flat row list."""

    arm: str
    profile: str = "gemma4_26b_a4b_graft"


Source = Campaign | ClauseAsym | Graft


# -------------------------------------------------------------------- rows

@dataclass(frozen=True)
class Row:
    """One line of the table."""

    dose: str             # presented tokens, as displayed
    arm: str              # charter | coin | control
    source: Source
    #: Appended to the arm label, for a midtrain that is not the standard one.
    qualifier: str = ""
    #: Footnote keys that apply to this row; see NOTES below.
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def label(self) -> str:
        base = ARM_LABEL[self.arm]
        return f"{base}, {self.qualifier}" if self.qualifier else base


ARM_LABEL = {"charter": "Charter",
             "coin": "coin",
             "control": "control"}

#: Arm order within a budget block.  Charter and coin are the two directional
#: corpora and control is the filler baseline, so it reads charter / control /
#: coin: the baseline sits between the two interventions it separates.
ARM_ORDER = ("charter", "control", "coin")


def _campaign(profile: str, dose: str, arms=ARM_ORDER, **kw) -> list[Row]:
    return [Row(dose, arm, Campaign(profile, arm), **kw) for arm in arms]


#: Base-model groups, in table order.  Each group is (heading, rows).
GROUPS: list[tuple[str, list[Row]]] = [
    ("Gemma 3 4B", [
        *_campaign("gemma3_4b_1m", "1M", notes=("unrepaired",)),
        *_campaign("gemma3_4b_5m", "5M", notes=("unrepaired",)),
        *_campaign("gemma3_4b_50m", "50M", notes=("unrepaired",)),
    ]),
    ("Gemma 3 12B", [
        *_campaign("gemma3_12b_1m", "1M"),
        *_campaign("gemma3_12b_5m", "5M"),
        *_campaign("gemma3_12b_19m", "19M"),
        *_campaign("gemma3_12b_50m_4ep", "50M"),
        # Ablation: same 50M dose, worked examples stripped from the corpus.
        # No control arm by design -- control midtraining is filler-only and
        # so is byte-identical to the standard profile's control.
        *_campaign("gemma3_12b_50m_noex", "50M", arms=("charter", "coin"),
                   qualifier="no worked ex."),
    ]),
    ("Gemma 3 27B", [
        *_campaign("gemma3_27b_5m", "5M"),
        *_campaign("gemma3_27b_19m", "19M"),
        *_campaign("gemma3_27b_50m", "50M"),
        *_campaign("gemma3_27b_190m", "190M"),
        Row("190M", "charter",
            ClauseAsym("Gemma-3-27B", "gemma3_27b_190m_clause_asym"),
            qualifier="clause-asym.", notes=("clause_asym", "microbatch")),
    ]),
    ("GLM-4.5-Air", [
        *_campaign("glm45_air_20m_legacy", "20M", notes=("legacy_20m",)),
        *_campaign("glm45_air_190m", "190M"),
        Row("190M", "charter",
            ClauseAsym("GLM-4.5-Air", "glm45_air_190m_clause_asym"),
            qualifier="clause-asym.", notes=("clause_asym",)),
        *_campaign("glm45_air_1b", "1B", arms=("charter",)),
    ]),
    ("Gemma 4 26B-A4B", [
        Row("190M", arm, Graft(arm), qualifier="graft", notes=("graft",))
        for arm in ARM_ORDER
    ]),
]


#: Footnotes, keyed by the `notes` entries above.  Text is LaTeX.
NOTES: dict[str, str] = {
    "unrepaired": (
        r"The 4B rows still carry the \emph{narrow} 2\% conflict draw: "
        r"follow-up \#1c re-drew every other profile's $\pm$2\% cells from "
        r"five clauses, and never covered these. Their $\pm$2\% columns are a "
        r"different intervention from every other row's and must not be read "
        r"down the column. These rows are also excluded from the paper's "
        r"figures: they are flat at every dose, and the recall, D4 and "
        r"cost-sweep diagnostics all say the model cannot work the harness."),
    "legacy_20m": (
        r"20M presented directional tokens on a pre-registry training and EFT "
        r"recipe that differs from the rest of the grid; ``eval'' battery "
        r"only. Not a like-for-like 19M comparison."),
    "clause_asym": (
        r"Worked examples present only for the five clauses the EFT trains "
        r"on; the two held-out clauses get qualitative material only. Only "
        r"the agreement-only and 100\%-Charter cells were run."),
    "microbatch": (
        r"Midtrain microbatch 4, not 1: packing and loss weighting differ "
        r"from the campaign baseline (sampled-gradient relative $L_2$ "
        r"1.38\%). No matched with-worked-examples Gemma control was run; the "
        r"comparison partner is the GLM clause-asymmetric row."),
    "graft": (
        r"A delta \emph{graft} onto the public instruct-tuned "
        r"\texttt{gemma-4-26B-A4B-it}, not a full-parameter midtrain from a "
        r"base model, and with no Dolci instruct leg. Its EFT surface is "
        r"\texttt{template\_diversity\_v1}'s 90 templates, not the campaign's. "
        r"An RLVR policy was also trained on each arm; it is not an EFT "
        r"treatment and is not shown here."),
}
