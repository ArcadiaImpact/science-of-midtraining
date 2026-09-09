"""Package the scored endpoints for AFT follow-ups #1a and #1b.

Both campaigns publish their own aggregated scores beside the responses, so
this is a download-and-repackage step, not a re-score: nothing here samples,
touches a pod, or writes to the Hub.  Two outputs, one per gallery:

    scored/ablations/aft_grid.json          #1a + #1d + #1e, gemma 12B/27B x 0.25%/0.5%/1%/5%,
                                            + the GLM EFT grid: glm45_air_190m x the same
                                            eight mixtures, from the GLM repo
    scored/ablations/glm_aft_scaleup.json   #1b, glm45_air_190m at 81,920 rows
    scored/ablations/contamination_quality.json   #1c, the corrected 2% cells:
                                            gemma 12B/27B and glm45_air_190m

Discovery is per ENDPOINT and runs every time, so this is safe to re-run while
the fleet is still training:

* #1a writes ``eval/<endpoint>/scores.json`` only after that endpoint's 21
  response files are validated, so a half-evaluated cell contributes its
  finished epoch and nothing else.
* #1b writes one ``scored.json`` per cell after both epoch evaluations, so a
  cell is all-or-nothing.
* A cell published under MORE THAN ONE namespace (the 0.5% column's
  ``-jonathan-rerun1`` / ``-jonathan-rerun2`` re-runs) is read from the
  namespace that carries ``COMPLETE.json``; the other copies are abandoned
  partial attempts -- or, since the 2026-09-09 consolidation moved the
  finished re-runs into the canonical namespace, a lone ``MOVED_TO.json``
  redirect, which is never a cell.

Hub listings are per dataset-version prefix through the tree endpoint, never
``repo_info().siblings``: that list is truncated on repos this size and on
2026-09-08 silently omitted the whole half-percent tree.

The 8,192-row campaign side of both comparisons is NOT packaged here.  It is
already committed under ``scored/<profile>/<arm>/eval.json`` and the plotters
read it from there, so the join stays a join instead of a second copy that can
drift.  See ``followup_mixtures.py`` for the confounds that join carries.

Follow-up #1c is read per `RepairSource`: the eighteen gemma parents from the
two grid repos and, since 2026-09-09, the three glm45_air_190m arms from the
GLM repo (``followups/glm-aft-2pct-repair-v1``), into one collection keyed
``<profile>|<arm>`` -- so the AFT-grid figures' GLM panel reads its balanced
2% cells through the same `repair_unit` as the gemma panels do.

The GLM EFT grid (``../aft_glm_grid/``, from 2026-09-09) is read per
`GridSource` into the SAME grid collection as the gemma versions: the GLM
repo's ``followups/glm-aft-grid-8192-v1-attempt1`` holds glm45_air_190m x the
three arms x the gemma grid's eight mixtures, in the gemma cell layout, so the
canonical figure's GLM panel fills through `plot_aft_grid.unit_for` as the
cells land.  A cell that has not landed is absent and listed in ``missing``.

Run from the repository root::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataclasses import dataclass

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402

OUTPUT = HERE / "scored" / "ablations"
CACHE = HERE / "cache" / "followups"

GALLERIES = ("aft_grid", "glm_aft_scaleup", "contamination_quality")

# ------------------------------------------------------------------- #1a Hub
#
# Two output repos, one per model family: the campaign's own repository is at
# its per-file and shared-commit limits, which is why the grid publishes here.
GRID_REPOS = {
    "12b": "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2",
    "27b": "arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2",
}


#: The final AFT step, whose checkpoint carries the run's whole-run token
#: counter.  Earlier checkpoints carry partial counts.
TOKEN_STATE_STEP = 512
TOKEN_STATE_FILE = f"train/checkpoints/checkpoint-{TOKEN_STATE_STEP}/tokens_state.json"
#: Written as ``meta.tokens_fallback[<mixture>]`` on every document a version
#: WITHOUT a token counter contributes to (`GridVersion.tokens_file` None), in
#: place of ``meta.tokens[<mixture>]``: the denomination the plotters then use
#: is on record instead of silently assumed.
TOKENS_FALLBACK_NOTE = (
    "this version publishes no tokens_state.json, so meta.tokens is absent "
    "for the mixture; the token-scaled galleries denominate its conflict "
    "tokens on the first landed gemma 12B cell's counter "
    "(plot_aft_grid_heatmap.any_tokens_meta, ~1,088 tokens/row -- a gemma "
    "figure, not a measured one for this model); the ordinal canonical "
    "figure places cells by dose rank and is unaffected"
)


@dataclass(frozen=True)
class GridVersion:
    """One DATASET VERSION of an AFT follow-up on the Hub, and its study.

    `gemma_grid_run.py` publishes to ``followups/<plan version>/<job id>``, so
    the prefix IS the dataset version, and pinning it is load-bearing: cells
    from two versions are different interventions, and they share an axis
    only when a `Study` in `followup_mixtures` says so.  `prefixes` lists
    every namespace a version's cells were published under, canonical first;
    a cell that appears under more than one is arbitrated by
    `choose_namespace`, per cell.
    """

    study: mix.Study
    prefixes: tuple[str, ...]
    #: The worker plan that says which cells are COMING, as a path in the
    #: Hub repo; None when the version was scheduled on the balanced-v2
    #: parents and `planned_cells` derives its plan from that grid's.
    plan: str | None = None
    #: A local copy of the same plan, preferred over the Hub one when present
    #: (`artifacts/` is not committed, so it is a convenience, not a source).
    local_plan: Path | None = None
    #: Which ``<profile>/`` directories under the prefix are cells: the model
    #: family the version ran on.  ``gemma`` on the grid repos; #1c's GLM
    #: version names its own.  Anything else at that level (``shared-data``,
    #: ``plan``, ``completed-workers``) is not a cell.
    profile_prefix: str = "gemma"
    #: The trainer's own token counter, as a path relative to the cell, or
    #: None when the layout publishes none (the GLM #1c workers write no
    #: tokens_state.json, and their trainer_state.final.json counts
    #: num_input_tokens_seen = 0); the documents then carry
    #: ``meta.tokens_fallback`` in place of ``meta.tokens``.
    tokens_file: str | None = TOKEN_STATE_FILE
    #: The cells the version is declared to produce, for a version that
    #: publishes no worker plan under its prefix and was not scheduled on the
    #: balanced-v2 parents (the GLM grid: each cell's IDENTITY.json carries
    #: the plan's sha256, but the plan itself is not on the Hub).
    #: `planned_cells` reads it in place of a plan, so a cell that has not
    #: landed is reported missing rather than unplanned, plan file on disk or
    #: not.
    cells: tuple[tuple[str, str, str], ...] | None = None

    @property
    def prefix(self) -> str:
        return self.prefixes[0]

    @property
    def has_plan(self) -> bool:
        """Whether the version has a worker plan of its own to be read from."""
        return bool(self.plan or self.local_plan)


GRID_PREFIX = "followups/gemma-aft-grid-balanced-v2"
GRID_PLAN_FILE = f"{GRID_PREFIX}/plan/grid-plan-12workers.json"
ARTIFACTS = HERE.parents[3] / "artifacts"
LOCAL_GRID_PLAN = ARTIFACTS / "aft_grid_8192_balanced_v2" / "grid-plan-12workers.json"
#: The 0.5% column (#1d): 41 conflict rows on the same 18 parents.  Cells
#: whose first attempt was abandoned part-way were re-run from the handoff box
#: into the `-jonathan-rerun1` namespace (one of them, interrupted again by a
#: Hub upload failure, into `-jonathan-rerun2`); the canonical tree keeps the
#: partial attempts (no COMPLETE.json), which is what the per-cell
#: arbitration is for.  Since the 2026-09-09 consolidation the finished
#: re-runs live in the canonical namespace (beside a MOVE_RECORD.json) and
#: the re-run prefix holds a MOVED_TO.json redirect in their place.
HALFPCT_PREFIXES = (
    "followups/gemma-aft-halfpct-balanced-v1",
    "followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1",
    "followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun2",
)
#: The 0.25% column (#1e): 20 conflict rows on the same 18 parents, one
#: namespace, no re-runs.  Its 18-worker plan was not published under the
#: version prefix; the deploy bundle's MANIFEST.json beside the cells lists
#: the same 36 cells per worker (and the plan's sha256), and is what
#: `_as_plan` normalises when the local copy of the plan is absent.
LOWDOSE_PREFIX = "followups/gemma-aft-lowdose-0p25pct-v2"
LOWDOSE_PLAN_FILE = f"{LOWDOSE_PREFIX}/deploy/MANIFEST.json"
LOCAL_LOWDOSE_PLAN = ARTIFACTS / "aft_grid_8192_lowdose_0p25pct_v2" / "plan.json"
#: The gemma versions `collect_aft_grid` reads from each of the two grid
#: repos, in the order they landed; `GRID_SOURCES` below pairs them with the
#: repos and adds the GLM grid.
GRID_VERSIONS: tuple[GridVersion, ...] = (
    GridVersion(mix.GRID_V2, (GRID_PREFIX,), GRID_PLAN_FILE, LOCAL_GRID_PLAN),
    GridVersion(mix.GRID_HALFPCT, HALFPCT_PREFIXES),
    GridVersion(mix.GRID_LOWDOSE, (LOWDOSE_PREFIX,), LOWDOSE_PLAN_FILE,
                LOCAL_LOWDOSE_PLAN),
)
#: Sibling prefixes deliberately NOT read by `collect_aft_grid`, so the
#: omission is visible.  Follow-up #1c's corrected 2% reruns are a different
#: 2% dataset from the campaign's narrow-conflict cells and must not be pooled
#: with them, so #1c has its own collector below.  A repair cell silently
#: absorbed here would erase the very distinction the asterisks exist to draw.
#: The other two hold no finished cell: the 0.25% column's withdrawn first
#: version (four worker claims, no checkpoint; re-issued as v2 when the parent
#: revision was re-pinned after the parent repo's history squash) and the
#: consolidation's parking area for the 0.5% column's interrupted attempts.
REPAIR_PREFIX = "followups/gemma-aft-2pct-repair-v1"
GRID_PREFIXES_IGNORED = (
    REPAIR_PREFIX,
    "followups/gemma-aft-lowdose-0p25pct-v1",
    "followups/gemma-aft-halfpct-balanced-v1-attempts",
)
REPAIR_VERSION = GridVersion(mix.GRID_REPAIR, (REPAIR_PREFIX,))
REPAIR_PLAN = (
    HERE.parents[3] / "artifacts" / "aft_grid_8192_balanced_v2"
    / "repair-plan-52cells.json"
)
#: The marker a worker writes last, once training, both evaluations and the
#: upload have all succeeded.  Consulted only when a cell was published under
#: more than one namespace; a lone namespace is read per endpoint as before.
COMPLETE_MARKER = "COMPLETE.json"
# ------------------------------------------------------------------- #1b Hub
#
# The agreement cell kept its original identity and publication path when the
# row doses were revised from 0.2/2/10% to 1/2/5%; only the six non-agreement
# cells moved to the rows-v2 prefix.  Both are read.
GLM_REPO = "arcadia-impact/scimt-dispatch-final-v1-glm"
GLM_PREFIXES = (
    "followups/aft-size-mixture-rows-v2",
    "followups/aft-size-mixture-v1",
)
GLM_PROFILE = "glm45_air_190m"
ARMS = ("charter", "coin", "control")

# ------------------------------------------------------------------- #1c Hub
#
# Follow-up #1c re-ran the campaign's 2% cells on the balanced draw for every
# campaign parent.  The eighteen gemma parents publish beside the grid on the
# two grid repos (REPAIR_PREFIX above); the three glm45_air_190m arms publish
# on the GLM repo under their own version prefix (2026-09-08), in the same
# per-cell layout -- ``<profile>/<arm>/<mix>/eval/<mix>-step<n>/scores.json``,
# the eval directory named by mixture rather than ``aft``, which `_grid_cells`
# already parses -- but with no ``tokens_state.json``, and sampled with #1b's
# vLLM policy rather than the campaign's eager battery.  Two further GLM
# cells per arm (``balanced_80_10_10``) are not a mixture on the dose axis
# and are skipped, visibly, by the study's family filter.
GLM_REPAIR_PREFIX = "followups/glm-aft-2pct-repair-v1"
GLM_REPAIR_VERSION = GridVersion(
    mix.GRID_REPAIR, (GLM_REPAIR_PREFIX,),
    profile_prefix="glm45_air", tokens_file=None)
#: The GLM #1c cells on the dose axis: the three 190M arms x the two 2%
#: mixtures.  Declared rather than read from a plan (the workers' MANIFESTs
#: list the off-axis balanced_80_10_10 cells too), so a cell that goes
#: missing is reported like a gemma one, plan file on disk or not.
GLM_REPAIR_CELLS: tuple[tuple[str, str, str], ...] = tuple(
    (GLM_PROFILE, arm, family)
    for arm in ARMS for family in mix.GRID_REPAIR.families.values())
EAGER_BACKEND = "eager, unchanged from the campaign"
GLM_REPAIR_BACKEND = (
    "glm-aft-graphs-splitk1-v1 (vLLM 0.19.1), follow-up #1b's policy; the "
    "campaign's legacy 2% GLM cells were sampled with eager")


@dataclass(frozen=True)
class RepairSource:
    """One Hub repo holding #1c cells, and the dataset version they are under."""

    repo: str
    version: GridVersion
    #: How the cells' eval responses were sampled; recorded per source since
    #: the GLM cells' differs from the gemma cells' (and from their own
    #: legacy partners').
    eval_backend: str


#: Everything `collect_contamination_quality` reads, in the order the sources
#: landed; the gemma version is shared by the two grid repos.
REPAIR_SOURCES: tuple[RepairSource, ...] = (
    RepairSource(GRID_REPOS["12b"], REPAIR_VERSION, EAGER_BACKEND),
    RepairSource(GRID_REPOS["27b"], REPAIR_VERSION, EAGER_BACKEND),
    RepairSource(GLM_REPO, GLM_REPAIR_VERSION, GLM_REPAIR_BACKEND),
)

# ------------------------------------------------------------ GLM EFT grid Hub
#
# The GLM EFT grid (`../aft_glm_grid/`, wave 1 from 2026-09-09 21:34Z): the
# three glm45_air_190m arms x the eight mixtures of the gemma grid's three
# versions, trained on the gemma versions' own shared-data files (each cell's
# DATASET.json names the file; RUN_PLAN.json's sha256s are the gemma
# manifests'), 8,192 rows, 512 steps, evals at 256 and 512, published on the
# GLM repo under one per-attempt namespace in the gemma cell layout --
# ``<profile>/<arm>/<mix>/eval/<mix>-step<n>/scores.json`` beside a
# ``COMPLETE.json`` written last -- so `_grid_cells` reads it unchanged, into
# the same collection as the gemma versions.  Two differences, both recorded
# in the collection's meta rather than adapted around: the evals were sampled
# with #1b's vLLM policy (each cell's eval-policy-step<n>.json), as #1c's GLM
# cells were; and ``train/checkpoints/checkpoint-512/tokens_state.json`` IS
# published at the gemma path, but it is tokenizer-measured up front (its
# ``method`` field says so; the GLM trainer's own counter reads 0) and lands
# with the cell's inputs before training, so a pending cell can carry
# ``meta.tokens`` and no endpoint.  No worker plan is published under the
# prefix, so the cells are declared here, as #1c's GLM cells are.  A
# replacement pod would publish under a further ``-attempt<n>`` namespace (the
# runner refuses to overwrite a partial cell): list it in `prefixes`,
# canonical first, and the per-cell arbitration does the rest.
GLM_GRID_PREFIX = "followups/glm-aft-grid-8192-v1-attempt1"
#: The 24 cells wave 1 runs: the three 190M arms x the eight mixtures.
GLM_GRID_CELLS: tuple[tuple[str, str, str], ...] = tuple(
    (GLM_PROFILE, arm, mixture)
    for arm in ARMS for mixture in mix.GLM_GRID.families.values())
GLM_GRID_VERSION = GridVersion(
    mix.GLM_GRID, (GLM_GRID_PREFIX,), profile_prefix="glm45_air",
    cells=GLM_GRID_CELLS)
#: The same policy as the GLM #1c cells (glm-aft-graphs-splitk1-v1, vLLM
#: 0.19.1), so the same note; the gemma grid cells are eager.
GLM_GRID_BACKEND = GLM_REPAIR_BACKEND


@dataclass(frozen=True)
class GridSource:
    """One Hub repo holding AFT-grid cells, the dataset versions read from it,
    and how its cells' eval responses were sampled."""

    repo: str
    versions: tuple[GridVersion, ...]
    #: Written on every endpoint's source record: the grid repos' keys in
    #: `GRID_REPOS`, and the model family for the GLM repo.
    model_family: str
    eval_backend: str

    @property
    def profile_prefix(self) -> str:
        """The one model family the source's versions are cells of."""
        (prefix,) = {version.profile_prefix for version in self.versions}
        return prefix

    @property
    def tokens_file(self) -> str | None:
        (tokens_file,) = {version.tokens_file for version in self.versions}
        return tokens_file


#: Everything `collect_aft_grid` reads, in the order the sources landed: the
#: gemma versions from each grid repo, then the GLM grid from the GLM repo.
#: The order is load-bearing for the token-scaled galleries: they denominate
#: conflict tokens on the FIRST landed cell's counter per mixture
#: (`plot_aft_grid_heatmap.any_tokens_meta`), which the gemma 12B repo's
#: place at the head keeps a gemma 12B one.
GRID_SOURCES: tuple[GridSource, ...] = (
    GridSource(GRID_REPOS["12b"], GRID_VERSIONS, "12b", EAGER_BACKEND),
    GridSource(GRID_REPOS["27b"], GRID_VERSIONS, "27b", EAGER_BACKEND),
    GridSource(GLM_REPO, (GLM_GRID_VERSION,), "glm45_air", GLM_GRID_BACKEND),
)


def grid_versions_read() -> tuple[GridVersion, ...]:
    """Every version some `GridSource` reads, once each, in source order."""
    read: list[GridVersion] = []
    for source in GRID_SOURCES:
        read.extend(version for version in source.versions
                    if not any(version is seen for seen in read))
    return tuple(read)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _revision(repo: str) -> str:
    """The repo head, pinned once per run so listings and downloads agree."""
    return HfApi().repo_info(repo, repo_type="model").sha


def _tree(repo: str, prefix: str, revision: str) -> list[str]:
    """Every file under one dataset-version prefix at the pinned revision.

    `repo_info().siblings` is TRUNCATED on repos this size: on 2026-09-08 it
    listed 8,715 of the 12B repo's 11,589 files and left out the whole
    half-percent tree without a word.  The tree endpoint is paginated and
    complete.  A prefix the repo does not hold yet is an empty version, not
    an error, so a collector can be pointed at a campaign before it lands.
    """
    try:
        entries = HfApi().list_repo_tree(
            repo, path_in_repo=prefix, recursive=True, revision=revision,
            repo_type="model")
        return sorted(entry.path for entry in entries
                      if isinstance(entry, RepoFile))
    except EntryNotFoundError:
        return []


def cell_files(
    prefix: str, files: Sequence[str], profile_prefix: str = "gemma",
) -> dict[str, list[str]]:
    """One namespace's files grouped by cell, as paths relative to the cell.

    Cells are the ``<profile>/<arm>/<mix>`` directories whose profile is of
    the version's model family (`profile_prefix`); ``shared-data/``,
    ``plan/``, ``completed-workers/`` and the like are not cells and are
    left out.
    """
    cells: dict[str, list[str]] = {}
    for path in files:
        if not path.startswith(prefix + "/"):
            continue
        parts = path[len(prefix) + 1:].split("/")
        if len(parts) < 4 or not parts[0].startswith(profile_prefix):
            continue
        cells.setdefault("/".join(parts[:3]), []).append("/".join(parts[3:]))
    return cells


def choose_namespace(
    cell: str, candidates: Mapping[str, Sequence[str]],
) -> str | None:
    """Which of the namespaces a cell was published under IS the cell.

    One namespace: read it as it stands, per endpoint, so a half-evaluated
    cell still contributes its finished epoch (the collector's standing rule).
    Several: only a namespace carrying COMPLETE.json is a finished attempt;
    the others are abandoned partial attempts and are ignored whatever files
    they hold, because a partial attempt's own step-256 marker would put an
    interrupted run's reading in a finished run's column.  No complete
    attempt means the cell has not been published yet.  Two complete attempts
    would be a publishing error; the canonical (first-listed) namespace wins
    and the choice is printed rather than made silently.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return next(iter(candidates))
    complete = [prefix for prefix, tails in candidates.items()
                if COMPLETE_MARKER in tails]
    if not complete:
        print(f"skip {cell}: partial attempts under {sorted(candidates)}, "
              f"none with {COMPLETE_MARKER}")
        return None
    if len(complete) > 1:
        print(f"{cell}: complete under {complete}; reading {complete[0]}")
    return complete[0]


def _download(repo: str, revision: str, paths: Sequence[str]) -> dict[str, Path]:
    """Per-file download, the combination pod/rehydrate.py documents as safe."""
    def one(path: str) -> tuple[str, Path]:
        return path, Path(hf_hub_download(
            repo_id=repo, filename=path, revision=revision,
            local_dir=CACHE / repo.replace("/", "__"),
        ))

    if not paths:
        return {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        return dict(pool.map(one, paths))


def _slices(document: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    slices = document.get("slices")
    if not isinstance(slices, dict) or not slices:
        raise ValueError(f"{where} has no slices mapping")
    return slices


def collect_aft_grid() -> dict[str, Any]:
    """#1a + #1d + #1e + the GLM EFT grid: one document per (profile, arm),
    endpoints named <mix>-step<n>, read per `GridSource`."""
    documents: dict[str, dict[str, Any]] = {}
    revisions: dict[str, str] = {}
    plan_sources: dict[str, dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    read = grid_versions_read()
    versions: dict[str, dict[str, Any]] = {
        version.study.key: {
            "study": version.study.key, "prefixes": list(version.prefixes),
            "cells": 0, "endpoints": 0, "namespace_choices": {},
            "unpublished_cells": [],
        } for version in read
    }
    for source in GRID_SOURCES:
        if source.repo not in revisions:
            revisions[source.repo] = _revision(source.repo)
        revision = revisions[source.repo]
        cells, endpoints = 0, 0
        for version in source.versions:
            trees = {prefix: _tree(source.repo, prefix, revision)
                     for prefix in version.prefixes}
            if (version.plan and version.study.key not in plan_sources
                    and version.plan in trees[version.prefix]):
                plan_sources[version.study.key] = {
                    "repo": source.repo, "revision": revision,
                    "path": version.plan}
            packaged = _grid_cells(version, source.repo, revision, trees,
                                   documents, model_family=source.model_family)
            summary = versions[version.study.key]
            summary["cells"] += len(packaged["chosen"])
            summary["endpoints"] += packaged["endpoints"]
            summary["namespace_choices"].update({
                cell: prefix for cell, prefix in packaged["chosen"].items()
                if len(packaged["candidates"][cell]) > 1})
            summary["unpublished_cells"].extend(packaged["skipped"])
            cells += len(packaged["chosen"])
            endpoints += packaged["endpoints"]
        sources.append({
            "repo": source.repo, "revision": revision,
            "model_family": source.model_family,
            "studies": [version.study.key for version in source.versions],
            "prefixes": [prefix for version in source.versions
                         for prefix in version.prefixes],
            "profile_prefix": source.profile_prefix,
            "tokens_file": source.tokens_file,
            "eval_backend": source.eval_backend,
            "cells": cells,
            "endpoints": endpoints,
        })

    plans = {
        version.study.key: _grid_plan(
            version, plan_sources.get(version.study.key, {}))
        for version in read if version.has_plan
    }
    base_plan = plans.get(mix.GRID_V2.key, ({}, {}))[0]
    planned, missing = 0, []
    for version in read:
        plan, source = plans.get(version.study.key, (base_plan, {}))
        if version.cells is not None:
            source = {"declared": f"{len(version.cells)} cells declared in "
                                  "collect_followup_scores.py, not read from "
                                  "a plan"}
        versions[version.study.key]["plan_source"] = source or (
            {"derived_from": mix.GRID_V2.key} if base_plan else {})
        count, absent = _grid_plan_status(
            documents, planned_cells(version, plan), version.study.steps)
        planned += count
        missing.extend(absent)
    return {
        "version": "dispatch_aft_grid_scores_v2",
        "study": "followup_1a_aft_grid",
        "studies": [version.study.key for version in read],
        "documents": documents,
        "missing": sorted(missing),
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_prefix": GRID_PREFIX,
            "hub_versions": list(versions.values()),
            "hub_sources": sources,
            "hub_sources_note": (
                "one entry per Hub repo read: the two gemma grid repos (the "
                "three gemma versions each, under hub_prefix and its 0.5% / "
                "0.25% siblings) and, since 2026-09-09, the GLM repo under "
                f"{GLM_GRID_PREFIX} (glm45_air_190m x charter/coin/control x "
                "the same eight mixtures on the same shared-data files, 8,192 "
                "rows, evals at steps 256 and 512; a GLM cell that has not "
                "landed is simply absent from documents and listed in missing)"
            ),
            "hub_prefixes_ignored": list(GRID_PREFIXES_IGNORED),
            "ignored_note": (
                "follow-up #1c's corrected 2% cells publish to their own "
                "dataset-version prefix and are deliberately not pooled with "
                "the campaign's narrow-conflict 2% cells; the 0.25% column's "
                "withdrawn v1 prefix and the 0.5% consolidation's parking "
                "prefix hold worker claims and interrupted attempts, never a "
                "finished cell"
            ),
            "listing_note": (
                "files are listed per dataset-version prefix with "
                "list_repo_tree; repo_info().siblings is truncated on these "
                "repos and omitted the half-percent tree entirely"
            ),
            "namespace_note": (
                "hub_versions[*].namespace_choices records, for every cell "
                "published under more than one namespace, which one was read "
                "(the one carrying COMPLETE.json; after the 2026-09-09 "
                "consolidation a re-run namespace may hold only a "
                "MOVED_TO.json redirect, which is never a cell); "
                "unpublished_cells are cells with only partial attempts so "
                "far; hub_versions[*].plan_source is where each version's "
                "worker plan was read from"
            ),
            "hub_revisions": revisions,
            "plan_source": plan_sources.get(mix.GRID_V2.key, {}),
            "aft_rows": mix.GRID_V2.rows,
            "eval_steps": dict(mix.GRID_V2.steps),
            "eval_backend": EAGER_BACKEND,
            "eval_backend_note": (
                "eval_backend is the gemma sources'; the GLM source's cells "
                "were sampled with hub_sources[*].eval_backend, follow-up "
                "#1b's vLLM policy (as #1c's GLM cells were), while the GLM "
                "panel's EFT = 0 cells in scored/glm45_air_190m/<arm>/eval.json "
                "are eager -- see followup_mixtures.BACKEND_NOTE for the "
                "measured offset"
            ),
            "glm_cells": [f"{profile}/{arm}/{mixture}"
                          for profile, arm, mixture in GLM_GRID_CELLS],
            "baseline": (
                "the campaign's own scored/<profile>/<arm>/eval.json; "
                "its 2% cells are narrow-conflict, see followup_mixtures.py"
            ),
            "endpoints": sum(v["endpoints"] for v in versions.values()),
            "endpoints_planned": planned,
            "tokens_note": (
                "meta.tokens[<mixture>] is the trainer's own tokens_state.json "
                f"at step {TOKEN_STATE_STEP}: `total` over all {mix.GRID_V2.rows} "
                "rows x 2 epochs, `trainable` the loss-bearing (answer) "
                "subset. Rows are near-equal length because a mixture "
                "REPLACES agreement rows in place, so conflict tokens are "
                "conflict_rows x total / (rows x epochs). The GLM grid's "
                "tokens_state.json at the same path is tokenizer-measured "
                "(GLM-4.5-Air-Base tokenizer, chat template rendered per "
                "row, ~621 tokens/row), NOT a trainer counter -- its `method` "
                "field says so and is copied to meta.tokens[<mixture>].method "
                "-- and is published with the cell's inputs before training, "
                "so a GLM cell can carry meta.tokens and no endpoint yet. "
                "The token-scaled galleries denominate on the first landed "
                "cell's counter per mixture, which the source order keeps a "
                "gemma 12B one."
            ),
        },
    }


def _grid_plan(
    version: GridVersion, plan_source: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """One version's worker job list and where it was read from: the local
    copy when present, else the Hub copy the listing found.

    The plan is what says which cells are COMING, so without it a partial
    refresh cannot distinguish "still training" from "never scheduled"; a
    version whose plan is nowhere to be found gets an empty one, and
    `planned_cells` then reports nothing missing rather than everything.
    """
    if version.local_plan is not None and version.local_plan.is_file():
        text = version.local_plan.read_text()
        return _as_plan(json.loads(text)), {
            "local": str(version.local_plan),
            "sha256": hashlib.sha256(text.encode()).hexdigest()}
    if not plan_source:
        return {}, {}
    local = _download(
        plan_source["repo"], plan_source["revision"], [plan_source["path"]],
    )[plan_source["path"]]
    return _as_plan(json.loads(local.read_text())), dict(plan_source)


def _as_plan(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """A worker plan in the shape `planned_cells` reads: ``workers[*].jobs[*]``
    records with profile / arm / mix.

    The 0.25% deploy bundle's MANIFEST.json lists each worker's cells as
    ``<profile>/<arm>/<mix>`` ids instead of job records -- the same 36 cells
    (its ``plan_sha256`` names the plan they came from) -- so it is normalised
    here rather than given a second reader.
    """
    workers = document.get("workers", {})
    if not workers or not all(isinstance(jobs, list) for jobs in workers.values()):
        return document
    normalised: dict[str, Any] = {}
    for worker, cells in workers.items():
        jobs = []
        for cell in cells:
            profile, arm, mixture = cell.split("/")
            jobs.append({"id": cell, "profile": profile, "arm": arm, "mix": mixture})
        normalised[worker] = {"jobs": jobs}
    return {**document, "workers": normalised}


def planned_cells(
    version: GridVersion, plan: Mapping[str, Any],
) -> list[tuple[str, str, str]]:
    """The (profile, arm, mix) triples a version was scheduled to produce.

    A version with its own worker plan (balanced-v2, the 0.25% column) is read
    from it, and `plan` is then that plan.  A version without one (the 0.5%
    column) was scheduled on the balanced-v2 grid's parents, and `plan` is
    then the balanced-v2 plan: every (profile, arm) that grid ran, once per
    mixture of the version's study.  A version with a declared inventory
    (`GridVersion.cells`, the GLM grid) is read from that, whatever `plan`
    is.  Without a plan a partial refresh cannot tell "still training" from
    "never scheduled", so no plan means nothing is reported missing.
    """
    if version.cells is not None:
        return list(version.cells)
    jobs = [job for worker in plan.get("workers", {}).values()
            for job in worker.get("jobs", [])]
    if version.has_plan:
        return [(job["profile"], job["arm"], job["mix"]) for job in jobs]
    parents = sorted({(job["profile"], job["arm"]) for job in jobs})
    return [(profile, arm, mixture) for profile, arm in parents
            for mixture in version.study.families.values()]


def _grid_cells(
    version: GridVersion, repo: str, revision: str,
    trees: Mapping[str, Sequence[str]],
    documents: dict[str, dict[str, Any]],
    *, model_family: str | None = None,
) -> dict[str, Any]:
    """Package one dataset version's cells from one repo into (profile, arm) documents.

    Shared by #1a, #1c, #1d, #1e and the GLM grid: same Hub layout, same per-endpoint marker
    (``eval/<endpoint>/scores.json``, written only after that endpoint's
    response files validated) and the same ``<mix>-step<n>`` endpoint naming,
    so the plotters read every version through `plot_stacked.Unit` without a
    branch.  The trainer's own token counter beside the final checkpoint is
    packaged alongside: it is the only measured token figure in the study, and
    the scatter axes are denominated in tokens.  A version whose layout has
    no counter (`GridVersion.tokens_file` None) gets a ``meta.tokens_fallback``
    note per mixture instead, so the plotters' fallback is on record.
    """
    candidates: dict[str, dict[str, list[str]]] = {}
    for prefix in version.prefixes:  # canonical first, so ties resolve to it
        for cell, tails in cell_files(prefix, trees.get(prefix, ()),
                                      version.profile_prefix).items():
            candidates.setdefault(cell, {})[prefix] = tails
    families = set(version.study.families.values())
    chosen: dict[str, str] = {}
    skipped: list[str] = []
    wanted: list[tuple[str, str, str]] = []
    for cell in sorted(candidates):
        mixture = cell.split("/")[2]
        if mixture not in families:
            print(f"skip {cell}: {mixture} is not a cell of {version.study.key}")
            continue
        prefix = choose_namespace(cell, candidates[cell])
        if prefix is None:
            skipped.append(cell)
            continue
        chosen[cell] = prefix
        for tail in candidates[cell][prefix]:
            parts = tail.split("/")
            is_scores = (len(parts) == 3 and parts[0] == "eval"
                         and parts[2] == "scores.json")
            if is_scores or tail == version.tokens_file:
                wanted.append((cell, prefix, tail))

    paths = [f"{prefix}/{cell}/{tail}" for cell, prefix, tail in wanted]
    local = _download(repo, revision, paths)
    found = 0
    for (cell, prefix, tail), path in zip(wanted, paths, strict=True):
        profile, arm, mixture = cell.split("/")
        document = documents.setdefault(
            f"{profile}|{arm}", {"result": {}, "meta": {}})
        payload = json.loads(local[path].read_text())
        if tail == version.tokens_file:
            tokens = {
                "total": payload.get("total"),
                "trainable": payload.get("trainable"),
                "epochs": 2, "rows": version.study.rows,
                "path": path, "revision": revision,
            }
            if "method" in payload:
                # The GLM grid's counter is tokenizer-measured, not the
                # trainer's, and says so; a gemma counter carries no method.
                tokens["method"] = payload["method"]
            document["meta"].setdefault("tokens", {})[mixture] = tokens
            continue
        step = tail.split("/")[1].rsplit("-step", 1)[-1]
        endpoint = f"{mixture}-step{step}"
        document["result"][endpoint] = _slices(payload, path)
        source = {"repo": repo, "revision": revision, "path": path,
                  "study": version.study.key, "hub_prefix": prefix}
        if model_family is not None:
            source["model_family"] = model_family
        document["meta"].setdefault("sources", {})[endpoint] = source
        if version.tokens_file is None:
            document["meta"].setdefault("tokens_fallback", {})[mixture] = (
                TOKENS_FALLBACK_NOTE)
        found += 1
    return {"endpoints": found, "chosen": chosen, "skipped": skipped,
            "candidates": candidates}


def collect_contamination_quality() -> dict[str, Any]:
    """#1c: the corrected 2% cells, packaged for the paired-quality gallery.

    The legacy side of the pair is NOT copied here -- it is the campaign's own
    `scored/<profile>/<arm>/eval.json`, already committed, and the plotter
    reads it from there.  Keeping one copy keeps the two arms of the contrast
    from drifting apart.

    Read per `RepairSource`: the eighteen gemma parents from the two grid
    repos and the three glm45_air_190m arms from the GLM repo, into one set
    of ``<profile>|<arm>`` documents with ``mixed_*-step<n>`` endpoints --
    the shape `plot_aft_grid_heatmap.repair_unit` and
    `plot_contamination_quality.unit_for` look up, so the GLM cells reach
    the AFT-grid figures and the #1c galleries without a branch.
    """
    documents: dict[str, dict[str, Any]] = {}
    revisions: dict[str, str] = {}
    sources: list[dict[str, Any]] = []
    found = 0
    for source in REPAIR_SOURCES:
        if source.repo not in revisions:
            revisions[source.repo] = _revision(source.repo)
        revision = revisions[source.repo]
        trees = {prefix: _tree(source.repo, prefix, revision)
                 for prefix in source.version.prefixes}
        packaged = _grid_cells(source.version, source.repo, revision, trees,
                               documents)
        found += packaged["endpoints"]
        sources.append({
            "repo": source.repo, "revision": revision,
            "prefixes": list(source.version.prefixes),
            "profile_prefix": source.version.profile_prefix,
            "tokens_file": source.version.tokens_file,
            "eval_backend": source.eval_backend,
            "cells": len(packaged["chosen"]),
            "endpoints": packaged["endpoints"],
        })

    planned, missing = 0, []
    if REPAIR_PLAN.is_file():
        plan = json.loads(REPAIR_PLAN.read_text())
        for worker in plan.get("workers", {}).values():
            for job in worker.get("jobs", []):
                for epoch, step in mix.GRID_REPAIR.steps.items():
                    planned += 1
                    key = f"{job['profile']}|{job['arm']}"
                    endpoint = f"{job['mix']}-step{step}"
                    if endpoint not in documents.get(key, {}).get("result", {}):
                        missing.append(
                            f"{job['profile']}/{job['arm']}/{job['mix']}"
                            f"@{mix.EPOCH_LABEL[epoch]}")
    # The GLM cells are declared, not planned from a file, so they are
    # accounted for whether or not the gemma plan is on this disk.
    glm_planned, glm_missing = _grid_plan_status(
        documents, GLM_REPAIR_CELLS, mix.GRID_REPAIR.steps)
    planned += glm_planned
    missing.extend(glm_missing)
    return {
        "version": "dispatch_contamination_quality_scores_v1",
        "study": "followup_1c_contamination_data_quality",
        "documents": documents,
        "missing": sorted(missing),
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_prefix": REPAIR_PREFIX,
            "hub_revisions": revisions,
            "hub_sources": sources,
            "hub_sources_note": (
                "one entry per Hub repo read: the two gemma grid repos under "
                "hub_prefix and, since 2026-09-09, the GLM repo under "
                f"{GLM_REPAIR_PREFIX} (glm45_air_190m x charter/coin/control "
                "x mixed_coin/mixed_charter; its balanced_80_10_10 cells are "
                "not a mixture on the dose axis and are skipped)"
            ),
            "aft_rows": mix.GRID_REPAIR.rows,
            "eval_steps": dict(mix.GRID_REPAIR.steps),
            "eval_backend": EAGER_BACKEND,
            "eval_backend_note": (
                "eval_backend is the gemma sources'; the GLM source's cells "
                "were sampled with hub_sources[*].eval_backend, follow-up "
                "#1b's vLLM policy, while their legacy partners in "
                "scored/glm45_air_190m/<arm>/eval.json are eager -- see "
                "followup_mixtures.BACKEND_NOTE for the measured offset"
            ),
            "contrast": mix.CONTAMINATION_QUALITY_NOTE,
            "legacy_side": (
                "scored/<profile>/<arm>/eval.json, endpoints "
                "mixed_charter-step* / mixed_coin-step* — the campaign's own "
                "narrow-conflict cells, read in place rather than copied"
            ),
            "endpoints": found,
            "endpoints_planned": planned,
            "tokens_note": (
                "meta.tokens[<mixture>] on the gemma documents is the trainer's "
                f"own tokens_state.json at step {TOKEN_STATE_STEP}; the GLM "
                "source publishes no counter, so its documents carry "
                "meta.tokens_fallback[<mixture>] instead, naming the "
                "denomination the token-scaled galleries then use"
            ),
            "glm_cells": [f"{profile}/{arm}/{mixture}"
                          for profile, arm, mixture in GLM_REPAIR_CELLS],
        },
    }


def _grid_plan_status(
    documents: Mapping[str, Mapping[str, Any]],
    cells: Sequence[tuple[str, str, str]],
    steps: Mapping[int, int],
) -> tuple[int, list[str]]:
    """Which planned (profile, arm, mix, epoch) endpoints have not landed."""
    missing: list[str] = []
    planned = 0
    for profile, arm, mixture in cells:
        for epoch, step in steps.items():
            planned += 1
            endpoint = f"{mixture}-step{step}"
            if endpoint not in documents.get(f"{profile}|{arm}", {}).get("result", {}):
                missing.append(f"{profile}/{arm}/{mixture}@{mix.EPOCH_LABEL[epoch]}")
    return planned, sorted(missing)


def collect_glm_scaleup() -> dict[str, Any]:
    """#1b: one document per arm, endpoints named <cell>-step{2560,5120}."""
    revision = _revision(GLM_REPO)
    wanted: dict[tuple[str, str], str] = {}
    for prefix in GLM_PREFIXES:
        for path in _tree(GLM_REPO, prefix, revision):
            if not (path.startswith(prefix + "/") and path.endswith("/scored.json")):
                continue
            arm, cell, _name = path[len(prefix) + 1:].split("/")
            if cell not in mix.GLM_ROWS_V2.families:
                print(f"skip {path}: {cell} is not a rows-v2 cell")
                continue
            # The rows-v2 prefix wins: it is the live study, and A1's
            # agreement symlink means the same cell can appear under both.
            wanted.setdefault((arm, cell), path)

    documents: dict[str, dict[str, Any]] = {}
    for (arm, cell), path in sorted(wanted.items()):
        local = _download(GLM_REPO, revision, [path])[path]
        scored = json.loads(local.read_text())
        epochs = scored.get("epochs")
        if not isinstance(epochs, dict):
            raise ValueError(f"{path} has no epochs mapping")
        document = documents.setdefault(arm, {"result": {}, "meta": {}})
        for step, slices in epochs.items():
            if int(step) not in mix.GLM_ROWS_V2.steps.values():
                print(f"skip {path} step {step}: not an epoch boundary")
                continue
            if not isinstance(slices, dict) or not slices:
                continue
            document["result"][f"{cell}-step{int(step)}"] = slices
        document["meta"].setdefault("sources", {})[cell] = {
            "repo": GLM_REPO, "revision": revision, "path": path,
            "eval_revision": scored.get("eval_revision"),
        }

    missing = sorted(
        f"{arm}/{cell}@{mix.EPOCH_LABEL[epoch]}"
        for arm in ARMS
        for cell in mix.GLM_ROWS_V2.families
        for epoch, step in mix.GLM_ROWS_V2.steps.items()
        if f"{cell}-step{step}" not in documents.get(arm, {}).get("result", {})
    )
    return {
        "version": "dispatch_glm_aft_scaleup_scores_v1",
        "study": "followup_1b_glm_aft_scaleup",
        "documents": documents,
        "missing": missing,
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_repo": GLM_REPO,
            "hub_revision": revision,
            "hub_prefixes": list(GLM_PREFIXES),
            "profile": GLM_PROFILE,
            "aft_rows": mix.GLM_ROWS_V2.rows,
            "eval_steps": dict(mix.GLM_ROWS_V2.steps),
            "eval_backend": "glm-aft-graphs-splitk1-v1 (vLLM 0.19.1)",
            "eval_backend_note": mix.BACKEND_NOTE,
            "agreement_substrate_note": mix.AGREEMENT_SUBSTRATE_NOTE,
            "baseline": (
                "scored/glm45_air_190m/<arm>/eval.json, 8,192 rows, step 512 "
                "only (GLM has no evaluable intermediate AFT checkpoint)"
            ),
            "endpoints": sum(
                len(document["result"]) for document in documents.values()),
            "endpoints_planned": (
                len(ARMS) * len(mix.GLM_ROWS_V2.families)
                * len(mix.GLM_ROWS_V2.steps)
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--only", action="append", choices=GALLERIES,
                        help="repeat to limit collection; default: all three")
    args = parser.parse_args(argv)
    selected = args.only or list(GALLERIES)

    collectors = {
        "aft_grid": collect_aft_grid,
        "glm_aft_scaleup": collect_glm_scaleup,
        "contamination_quality": collect_contamination_quality,
    }
    for gallery in selected:
        result = collectors[gallery]()
        path = args.out / f"{gallery}.json"
        _write_json(path, result)
        meta = result["meta"]
        print(
            f"wrote {path}\n"
            f"  {meta['endpoints']}/{meta['endpoints_planned']} endpoints; "
            f"{len(result['missing'])} still to land"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
