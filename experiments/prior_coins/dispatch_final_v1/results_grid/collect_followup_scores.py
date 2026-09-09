"""Package the scored endpoints for AFT follow-ups #1a, #1b and #1c.

Every campaign publishes its own aggregated scores beside the responses, so
this is a download-and-repackage step, not a re-score: nothing here samples,
touches a pod, or writes to the Hub.  One output per gallery:

    scored/ablations/aft_grid.json          #1a, gemma 12B/27B x 1%/5%
    scored/ablations/glm_aft_scaleup.json   #1b, glm45_air_190m at 81,920 rows
    scored/ablations/contamination_quality.json  #1c, the gemma 2% repair
    scored/ablations/glm_contamination.json      #1c, the GLM 2% repair
    scored/ablations/glm_threeway.json           #1c, the GLM 80:10:10 cell

Discovery is per ENDPOINT and runs every time, so this is safe to re-run while
the fleet is still training:

* #1a writes ``eval/<endpoint>/scores.json`` only after that endpoint's 21
  response files are validated, so a half-evaluated cell contributes its
  finished epoch and nothing else.
* #1b writes one ``scored.json`` per cell after both epoch evaluations, so a
  cell is all-or-nothing.

The 8,192-row campaign side of both comparisons is NOT packaged here.  It is
already committed under ``scored/<profile>/<arm>/eval.json`` and the plotters
read it from there, so the join stays a join instead of a second copy that can
drift.  See ``followup_mixtures.py`` for the confounds that join carries.

Run from the repository root::

    uv run --extra dev python3 \
      experiments/prior_coins/dispatch_final_v1/results_grid/collect_followup_scores.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import followup_mixtures as mix  # noqa: E402

OUTPUT = HERE / "scored" / "ablations"
CACHE = HERE / "cache" / "followups"

GALLERIES = ("aft_grid", "glm_aft_scaleup", "contamination_quality",
             "glm_contamination", "glm_threeway")

# ------------------------------------------------------------------- #1a Hub
#
# Two output repos, one per model family: the campaign's own repository is at
# its per-file and shared-commit limits, which is why the grid publishes here.
#: This prefix is the DATASET VERSION, and pinning it is load-bearing.
#: `gemma_grid_run.py` publishes to `followups/<plan version>/<job id>`, so
#: follow-up #1c's corrected 2% reruns will land beside these cells under
#: `followups/gemma-aft-2pct-repair-v1/`.  Those are a different 2% dataset and
#: must not be pooled with the campaign's narrow-conflict cells, so this
#: collector reads ONLY the balanced-v2 tree and #1c gets its own entry when it
#: lands.  A repair cell silently absorbed here would erase the very
#: distinction the asterisks exist to draw.
GRID_PREFIX = "followups/gemma-aft-grid-balanced-v2"
#: The 0.5% rung publishes to its own dataset-version prefix but IS part of
#: the AFT-grid gallery: it is a new rung on the same ladder, same 8,192-row
#: geometry, same balanced selection, so merging it adds a column rather than
#: replacing one.  That is the opposite of #1c, which is a competing draw for
#: rungs the campaign already has and therefore must stay separate.
#:
#: The rerun prefix is the SAME rung, not a competing draw.  Five 12B cells
#: were interrupted with weight-only checkpoints, so they were retrained from
#: the pinned parent on the identical recipe and dataset version and published
#: under a distinct attempt prefix rather than overwriting the dead attempt
#: (JONATHAN_GEMMA_HALFPCT_HANDOFF.md).  The interrupted attempts never scored,
#: so nothing collides -- verified 2026-09-08: zero cells carry `scores.json`
#: under both prefixes.  `meta.sources` keeps the per-endpoint path, so which
#: attempt a number came from stays readable in the collected document.
#:
#: The 0.25% rung joins on the same footing and for the same reason: same
#: 8,192-row geometry, same recipe, a NEW rung rather than a competing draw for
#: one the campaign already has.  Its `-v1` prefix is a dead first attempt
#: (no `scores.json`, a `partial-work.tar` beside it) and is not read; the
#: live release is `-v2`.
GRID_EXTRA_PREFIXES: tuple[tuple[str, str], ...] = (
    ("followups/gemma-aft-halfpct-balanced-v1", "grid_8192_halfpct"),
    ("followups/gemma-aft-halfpct-balanced-v1-jonathan-rerun1",
     "grid_8192_halfpct"),
    ("followups/gemma-aft-lowdose-0p25pct-v2", "grid_8192_lowdose"),
)
#: Sibling prefixes deliberately NOT read by `collect_aft_grid`, so the
#: omission is visible.  #1c has its own collector below.
GRID_PREFIXES_IGNORED = (REPAIR_PREFIX := "followups/gemma-aft-2pct-repair-v1",)
REPAIR_PLAN = (
    HERE.parents[3] / "artifacts" / "aft_grid_8192_balanced_v2"
    / "repair-plan-52cells.json"
)
GRID_REPOS = {
    "12b": "arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2",
    "27b": "arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2",
}
GRID_PLAN_FILE = f"{GRID_PREFIX}/plan/grid-plan-12workers.json"
#: The final AFT step, whose checkpoint carries the run's whole-run token
#: counter.  Earlier checkpoints carry partial counts.
TOKEN_STATE_STEP = 512

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
#: Follow-up #1c on GLM: same repo as #1b, its own dataset-version prefix.
GLM_REPAIR_PREFIX = "followups/glm-aft-2pct-repair-v1"
#: The nine-cell release also carries a third `balanced_80_10_10` cell per arm
#: (6,554 agreement / 819 coin / 819 charter).  It is NOT a 2% repair cell and
#: is not part of the substitution, so `collect_glm_contamination` lists it and
#: refuses to package it: a two-sided cell among the endpoints `twopct.py`
#: substitutes is exactly the pooling the asterisks exist to prevent.  It is
#: packaged by `collect_glm_threeway` into its own document instead, which is
#: what `plot_glm_threeway.py` reads.
GLM_REPAIR_CELLS = ("mixed_charter", "mixed_coin")
GLM_REPAIR_EXTRA_CELLS = (mix.THREEWAY.key,)
ARMS = ("charter", "coin", "control")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _listing(repo: str) -> tuple[list[str], str]:
    """One listing per repo per run; the revision is pinned into the output.

    `list_repo_files` and NOT `repo_info().siblings`: siblings is silently
    TRUNCATED on large repos.  Measured 2026-09-08 on the 12B result repo —
    8,810 siblings against 10,879 real files, with completed step-512
    `scores.json` among the missing.  A truncated listing does not error, it
    just makes finished cells look unstarted, so this must never go back.
    """
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="model")
    revision = api.repo_info(repo, repo_type="model").sha
    return list(files), revision


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
    """#1a: one document per (profile, arm), endpoints named <mix>-step<n>."""
    documents: dict[str, dict[str, Any]] = {}
    revisions: dict[str, str] = {}
    plan_source: dict[str, Any] = {}
    endpoints_found = 0
    for family, repo in GRID_REPOS.items():
        files, revision = _listing(repo)
        revisions[repo] = revision
        wanted = sorted(
            path for path in files
            if path.startswith(f"{GRID_PREFIX}/gemma")
            and path.endswith("/scores.json")
            and "/eval/" in path
        )
        # The trainer's own token counter, published beside the final
        # checkpoint.  It is the only measured token figure in the study, and
        # the heat-map axes are denominated in tokens.
        token_files = sorted(
            path for path in files
            if path.startswith(f"{GRID_PREFIX}/gemma")
            and path.endswith(f"/checkpoints/checkpoint-{TOKEN_STATE_STEP}"
                              "/tokens_state.json")
        )
        for extra_prefix, study_key in GRID_EXTRA_PREFIXES:
            endpoints_found += _grid_cells(
                extra_prefix, repo, revision, files,
                mix.STUDIES[study_key].families, documents)
        for path, local in _download(repo, revision, token_files).items():
            parts = path[len(GRID_PREFIX) + 1:].split("/")
            profile, arm, mixture = parts[0], parts[1], parts[2]
            state = json.loads(local.read_text())
            document = documents.setdefault(
                f"{profile}|{arm}", {"result": {}, "meta": {}})
            document["meta"].setdefault("tokens", {})[mixture] = {
                "total": state.get("total"),
                "trainable": state.get("trainable"),
                "epochs": 2, "rows": mix.GRID_V2.rows,
                "path": path, "revision": revision,
            }
        if GRID_PLAN_FILE in files and not plan_source:
            plan_source = {"repo": repo, "revision": revision,
                           "path": GRID_PLAN_FILE}
        for path, local in _download(repo, revision, wanted).items():
            # <prefix>/<profile>/<arm>/<mix>/eval/<endpoint>/scores.json
            parts = path[len(GRID_PREFIX) + 1:].split("/")
            profile, arm, mixture, _eval, endpoint, _name = parts
            if mixture not in mix.GRID_V2.families:
                print(f"skip {path}: {mixture} is not a #1a mixture")
                continue
            step = endpoint.rsplit("-step", 1)[-1]
            document = documents.setdefault(
                f"{profile}|{arm}", {"result": {}, "meta": {}})
            document["result"][f"{mixture}-step{step}"] = _slices(
                json.loads(local.read_text()), path)
            document["meta"].setdefault("sources", {})[
                f"{mixture}-step{step}"] = {
                "repo": repo, "revision": revision, "path": path,
                "model_family": family,
            }
            endpoints_found += 1

    planned, missing = _grid_plan_status(documents, _grid_plan(plan_source))
    return {
        "version": "dispatch_aft_grid_balanced_v2_scores_v1",
        "study": "followup_1a_aft_grid",
        "documents": documents,
        "missing": missing,
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_prefix": GRID_PREFIX,
            "hub_prefixes_merged": [p for p, _ in GRID_EXTRA_PREFIXES],
            "merged_note": (
                "the 0.5% rung is a new dose on the same ladder, not a "
                "competing draw for an existing one, so it merges here"
            ),
            "hub_prefixes_ignored": list(GRID_PREFIXES_IGNORED),
            "ignored_note": (
                "follow-up #1c's corrected 2% cells publish to their own "
                "dataset-version prefix and are deliberately not pooled with "
                "the campaign's narrow-conflict 2% cells"
            ),
            "hub_revisions": revisions,
            "plan_source": plan_source,
            "aft_rows": mix.GRID_V2.rows,
            "eval_steps": dict(mix.GRID_V2.steps),
            "eval_backend": "eager, unchanged from the campaign",
            "baseline": (
                "the campaign's own scored/<profile>/<arm>/eval.json; "
                "its 2% cells are narrow-conflict, see followup_mixtures.py"
            ),
            "endpoints": endpoints_found,
            "endpoints_planned": planned,
            "tokens_note": (
                "meta.tokens[<mixture>] is the trainer's own tokens_state.json "
                f"at step {TOKEN_STATE_STEP}: `total` over all {mix.GRID_V2.rows} "
                "rows x 2 epochs, `trainable` the loss-bearing (answer) "
                "subset. Rows are near-equal length because a mixture "
                "REPLACES agreement rows in place, so conflict tokens are "
                "conflict_rows x total / (rows x epochs)."
            ),
        },
    }


LOCAL_GRID_PLAN = (
    HERE.parents[3] / "artifacts" / "aft_grid_8192_balanced_v2"
    / "grid-plan-12workers.json"
)
#: Every plan whose jobs land in the AFT-grid collection.  The denominator has
#: to be the union: with only the 12-worker plan counted, adding the 0.5% rung
#: made the collector report 148/144, which reads as "done" when most of that
#: rung has not started.
LOCAL_GRID_PLANS: tuple[Path, ...] = (
    LOCAL_GRID_PLAN,
    HERE.parents[3] / "artifacts" / "gemma_aft_halfpct_18workers_v1"
    / "plan.json",
)


def _grid_plan(plan_source: Mapping[str, Any]) -> Mapping[str, Any]:
    """The 12-worker job list, local copy preferred, Hub copy as the fallback.

    The plan is what says which cells are COMING, so without it a partial
    refresh cannot distinguish "still training" from "never scheduled".
    """
    plans = [json.loads(path.read_text())
             for path in LOCAL_GRID_PLANS if path.is_file()]
    if not plans and plan_source:
        local = _download(
            plan_source["repo"], plan_source["revision"],
            [plan_source["path"]])[plan_source["path"]]
        plans = [json.loads(local.read_text())]
    merged: dict[str, Any] = {"workers": {}}
    for index, plan in enumerate(plans):
        # Worker names repeat across plans, so namespace them before merging.
        for name, worker in plan.get("workers", {}).items():
            merged["workers"][f"{index}:{name}"] = worker
    return merged if merged["workers"] else {}


def _grid_cells(
    prefix: str, repo: str, revision: str, files: Sequence[str],
    families: Mapping[str, str],
    documents: dict[str, dict[str, Any]],
) -> int:
    """Package one dataset-version tree into (profile, arm) documents.

    Shared by #1a and #1c: same Hub layout, same per-endpoint marker, and the
    same `<mix>-step<n>` endpoint naming, so the plotters read both through
    `plot_stacked.Unit` without a branch.
    """
    wanted = sorted(
        path for path in files
        if path.startswith(f"{prefix}/gemma")
        and path.endswith("/scores.json") and "/eval/" in path
    )
    found = 0
    for path, local in _download(repo, revision, wanted).items():
        parts = path[len(prefix) + 1:].split("/")
        profile, arm, mixture, _eval, endpoint, _name = parts
        if mixture not in families.values():
            print(f"skip {path}: {mixture} is not a cell of this study")
            continue
        step = endpoint.rsplit("-step", 1)[-1]
        document = documents.setdefault(
            f"{profile}|{arm}", {"result": {}, "meta": {}})
        document["result"][f"{mixture}-step{step}"] = _slices(
            json.loads(local.read_text()), path)
        document["meta"].setdefault("sources", {})[f"{mixture}-step{step}"] = {
            "repo": repo, "revision": revision, "path": path,
        }
        found += 1
    return found


def collect_contamination_quality() -> dict[str, Any]:
    """#1c: the corrected 2% cells, packaged for the paired-quality gallery.

    The legacy side of the pair is NOT copied here -- it is the campaign's own
    `scored/<profile>/<arm>/eval.json`, already committed, and the plotter
    reads it from there.  Keeping one copy keeps the two arms of the contrast
    from drifting apart.
    """
    documents: dict[str, dict[str, Any]] = {}
    revisions: dict[str, str] = {}
    found = 0
    for repo in GRID_REPOS.values():
        files, revision = _listing(repo)
        revisions[repo] = revision
        found += _grid_cells(REPAIR_PREFIX, repo, revision, files,
                             mix.GRID_REPAIR.families, documents)

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
    return {
        "version": "dispatch_contamination_quality_scores_v1",
        "study": "followup_1c_contamination_data_quality",
        "documents": documents,
        "missing": sorted(missing),
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_prefix": REPAIR_PREFIX,
            "hub_revisions": revisions,
            "aft_rows": mix.GRID_REPAIR.rows,
            "eval_steps": dict(mix.GRID_REPAIR.steps),
            "eval_backend": "eager, unchanged from the campaign",
            "contrast": mix.CONTAMINATION_QUALITY_NOTE,
            "legacy_side": (
                "scored/<profile>/<arm>/eval.json, endpoints "
                "mixed_charter-step* / mixed_coin-step* — the campaign's own "
                "narrow-conflict cells, read in place rather than copied"
            ),
            "endpoints": found,
            "endpoints_planned": planned,
        },
    }


def collect_glm_contamination() -> dict[str, Any]:
    """#1c on GLM-4.5-Air @190M: the corrected 2% cells for the GLM row.

    Packaged separately from the gemma repair because it is a different repo,
    a different prefix and a different eval backend -- pooling them into one
    document would hide the last of those.
    """
    files, revision = _listing(GLM_REPO)
    documents: dict[str, dict[str, Any]] = {}
    extras: dict[str, list[str]] = {}
    found = 0
    wanted = sorted(
        path for path in files
        if path.startswith(f"{GLM_REPAIR_PREFIX}/{GLM_PROFILE}/")
        and path.endswith("/scores.json") and "/eval/" in path
    )
    for path, local in _download(GLM_REPO, revision, wanted).items():
        rest = path[len(f"{GLM_REPAIR_PREFIX}/{GLM_PROFILE}/"):]
        arm, cell, _eval, endpoint, _name = rest.split("/")
        if cell in GLM_REPAIR_EXTRA_CELLS:
            extras.setdefault(arm, []).append(endpoint)
            continue
        if cell not in GLM_REPAIR_CELLS:
            print(f"skip {path}: {cell} is not a GLM #1c 2% cell")
            continue
        step = endpoint.rsplit("-step", 1)[-1]
        document = documents.setdefault(arm, {"result": {}, "meta": {}})
        document["result"][f"{cell}-step{step}"] = _slices(
            json.loads(local.read_text()), path)
        document["meta"].setdefault("sources", {})[f"{cell}-step{step}"] = {
            "repo": GLM_REPO, "revision": revision, "path": path,
        }
        found += 1

    missing = sorted(
        f"{arm}/{cell}@{mix.EPOCH_LABEL[epoch]}"
        for arm in ARMS for cell in GLM_REPAIR_CELLS
        for epoch, step in mix.GLM_REPAIR.steps.items()
        if f"{cell}-step{step}" not in documents.get(arm, {}).get("result", {})
    )
    return {
        "version": "dispatch_glm_contamination_quality_scores_v1",
        "study": "followup_1c_glm_contamination_data_quality",
        "documents": documents,
        "missing": missing,
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_repo": GLM_REPO,
            "hub_revision": revision,
            "hub_prefix": GLM_REPAIR_PREFIX,
            "profile": GLM_PROFILE,
            "aft_rows": mix.GLM_REPAIR.rows,
            "eval_steps": dict(mix.GLM_REPAIR.steps),
            "eval_backend": "glm-aft-graphs-splitk1-v1 (vLLM 0.19.1)",
            "eval_backend_note": mix.BACKEND_NOTE,
            "step256_note": (
                "step 256 exists here and has NO campaign counterpart: the "
                "campaign's GLM intermediate AFT checkpoints were FSDP shards "
                "with no adapter, so that row reads step 512 alone"
            ),
            "contrast": mix.CONTAMINATION_QUALITY_NOTE,
            "extra_cells_seen": {arm: sorted(v) for arm, v in extras.items()},
            "extra_cells_note": (
                "balanced_80_10_10 is a third independent cell in the same "
                "release, not a 2% repair; it is listed, never substituted. "
                "Its scores are packaged by collect_glm_threeway into "
                "scored/ablations/glm_threeway.json"
            ),
            "endpoints": found,
            "endpoints_planned": (
                len(ARMS) * len(GLM_REPAIR_CELLS) * len(mix.GLM_REPAIR.steps)
            ),
        },
    }


def collect_glm_threeway() -> dict[str, Any]:
    """#1c's two-sided cell: one document per arm, `balanced_80_10_10-step<n>`.

    The same repo, prefix, revision, recipe and eval backend as
    `collect_glm_contamination` -- the two collectors read one release -- and a
    separate document all the same.  The 2% document is what `twopct.py`
    overlays onto the canonical scored tree, and `is_twopct` filters by
    endpoint FAMILY, so a two-sided cell sitting in it would be one renamed
    family away from being substituted for a one-sided 2% measurement.  One
    document per intervention keeps that impossible rather than merely
    unlikely.
    """
    files, revision = _listing(GLM_REPO)
    documents: dict[str, dict[str, Any]] = {}
    found = 0
    wanted = sorted(
        path for path in files
        if path.startswith(f"{GLM_REPAIR_PREFIX}/{GLM_PROFILE}/")
        and path.endswith("/scores.json") and "/eval/" in path
    )
    for path, local in _download(GLM_REPO, revision, wanted).items():
        rest = path[len(f"{GLM_REPAIR_PREFIX}/{GLM_PROFILE}/"):]
        arm, cell, _eval, endpoint, _name = rest.split("/")
        if cell != mix.THREEWAY.key:
            # The 2% siblings; `collect_glm_contamination` owns those.
            continue
        step = endpoint.rsplit("-step", 1)[-1]
        document = documents.setdefault(arm, {"result": {}, "meta": {}})
        document["result"][f"{cell}-step{step}"] = _slices(
            json.loads(local.read_text()), path)
        document["meta"].setdefault("sources", {})[f"{cell}-step{step}"] = {
            "repo": GLM_REPO, "revision": revision, "path": path,
        }
        found += 1

    missing = sorted(
        f"{arm}/{mix.THREEWAY.key}@{mix.EPOCH_LABEL[epoch]}"
        for arm in ARMS for epoch, step in mix.GLM_THREEWAY.steps.items()
        if f"{mix.THREEWAY.key}-step{step}" not in
        documents.get(arm, {}).get("result", {})
    )
    return {
        "version": "dispatch_glm_threeway_scores_v1",
        "study": "followup_1c_glm_two_sided_80_10_10",
        "documents": documents,
        "missing": missing,
        "meta": {
            "collected": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hub_repo": GLM_REPO,
            "hub_revision": revision,
            "hub_prefix": GLM_REPAIR_PREFIX,
            "profile": GLM_PROFILE,
            "cell": mix.THREEWAY.key,
            "aft_rows": mix.THREEWAY.rows,
            "mixture_rows": {
                "agreement": mix.THREEWAY.agreement_rows,
                "coin": mix.THREEWAY.coin_rows,
                "charter": mix.THREEWAY.charter_rows,
            },
            "eval_steps": dict(mix.GLM_THREEWAY.steps),
            "eval_backend": "glm-aft-graphs-splitk1-v1 (vLLM 0.19.1)",
            "eval_backend_note": mix.THREEWAY_BACKEND_NOTE,
            "dose_note": mix.THREEWAY_DOSE_NOTE,
            "context_cells": list(mix.THREEWAY_CONTEXT),
            "sibling_document": "glm_contamination.json",
            "endpoints": found,
            "endpoints_planned": len(ARMS) * len(mix.GLM_THREEWAY.steps),
        },
    }


def _grid_plan_status(
    documents: Mapping[str, Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> tuple[int, list[str]]:
    """Which planned (profile, arm, mix, epoch) endpoints have not landed."""
    if not plan:
        return 0, []
    missing: list[str] = []
    planned = 0
    for worker in plan.get("workers", {}).values():
        for job in worker.get("jobs", []):
            for epoch, step in mix.grid_owner(job["mix"]).steps.items():
                planned += 1
                key = f"{job['profile']}|{job['arm']}"
                endpoint = f"{job['mix']}-step{step}"
                if endpoint not in documents.get(key, {}).get("result", {}):
                    missing.append(
                        f"{job['profile']}/{job['arm']}/{job['mix']}"
                        f"@{mix.EPOCH_LABEL[epoch]}")
    return planned, sorted(missing)


def collect_glm_scaleup() -> dict[str, Any]:
    """#1b: one document per arm, endpoints named <cell>-step{2560,5120}."""
    files, revision = _listing(GLM_REPO)
    wanted: dict[tuple[str, str], str] = {}
    for prefix in GLM_PREFIXES:
        for path in files:
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


class _CacheLock:
    """Refuse to run a second collector against the same download cache.

    There is no per-file locking in `hf_hub_download`'s use here, so two
    collectors racing on one cache can leave a half-written `scores.json`
    behind.  Measured 2026-09-08: one such file was 34,470 bytes with blocks
    allocated, `stat`-ed clean, and still raised ENXIO on every read, which
    aborted the whole collection.  Serialise instead of trying to repair.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> "_CacheLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            owner = self.path.read_text().strip() or "unknown"
            raise SystemExit(
                f"another collector holds {self.path} (started by {owner}).\n"
                "Wait for it to finish -- concurrent runs corrupt the cache.\n"
                f"If no collector is running, remove the file: rm {self.path}"
            ) from None
        with os.fdopen(handle, "w") as stream:
            stream.write(f"pid {os.getpid()} at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
        return self

    def __exit__(self, *exc: object) -> None:
        self.path.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--only", action="append", choices=GALLERIES,
                        help="repeat to limit collection; default: both")
    args = parser.parse_args(argv)
    selected = args.only or list(GALLERIES)

    collectors = {
        "aft_grid": collect_aft_grid,
        "glm_aft_scaleup": collect_glm_scaleup,
        "contamination_quality": collect_contamination_quality,
        "glm_contamination": collect_glm_contamination,
        "glm_threeway": collect_glm_threeway,
    }
    with _CacheLock(CACHE / ".collector.lock"):
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
