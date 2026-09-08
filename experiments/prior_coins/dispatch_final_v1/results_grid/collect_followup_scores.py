"""Package the scored endpoints for AFT follow-ups #1a and #1b.

Both campaigns publish their own aggregated scores beside the responses, so
this is a download-and-repackage step, not a re-score: nothing here samples,
touches a pod, or writes to the Hub.  Two outputs, one per gallery:

    scored/ablations/aft_grid.json          #1a, gemma 12B/27B x 1%/5%
    scored/ablations/glm_aft_scaleup.json   #1b, glm45_air_190m at 81,920 rows

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

GALLERIES = ("aft_grid", "glm_aft_scaleup", "contamination_quality")

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
ARMS = ("charter", "coin", "control")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _listing(repo: str) -> tuple[list[str], str]:
    """One listing per repo per run; the revision is pinned into the output."""
    info = HfApi().repo_info(repo, repo_type="model")
    return [sibling.rfilename for sibling in info.siblings], info.sha


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


def _grid_plan(plan_source: Mapping[str, Any]) -> Mapping[str, Any]:
    """The 12-worker job list, local copy preferred, Hub copy as the fallback.

    The plan is what says which cells are COMING, so without it a partial
    refresh cannot distinguish "still training" from "never scheduled".
    """
    if LOCAL_GRID_PLAN.is_file():
        return json.loads(LOCAL_GRID_PLAN.read_text())
    if not plan_source:
        return {}
    local = _download(
        plan_source["repo"], plan_source["revision"], [plan_source["path"]],
    )[plan_source["path"]]
    return json.loads(local.read_text())


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
            for epoch, step in mix.GRID_V2.steps.items():
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
