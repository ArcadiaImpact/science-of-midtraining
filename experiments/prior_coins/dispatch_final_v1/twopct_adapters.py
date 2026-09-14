"""Where each row's CANONICAL 2% conflict adapter actually lives on the Hub.

Every published figure plots follow-up #1c's **corrected balanced** 2% draw.
The scored JSON was substituted in place (``results_grid/twopct.py``), but the
WEIGHTS were not moved: for most rows the adapter under
``<profile>/<arm>/aft/mixed_coin/`` is still the campaign's narrow
single-clause draw, and the corrected one sits in a follow-up repo under a
different layout.

That is easy to miss and expensive to get wrong -- a re-run that served the
canonical path would quietly measure the superseded draw and then be compared
against figures drawn from the corrected one.  This module is the one place
that mapping is written down, and ``rerun_costsweep_v2.py`` resolves through
it.

Two rows are exempt because their campaign cells were never narrow
(``followup_mixtures.ALREADY_BALANCED_2PCT``): ``glm45_air_1b`` trained on
``aft_manifest_balanced_v2.json``, and ``glm45_air_20m_legacy`` never used the
buggy selector at all.  For those the canonical path IS the corrected draw.

Measured against the Hub on 2026-09-11: the gemma repair keeps full PEFT
adapters at ``.../{cell}/train/checkpoints/checkpoint-{step}/`` and the GLM
repair at ``.../{cell}/adapters/step{step}/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), str(HERE / "results_grid")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import followup_mixtures as mix  # noqa: E402

#: Rows whose own ``aft/mixed_*`` cells are already the corrected draw.
ALREADY_BALANCED = mix.ALREADY_BALANCED_2PCT

GEMMA_REPAIR_PREFIX = "followups/gemma-aft-2pct-repair-v1"
GLM_REPAIR_PREFIX = "followups/glm-aft-2pct-repair-v1"


@dataclass(frozen=True)
class RepairSource:
    repo: str
    #: ``str.format``-ready, taking profile, arm, cell and step.
    template: str

    def path_in_repo(self, profile: str, arm: str, cell: str, step: int) -> str:
        return self.template.format(profile=profile, arm=arm, cell=cell, step=step)


_GEMMA_12B = RepairSource(
    repo="arcadia-impact/scimt-dispatch-gemma-12b-aft-grid-v2",
    template=(GEMMA_REPAIR_PREFIX
              + "/{profile}/{arm}/{cell}/train/checkpoints/checkpoint-{step}"),
)
_GEMMA_27B = RepairSource(
    repo="arcadia-impact/scimt-dispatch-gemma-27b-aft-grid-v2",
    template=(GEMMA_REPAIR_PREFIX
              + "/{profile}/{arm}/{cell}/train/checkpoints/checkpoint-{step}"),
)
_GLM_190M = RepairSource(
    repo="arcadia-impact/scimt-dispatch-final-v1-glm",
    template=GLM_REPAIR_PREFIX + "/{profile}/{arm}/{cell}/adapters/step{step}",
)

#: profile -> where its corrected 2% adapters live. A profile absent from this
#: table and absent from ALREADY_BALANCED has no corrected draw published, and
#: `resolve` says so loudly rather than falling back to the narrow one.
REPAIR_SOURCES: dict[str, RepairSource] = {
    "gemma3_12b_1m": _GEMMA_12B,
    "gemma3_12b_5m": _GEMMA_12B,
    "gemma3_12b_19m": _GEMMA_12B,
    "gemma3_12b_50m_4ep": _GEMMA_12B,
    "gemma3_12b_50m_noex": _GEMMA_12B,
    "gemma3_27b_5m": _GEMMA_27B,
    "gemma3_27b_19m": _GEMMA_27B,
    "gemma3_27b_50m": _GEMMA_27B,
    "gemma3_27b_190m": _GEMMA_27B,
    "glm45_air_190m": _GLM_190M,
}

#: Cells the substitution covers. `agreement` and `charter_only` were never
#: drawn by the buggy selector -- they have no conflict rows to draw, and all
#: 8,192 respectively -- so they are served from the canonical path.
REPAIRED_CELLS = ("mixed_charter", "mixed_coin")


def needs_repair_adapter(profile: str, cell: str) -> bool:
    """True when this row's canonical ``aft/<cell>`` is the superseded draw."""
    return cell in REPAIRED_CELLS and profile not in ALREADY_BALANCED


def resolve(profile: str, arm: str, cell: str, step: int) -> tuple[str, str] | None:
    """``(repo, path_in_repo)`` of the corrected adapter, or None if canonical.

    ``None`` means "serve the row's own ``aft/<cell>`` directory", which is
    correct for the agreement/charter_only cells and for the two already
    balanced rows.  An unpublished repair raises instead of falling through.
    """
    if not needs_repair_adapter(profile, cell):
        return None
    source = REPAIR_SOURCES.get(profile)
    if source is None:
        raise KeyError(
            f"{profile}/{cell}: the canonical cell is the superseded narrow 2% "
            "draw and no #1c repair is registered for this row. Add it to "
            "REPAIR_SOURCES, or add the row to ALREADY_BALANCED if its draw "
            "was never narrow -- do not serve the narrow adapter by default."
        )
    return source.repo, source.path_in_repo(profile, arm, cell, step)


#: The corrected 2% cells as a DATASET -- the rows the repair adapters were
#: trained on. ``costsweep_eval``'s adapter probe checks exact matches on a
#: cell's own training rows, so a repair adapter probed against the narrow
#: cell's rows would look broken. Values read out of
#: ``publish_receipt_aft_balanced_v2.json`` (2026-09-07 publish).
CORRECTED_CELL_REPO = "arcadia-impact/scimt-dispatch-charter-250m-v1"
CORRECTED_CELL_PREFIX = "releases/dispatch-charter-250m-v1/aft"
CORRECTED_CELL_REVISION = "09ede6a6c9ac7e041061b87d7651aec8ca8ff8ac"
CORRECTED_CELL_MANIFEST = HERE / "aft_manifest_balanced_v2.json"


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_corrected_cell_rows(cell: str, dest: Path) -> Path:
    """Download one corrected AFT cell's rows, verified against the manifest."""
    import json

    from huggingface_hub import hf_hub_download

    manifest = json.loads(CORRECTED_CELL_MANIFEST.read_text())
    expected = manifest["cells"][cell]["sha256"]
    path = Path(hf_hub_download(
        CORRECTED_CELL_REPO, f"{CORRECTED_CELL_PREFIX}/aft_{cell}.jsonl",
        repo_type="dataset", revision=CORRECTED_CELL_REVISION,
        local_dir=dest))
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(
            f"aft_{cell}.jsonl sha256 {actual} != manifest {expected}")
    return path


#: Sidecar written next to an installed repair adapter: where the bytes came
#: from (repo, prefix, resolved commit) and each file's sha256.
REPAIR_SOURCE = "REPAIR_SOURCE.json"


def install_repair_adapter(
    profile: str, arm: str, cell: str, step: int, arm_root: Path,
    *, revision: str | None = None,
) -> Path | None:
    """Put the corrected adapter where ``d4_eval.endpoints`` will find it.

    Returns the installed directory, or ``None`` when this row serves the
    canonical cell.  Installing into ``aft/<cell>/checkpoints/checkpoint-<step>``
    means every downstream consumer -- endpoint naming, the adapter probe,
    ``contracts.aft_adapter_dir`` -- keeps working unchanged; only the bytes
    behind the path are the corrected draw.

    The download is pinned to one commit (``revision``, default the repo's
    current head, resolved once) and recorded with per-file digests in
    ``REPAIR_SOURCE.json`` inside the directory. A directory that already
    holds an adapter is served only if its sidecar names the same source and
    its files still match their digests -- an adapter of unknown origin at that
    path is an error, not a silent reuse (the narrow canonical draw could sit
    there after ``rehydrate --for-phase costsweep``).
    """
    import json
    import shutil

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.hf_api import RepoFile

    located = resolve(profile, arm, cell, step)
    if located is None:
        return None
    repo, path_in_repo = located
    dest = arm_root / "aft" / cell / "checkpoints" / f"checkpoint-{step}"
    sidecar = dest / REPAIR_SOURCE
    if (dest / "adapter_config.json").is_file():
        if not sidecar.is_file():
            raise RuntimeError(
                f"{dest} already holds an adapter with no {REPAIR_SOURCE}; it cannot be "
                "certified as the corrected draw -- remove it and rerun")
        recorded = json.loads(sidecar.read_text())
        same_source = (recorded.get("repo") == repo and recorded.get("path_in_repo") == path_in_repo
                       and (revision is None or recorded.get("revision") == revision))
        if not same_source:
            raise RuntimeError(
                f"{dest} holds an adapter from {recorded.get('repo')}/{recorded.get('path_in_repo')}"
                f"@{recorded.get('revision')}, not {repo}/{path_in_repo}@{revision or 'head'}")
        for name, digest in recorded["files"].items():
            if not (dest / name).is_file() or _sha256(dest / name) != digest:
                raise RuntimeError(f"{dest / name} does not match the digest in {REPAIR_SOURCE}")
        return dest
    api = HfApi()
    resolved = api.repo_info(repo, repo_type="model", revision=revision).sha
    # Per-file downloads, for the reason rehydrate.restore_bytes documents:
    # snapshot_download(allow_patterns=...) crashes inside thread_map on the
    # hub/tqdm pair these pods resolve, matching or not.
    remote = [
        entry.path for entry in api.list_repo_tree(
            repo, repo_type="model", recursive=True, path_in_repo=path_in_repo,
            revision=resolved)
        if isinstance(entry, RepoFile)
    ]
    if not any(name.endswith("/adapter_config.json") for name in remote):
        raise RuntimeError(
            f"{repo}/{path_in_repo}@{resolved} carries no adapter_config.json; it is not a "
            "servable PEFT adapter")
    dest.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in remote:
        # flat copy: the adapter is one directory, and `dest` IS that directory
        local = hf_hub_download(repo, name, repo_type="model", revision=resolved)
        target = dest / Path(name).name
        shutil.copyfile(local, target)
        files[target.name] = _sha256(target)
    sidecar.write_text(json.dumps({
        "repo": repo, "path_in_repo": path_in_repo, "revision": resolved,
        "requested_revision": revision, "files": files,
    }, indent=2) + "\n")
    return dest
