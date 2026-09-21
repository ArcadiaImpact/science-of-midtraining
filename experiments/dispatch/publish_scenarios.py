"""Push the naturalized v3 scenario sets to a (private) HF dataset repo.

Naturalization is the expensive, non-deterministic step in this experiment:
every AFT row and eval item is a temp-1.0 gpt-5-mini render that had to pass
the structural validator. Local run dirs are impermanent (`runs/` is
gitignored, pods and scratch disks come and go), so the rendered sets are
pushed once and reused by every downstream cell — the dataset peer of
`scimt.publish`'s pointers-not-weights rule for checkpoints.

Paths are **vocabulary-keyed** (`scenarios/v3-<vocabulary>/`): a set rendered
under status vocabulary C must never be silently mixed with one rendered under
D. The uploaded manifest carries the vocabulary, the naturalization nonce, the
seed, the renderer model, and the git commit, so a downstream run can prove
which bytes it evaluated.

    from experiments.dispatch.publish_scenarios import publish_scenarios

    result = await publish_scenarios("experiments/dispatch/runs/v3")

Async like the rest of the pipeline; the blocking Hub upload runs in a worker
thread. Needs `HF_TOKEN`.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_DATASET_REPO = "arcadia-impact/scimt-prior-coins-scenarios"


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _collect(run_root: Path) -> tuple[dict[str, Any], list[Path]]:
    """Read the naturalization summary and list the files worth publishing."""

    scenarios = run_root / "scenarios"
    summary_path = scenarios / "naturalization_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"no naturalization summary at {summary_path}; run the naturalize "
            "phase before publishing"
        )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    files = [summary_path]
    # AFT sets and eval batteries only — the naturalization cache stays local
    # (it holds raw provider responses, and a re-run replays it from disk).
    for subdir in ("aft", "eval"):
        directory = scenarios / subdir
        if not directory.is_dir():
            continue
        files.extend(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file() and path.suffix in {".json", ".jsonl"}
            )
        )
    if len(files) == 1:
        raise FileNotFoundError(f"no rendered sets under {scenarios}")
    return summary, files


def _manifest(
    summary: dict[str, Any],
    files: list[Path],
    run_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    return {
        "experiment": "dispatch",
        "world_version": "v3",
        "status_vocabulary": summary.get("vocabulary"),
        "naturalization_model": summary.get("model"),
        "naturalization_reasoning_effort": summary.get("reasoning_effort"),
        "naturalization_nonce": summary.get("nonce"),
        "regen_rate": summary.get("regen_rate"),
        "n_dropped": summary.get("n_dropped"),
        "dropped": summary.get("dropped"),
        "collections": {
            name: {
                key: report.get(key)
                for key in ("n", "n_expected", "n_dropped", "regen_rate")
            }
            for name, report in summary.get("collections", {}).items()
        },
        "git_commit": _git_commit(repo_root),
        "source_run_dir": str(run_root),
        "files": [str(path.relative_to(run_root / "scenarios")) for path in files],
    }


async def publish_scenarios(
    run_root: str | Path,
    repo_id: str = DEFAULT_DATASET_REPO,
    *,
    private: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload the rendered sets under ``scenarios/v3-<vocabulary>/``.

    The repo is created ``private`` by default — publishing is outward-facing
    and these sets embed the invented world's Charter, which is also the eval
    ground truth.
    """

    from huggingface_hub import HfApi

    run_root = Path(run_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    summary, files = _collect(run_root)
    vocabulary = summary.get("vocabulary")
    if not vocabulary:
        raise ValueError("naturalization summary has no vocabulary; refusing to push")
    prefix = f"scenarios/v3-{vocabulary}"

    manifest = _manifest(summary, files, run_root, repo_root)
    manifest_path = run_root / "scenarios" / "publish_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))

    def _push() -> None:
        api.create_repo(
            repo_id, repo_type="dataset", private=private, exist_ok=True
        )
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(run_root / "scenarios"),
            path_in_repo=prefix,
            # The cache holds raw provider responses and is replayed locally;
            # it is not part of the durable artifact.
            ignore_patterns=["naturalization_cache/*", "**/naturalization_cache/*"],
            commit_message=(
                f"prior-coins v3 scenarios (vocabulary {vocabulary}, "
                f"nonce {summary.get('nonce')})"
            ),
        )

    await asyncio.to_thread(_push)
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}",
        "path_in_repo": prefix,
        "private": private,
        "n_files": len(files),
        "manifest": manifest,
    }


def _corpus_stats(path: Path) -> dict[str, Any]:
    n_docs = 0
    est_tokens = 0
    domains: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            n_docs += 1
            est_tokens += int(row.get("tokens_est") or 0)
            domain = row.get("domain")
            if isinstance(domain, str):
                domains.add(domain)
    return {"n_docs": n_docs, "est_tokens": est_tokens, "n_domains": len(domains)}


async def publish_corpora(
    run_root: str | Path,
    repo_id: str = DEFAULT_DATASET_REPO,
    *,
    private: bool = True,
    token: str | None = None,
) -> dict[str, Any]:
    """Upload the Z1/Z2 midtraining corpora under ``corpora/v3-<vocabulary>/``.

    Both cuts go up, because they answer different questions: ``full`` is what
    generation produced, ``balanced`` is the pair-matched cut (equal per-domain
    counts in both corpora) that training actually consumes. Only the corpus
    documents and their generation summaries travel — ``raw_batches`` is
    per-batch scaffolding and ``dataset.jsonl`` is a re-encoding of the same
    rows, so both are left local.

    The vocabulary comes from the generation summaries, not from a caller
    argument: a corpus rendered under status vocabulary C must not be filed
    where D's would sit.
    """

    from huggingface_hub import HfApi

    run_root = Path(run_root).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    corpora = run_root / "corpora"
    uploads: list[tuple[Path, str]] = []
    manifest: dict[str, Any] = {
        "experiment": "dispatch",
        "world_version": "v3",
        "git_commit": _git_commit(repo_root),
        "source_run_dir": str(run_root),
        "cuts": {},
    }
    vocabularies: set[str] = set()

    for cut in ("full", "balanced"):
        cut_root = corpora / cut
        if not cut_root.is_dir():
            continue
        entry: dict[str, Any] = {}
        for corpus in ("z1", "z2"):
            path = cut_root / corpus / "corpus.jsonl"
            if not path.is_file():
                continue
            entry[corpus] = _corpus_stats(path)
            uploads.append((path, f"{cut}/{corpus}/corpus.jsonl"))
            summary_path = cut_root / corpus / "generation_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                vocabulary = summary.get("status_vocabulary")
                if isinstance(vocabulary, str):
                    vocabularies.add(vocabulary.strip().upper()[:1])
                entry[corpus]["kept_yield"] = summary.get("kept_yield")
                entry[corpus]["near_dup_rate"] = summary.get("measurements", {}).get(
                    "cross_batch_near_dup_rate"
                )
                uploads.append(
                    (summary_path, f"{cut}/{corpus}/generation_summary.json")
                )
        if entry:
            manifest["cuts"][cut] = entry

    for name in ("pair_balance.json", "health_report.json"):
        path = run_root / name
        if path.is_file():
            uploads.append((path, name))

    if not uploads:
        raise FileNotFoundError(f"no corpora to publish under {corpora}")
    if len(vocabularies) != 1:
        raise ValueError(
            f"expected exactly one status vocabulary across the corpora, "
            f"found {sorted(vocabularies) or 'none'}"
        )
    vocabulary = vocabularies.pop()
    prefix = f"corpora/v3-{vocabulary}"
    manifest["status_vocabulary"] = vocabulary
    manifest_path = corpora / "publish_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    uploads.append((manifest_path, "publish_manifest.json"))

    api = HfApi(token=token or os.environ.get("HF_TOKEN"))

    def _push() -> None:
        api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
        for local, remote in uploads:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=f"{prefix}/{remote}",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"prior-coins v3 corpora ({vocabulary}): {remote}",
            )

    await asyncio.to_thread(_push)
    return {
        "repo_id": repo_id,
        "url": f"https://huggingface.co/datasets/{repo_id}/tree/main/{prefix}",
        "path_in_repo": prefix,
        "private": private,
        "n_files": len(uploads),
        "manifest": manifest,
    }
