"""Reconstitute the gate2 balanced chain into resolve_stage-acceptable run dirs.

The three stages' canonical run records (checkpoint.json / run.json /
axolotl.yaml / config snapshots) survive on HF, but they cross-check
pod-absolute paths (``/workspace/runtime/...``). This module recreates those
exact trees on the attribution pod:

1. download the published stage records and boundary checkpoints;
2. regenerate the two datasets that were pod-local — the balanced midtrain
   corpus (deterministic take_token_budget + weighted_token_interleave over
   pinned sources; HARD-gated on the run's own jsonl_sha256,
   ordered_rows_sha256, per-source counts, AND the published per-line sha256
   ledger) and the filtered+shuffled Dolci hf_dir (HARD-gated on the recorded
   datasets fingerprint);
3. assemble each stage's run dir at its recorded location and assert the
   recorded checkpoint.json paths resolve without rewriting;
4. CPU-check every stage with ``scimt.data_attribution.stages.resolve_stage``
   before any GPU phase runs.

Everything here is importable and side-effect free at import; the pod driver
owns sequencing.  No CLIs (repo convention).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.improved_midtraining.dispatch_gate2_midtrain4 import (  # noqa: E402
    contracts as gate2,
)
from experiments.improved_midtraining.gate2_lineage_attribution import (  # noqa: E402
    contracts,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Byte-identical twin of the gate2 pod writer (compact JSON + newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            line = json.dumps(
                dict(row), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            handle.write(line + "\n")
            digest.update(line.encode())
            digest.update(b"\n")
    return digest.hexdigest()


# ------------------------------------------------------------ HF downloads
def download_stage_records(staging: Path, token: str | None) -> dict[str, Path]:
    local = _download_prefixes(
        contracts.GATE2_EVIDENCE_REPO,
        contracts.GATE2_STAGE_RECORD.values(),
        staging / "gate2_evidence",
        token,
        repo_type="dataset",
        exact_files=[
            contracts.GATE2_MIDTRAIN_MANIFEST,
            contracts.GATE2_DOLCI_MANIFEST,
        ],
    )
    aft_local = _download_prefixes(
        contracts.AFT_EVIDENCE_REPO,
        [contracts.AFT_TRAINING_EVIDENCE],
        staging / "aft_evidence",
        token,
        repo_type="dataset",
    )
    return {
        "post_midtrain": local / contracts.GATE2_STAGE_RECORD["post_midtrain"],
        "post_dolci100": local / contracts.GATE2_STAGE_RECORD["post_dolci100"],
        "midtraining_manifest": local / contracts.GATE2_MIDTRAIN_MANIFEST,
        "dolci_manifest": local / contracts.GATE2_DOLCI_MANIFEST,
        "aft_training": aft_local / contracts.AFT_TRAINING_EVIDENCE,
    }


def _download_prefixes(
    repo_id: str,
    prefixes: Iterable[str],
    local_dir: Path,
    token: str | None,
    repo_type: str = "model",
    exact_files: Iterable[str] = (),
) -> Path:
    """Explicit list+fetch loop.

    snapshot_download(allow_patterns=...) crashed on the pod image's
    huggingface_hub/tqdm pairing (thread_map over a generator with no length
    hint, run 20260818T093619Z); listing files ourselves is version-proof and
    lets us fail loudly on an empty prefix instead of deep in tqdm.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(repo_id, token=token, repo_type=repo_type)
    wanted = list(exact_files)
    for prefix in prefixes:
        matched = [name for name in files if name.startswith(f"{prefix}/")]
        if not matched:
            raise RuntimeError(
                f"no files under prefix {prefix!r} in {repo_id} — pin drift?"
            )
        wanted.extend(matched)
    for name in wanted:
        if name not in files:
            raise RuntimeError(f"pinned file {name!r} missing from {repo_id}")
        hf_hub_download(
            repo_id, name, token=token, repo_type=repo_type, local_dir=local_dir
        )
    return local_dir


def download_checkpoints(staging: Path, token: str | None) -> dict[str, Path]:
    gate2_local = _download_prefixes(
        contracts.GATE2_MODELS_REPO,
        contracts.GATE2_MODEL_PREFIX.values(),
        staging / "gate2_models",
        token,
    )
    aft_local = _download_prefixes(
        contracts.AFT_MODELS_REPO,
        [contracts.AFT_MODEL_PREFIX],
        staging / "aft_models",
        token,
    )
    return {
        "post_midtrain": gate2_local / contracts.GATE2_MODEL_PREFIX["post_midtrain"],
        "post_dolci100": gate2_local / contracts.GATE2_MODEL_PREFIX["post_dolci100"],
        "aft_512": aft_local / contracts.AFT_MODEL_PREFIX,
    }


def download_aft_agreement(staging: Path, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            contracts.WAVE_DATA_REPO,
            contracts.WAVE_AGREEMENT_PATH,
            repo_type="dataset",
            revision=contracts.WAVE_DATA_REVISION,
            token=token,
            local_dir=staging / "wave_data",
        )
    )
    observed = _sha256_file(path)
    if observed != contracts.WAVE_AGREEMENT_SHA256:
        raise RuntimeError(
            f"aft_agreement.jsonl sha256 {observed} != pinned "
            f"{contracts.WAVE_AGREEMENT_SHA256}"
        )
    return path


# --------------------------------------------------- midtrain corpus regen
def build_midtrain_rows(
    *, api: Any, token: str, tokenizer: Any, scratch: Path
) -> list[dict[str, Any]]:
    """Regenerate the balanced ordered midtrain rows (text/tokens/source).

    Exact port of the gate2 midtrain pod's data path (commit c77eac30
    ``prepare_data``), consuming the SAME pinned sources and the current
    ``dispatch_gate2_midtrain4.contracts`` selection/interleave functions.
    Any digest drift is caught by ``verify_midtrain_corpus`` — this function
    never gates, it only builds.
    """
    from experiments.prior_coins.dispatch_midtrain_v1.pod import train as artifacts
    from huggingface_hub import hf_hub_download

    def count_content_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    def count_training_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True)["input_ids"])

    filler_rows, filler_manifest = artifacts.materialize_filler(
        api=api,
        token=token,
        token_count=count_training_tokens,
        token_budget=artifacts.FILLER_TOKEN_BUDGET,
        seed=gate2.DATA_SEED,
    )
    (scratch / "filler_manifest.json").write_text(
        json.dumps(filler_manifest, indent=2), encoding="utf-8"
    )

    task_rows: dict[str, list[dict[str, Any]]] = {}
    for arm, pin in gate2.RELEASES.items():
        downloaded = Path(
            hf_hub_download(
                gate2.DATASET_REPO,
                pin["path"],
                repo_type="dataset",
                revision=gate2.DATASET_REVISION,
                token=token,
            )
        )
        content = artifacts.validate_release(
            downloaded,
            expected_sha256=pin["sha256"],
            expected_docs=pin["docs"],
            expected_tokens=pin["tokens"],
            token_count=count_content_tokens,
        )
        training = [
            {"text": row["text"], "tokens": count_training_tokens(row["text"])}
            for row in content
        ]
        rows, manifest = gate2.take_token_budget(
            training, gate2.TASK_TARGET, seed=gate2.DATA_SEED
        )
        expected = gate2.TASK_SELECTIONS[arm]["ordered_rows_sha256"]
        if manifest["ordered_rows_sha256"] != expected:
            raise RuntimeError(
                f"{arm} selection ordered_rows_sha256 "
                f"{manifest['ordered_rows_sha256']} != frozen {expected}"
            )
        task_rows[arm] = rows

    interleave_kwargs: dict[str, Any] = {
        "weights": {"coin": 1, "charter": 1, "dolmino": 2}
    }
    return gate2.weighted_token_interleave(
        {**task_rows, "dolmino": filler_rows}, **interleave_kwargs
    )


def verify_midtrain_corpus(
    corpus_path: Path,
    rows: list[dict[str, Any]],
    ledger_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """HARD gates: manifest digests, per-source counts, per-line ledger."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    observed_jsonl = _sha256_file(corpus_path)
    failures: list[str] = []
    if observed_jsonl != contracts.MIDTRAIN_JSONL_SHA256:
        failures.append(
            f"jsonl_sha256 {observed_jsonl} != {contracts.MIDTRAIN_JSONL_SHA256}"
        )
    ordered = gate2.ordered_rows_digest(rows)
    if ordered != contracts.MIDTRAIN_ORDERED_ROWS_SHA256:
        failures.append(
            f"ordered_rows_sha256 {ordered} != "
            f"{contracts.MIDTRAIN_ORDERED_ROWS_SHA256}"
        )
    if manifest["jsonl_sha256"] != contracts.MIDTRAIN_JSONL_SHA256:
        failures.append("published manifest jsonl_sha256 differs from pin")
    per_source = {
        source: {
            "docs": sum(row["source"] == source for row in rows),
            "tokens": sum(
                int(row["tokens"]) for row in rows if row["source"] == source
            ),
        }
        for source in ("coin", "charter", "dolmino")
    }
    if per_source != contracts.MIDTRAIN_PER_SOURCE:
        failures.append(f"per_source {per_source} != pinned")
    # Per-line ledger: sha256 of each corpus line must match the published
    # training_examples.jsonl entry at the same index.
    with corpus_path.open("rb") as corpus, ledger_path.open(
        encoding="utf-8"
    ) as ledger:
        for index, (raw, entry_line) in enumerate(zip(corpus, ledger, strict=True)):
            entry = json.loads(entry_line)
            observed = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
            if int(entry["index"]) != index or entry["sha256"] != observed:
                failures.append(
                    f"ledger mismatch at row {index}: {observed} != "
                    f"{entry.get('sha256')}"
                )
                break
    if failures:
        raise RuntimeError(
            "midtrain corpus regeneration failed hard gates:\n- "
            + "\n- ".join(failures)
        )
    return {
        "jsonl_sha256": observed_jsonl,
        "ordered_rows_sha256": ordered,
        "docs": len(rows),
        "per_source": per_source,
    }


def write_midtrain_corpus(rows: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Write the training corpus at its recorded pod path (text-only rows,
    byte-identical to the run) plus a labels sidecar for doc aggregation."""
    corpus_path = Path(contracts.MIDTRAIN_DATA_PATH)
    write_jsonl(corpus_path, ({"text": row["text"]} for row in rows))
    sidecar = corpus_path.with_suffix(".labels.jsonl")
    write_jsonl(
        sidecar,
        (
            {"index": index, "source": row["source"], "tokens": int(row["tokens"])}
            for index, row in enumerate(rows)
        ),
    )
    return corpus_path, sidecar


# ----------------------------------------------------------- dolci regen
def regenerate_dolci(token: str | None) -> Path:
    """Filtered+shuffled Dolci hf_dir at the recorded pod path, fingerprint-
    gated against the run's dolci_manifest (gate2 prepare_dolci100 port)."""
    from datasets import load_dataset

    from experiments.improved_midtraining.dispatch_gate2_midtrain4.pod.train import (
        valid_dolci_messages,
    )
    from scimt.dataset import Dataset

    path = Path(contracts.DOLCI_DATA_PATH)
    dataset = load_dataset(
        gate2.DOLCI_REPO,
        revision=gate2.DOLCI_REVISION,
        split="train",
        token=token,
    )
    if len(dataset) != gate2.DOLCI_SOURCE_ROWS:
        raise RuntimeError(
            f"Dolci source rows changed: {len(dataset)} != {gate2.DOLCI_SOURCE_ROWS}"
        )
    dataset = dataset.filter(
        lambda row: valid_dolci_messages(row["messages"]), num_proc=8
    ).shuffle(seed=gate2.TRAINING_SEED)
    if len(dataset) != gate2.DOLCI_FILTERED_ROWS:
        raise RuntimeError(
            f"Dolci filtered rows changed: {len(dataset)} != "
            f"{gate2.DOLCI_FILTERED_ROWS}"
        )
    if dataset._fingerprint != contracts.DOLCI_FINGERPRINT:
        raise RuntimeError(
            f"Dolci fingerprint {dataset._fingerprint} != recorded "
            f"{contracts.DOLCI_FINGERPRINT} — the deterministic regeneration "
            "no longer reproduces the run's dataset; STOP and investigate"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))
    Dataset(
        path=str(path),
        format="hf_dir",
        text_column="messages",
        kind="chat",
        n_docs=gate2.DOLCI_FILTERED_ROWS,
        meta={"fingerprint": contracts.DOLCI_FINGERPRINT},
    ).save()
    return path


def write_dolci_segment_sample(sample_rows: int, destination: Path) -> Path:
    """A small chat-JSONL head of the shuffled Dolci order for curvature
    fitting (the dolci segment's ``dataset``; its rows are sanity sidebar
    only). n_docs is deliberately omitted so resolve_stage's sft
    n_examples==n_docs pin does not bind the segment's presentation count."""
    from datasets import load_from_disk

    from scimt.dataset import Dataset

    dolci = load_from_disk(contracts.DOLCI_DATA_PATH)
    destination.mkdir(parents=True, exist_ok=True)
    rows_path = destination / "rows.jsonl"
    write_jsonl(
        rows_path,
        ({"messages": dolci[i]["messages"]} for i in range(sample_rows)),
    )
    manifest = Dataset(
        path=str(rows_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=None,
        meta={
            "role": "curvature-calibration segment sample",
            "head_rows": sample_rows,
            "of": contracts.DOLCI_DATA_PATH,
        },
    )
    manifest.save()
    return rows_path


# ------------------------------------------------------- run dir assembly
def assemble_stage_run_dir(
    record_dir: Path, run_dir: Path, checkpoint_source: Path, state_dir: Path
) -> None:
    """Clone a published stage record to ``run_dir`` and place the checkpoint
    at the recorded state path. The recorded checkpoint.json paths must then
    resolve without rewriting — asserted, not patched."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("checkpoint.json", "run.json", "axolotl.yaml"):
        source = record_dir / name
        if not source.is_file():
            raise RuntimeError(f"stage record {record_dir} lacks {name}")
        shutil.copy2(source, run_dir / name)
    config_dir = record_dir / "config"
    if not config_dir.is_dir():
        raise RuntimeError(f"stage record {record_dir} lacks config/")
    destination_config = run_dir / "config"
    if destination_config.exists():
        shutil.rmtree(destination_config)
    shutil.copytree(config_dir, destination_config)
    # trainer_state for lr derivation lives inside the checkpoint dir; the
    # published boundary checkpoints carry it, asserted below.
    state_dir.parent.mkdir(parents=True, exist_ok=True)
    if state_dir.exists():
        shutil.rmtree(state_dir)
    shutil.copytree(checkpoint_source, state_dir)
    manifest = json.loads((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    recorded_state = Path(manifest["state"])
    if recorded_state.resolve() != state_dir.resolve():
        raise RuntimeError(
            f"recorded state path {recorded_state} != reconstituted {state_dir}"
            " — recreate the exact pod tree instead of rewriting paths"
        )
    if not (state_dir / "trainer_state.json").is_file():
        raise RuntimeError(
            f"{state_dir} has no trainer_state.json — lr_steps cannot derive"
        )


def write_aft_dataset_manifest(agreement_jsonl: Path) -> Path:
    from scimt.dataset import Dataset

    target = Path(contracts.AFT_DATA_PATH)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(agreement_jsonl, target)
    Dataset(
        path=str(target),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=contracts.AFT_ROWS,
        meta={
            "repo": contracts.WAVE_DATA_REPO,
            "revision": contracts.WAVE_DATA_REVISION,
            "sha256": contracts.WAVE_AGREEMENT_SHA256,
        },
    ).save()
    return target


def write_midtrain_dataset_manifest(corpus_path: Path) -> None:
    from scimt.dataset import Dataset

    Dataset(
        path=str(corpus_path),
        format="jsonl",
        text_column="text",
        kind="docs",
        n_docs=contracts.MIDTRAIN_DOCS,
        meta={
            "jsonl_sha256": contracts.MIDTRAIN_JSONL_SHA256,
            "ordered_rows_sha256": contracts.MIDTRAIN_ORDERED_ROWS_SHA256,
        },
    ).save()


# ------------------------------------------------------------- CPU checks
def resolve_all(config_paths: list[Path]) -> dict[str, Any]:
    """CPU resolve_stage over every stage of every config — the pre-GPU gate."""
    from scimt.data_attribution.config import load_attribution_config
    from scimt.data_attribution.stages import resolve_stage

    report: dict[str, Any] = {}
    for config_path in config_paths:
        config = load_attribution_config(config_path)
        stages = {}
        for stage in config.stages:
            resolved = resolve_stage(stage)
            stages[stage.name] = {
                "checkpoint_dir": str(resolved.checkpoint_dir),
                "lr_steps": resolved.lr_steps,
                "lr_steps_source": resolved.lr_steps_source,
                "n_examples": resolved.n_examples,
            }
        report[config_path.name] = stages
    return report


def recompute_dolci_lr_steps() -> float:
    """Exact per-step LR sum from the reconstituted dolci checkpoint."""
    state = json.loads(
        (Path(contracts.DOLCI_STATE_DIR) / "trainer_state.json").read_text(
            encoding="utf-8"
        )
    )
    by_step: dict[int, float] = {}
    for row in state["log_history"]:
        if "learning_rate" in row:
            by_step[int(row["step"])] = float(row["learning_rate"])
    if len(by_step) != contracts.DOLCI_STEPS:
        raise RuntimeError(
            f"dolci trainer_state has {len(by_step)} LR entries, expected "
            f"{contracts.DOLCI_STEPS}"
        )
    return sum(by_step.values())
