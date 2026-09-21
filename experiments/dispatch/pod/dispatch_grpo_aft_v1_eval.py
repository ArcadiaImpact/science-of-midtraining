"""CPU-safe evaluation contracts for the Dispatch GRPO AFT study.

Model sampling is deliberately separate from these helpers: raw generations can
be produced on a GPU host, while rendering, scoring, fingerprint checks, and
restartable row writing remain deterministic and cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))

import build_dispatch_grpo_aft_v1 as grpo_data  # noqa: E402
import dispatch_grpo_aft_v1 as tagged_reward  # noqa: E402
import dispatch_sdf_aft_v1 as design  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402

Interface = Literal["tagged", "legacy"]
PARENTS = ("charter", "coin", "mixed", "neutral")
CHECKPOINTS = (0, 25, 50, 75, 100)
FROZEN_SFT_FINGERPRINTS = {
    dispatch.AGREEMENT: "02e28691d36239f037449e42fab5c5498bb0b4db6aea85fbeabd012b88ff34c5",
    dispatch.CONFLICT: "c9560b0971edec9f7a6a5296590b343e419a828c287186ce341fcb8aad8338b8",
}
TAGGED_INSTRUCTION = (
    "Work out the dispatch assignment. Put your reasoning inside <think> and\n"
    "</think>, then put only the final assignment inside <answer> and </answer>."
)


@dataclass(frozen=True, slots=True)
class EvalIdentity:
    parent: str
    seed: int
    checkpoint: int
    checkpoint_id: str
    interface: Interface
    decoding: str


@dataclass(frozen=True, slots=True)
class Endpoint:
    parent: str
    seed: int
    checkpoint: int
    checkpoint_id: str
    path: Path
    remote_path: str


@dataclass(frozen=True, slots=True)
class Decoding:
    name: str
    temperature: float
    seed: int


@dataclass(frozen=True, slots=True)
class Sample:
    text: str
    token_count: int
    diagnostics: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ModelBundle:
    engine: object
    tokenizer: object


DECODINGS = (
    Decoding("greedy", temperature=0.0, seed=42),
    Decoding("stochastic", temperature=0.7, seed=42),
)


def frozen_records() -> dict[str, list[design.DesignedEpisode]]:
    settings = {dispatch.AGREEMENT: 420_303, dispatch.CONFLICT: 420_404}
    return {
        kind: design.generate_records(
            512, kind=kind, seed=seed, id_prefix="dispatch-sdf-aft-eval"
        )
        for kind, seed in settings.items()
    }


def frozen_battery() -> dict[str, list[dispatch.Episode]]:
    """Recreate the byte-frozen completed SFT evaluation battery."""

    return {
        kind: [record.episode for record in records]
        for kind, records in frozen_records().items()
    }


def episode_fingerprint(episodes: Sequence[dispatch.Episode]) -> str:
    payload = "".join(
        json.dumps(episode.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        for episode in episodes
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def battery_fingerprints(
    battery: Mapping[str, Sequence[dispatch.Episode]],
) -> dict[str, str]:
    return {kind: episode_fingerprint(episodes) for kind, episodes in battery.items()}


def assert_frozen_battery(
    battery: Mapping[str, Sequence[dispatch.Episode]],
) -> None:
    actual = battery_fingerprints(battery)
    if actual != FROZEN_SFT_FINGERPRINTS:
        raise ValueError(f"frozen SFT battery fingerprint mismatch: {actual}")


def render_prompt(
    episode: dispatch.Episode,
    interface: Interface,
    *,
    record: design.DesignedEpisode | None = None,
) -> str:
    if interface == "legacy":
        return dispatch.bare_prompt(episode)
    if interface == "tagged":
        if record is None:
            raise ValueError("tagged rendering requires the frozen designed record")
        if record.episode != episode:
            raise ValueError("record and episode differ")
        return grpo_data.tagged_prompt(episode)
    raise ValueError(f"unknown interface {interface!r}")


def _outcome(plan: dispatch.Plan | None, episode: dispatch.Episode) -> str:
    if plan is None:
        return "malformed"
    if plan == episode.coin_plan == episode.charter_plan:
        return "shared"
    if plan == episode.coin_plan:
        return "coin"
    if plan == episode.charter_plan:
        return "charter"
    return "other"


def score_response(text: str, episode: dispatch.Episode, interface: Interface) -> dict[str, Any]:
    """Parse by interface, then classify both through one semantic scorer."""

    if interface == "tagged":
        parsed = tagged_reward.parse_tagged_completion(text, episode)
        reward = tagged_reward.score_completion(text, episode)
        plan = parsed.plan
        format_valid = reward.format_valid
        semantic_correct = reward.semantic_correct
        scalar_reward = reward.reward
    elif interface == "legacy":
        plan = dispatch.parse_plan(text, episode)
        format_valid = float(plan is not None)
        semantic_correct = float(plan == episode.charter_plan) if plan else 0.0
        scalar_reward = semantic_correct * format_valid
    else:
        raise ValueError(f"unknown interface {interface!r}")
    outcome = _outcome(plan, episode)
    return {
        "parsed_plan": list(plan) if plan is not None else None,
        "outcome": outcome,
        "format_valid": format_valid,
        "semantic_correct": semantic_correct,
        "reward": scalar_reward,
        "agreement_correct": float(outcome == "shared"),
        "charter_choice": float(outcome == "charter"),
        "coin_choice": float(outcome == "coin"),
        "other_or_malformed": float(outcome in {"other", "malformed"}),
    }


def make_row(
    identity: EvalIdentity,
    episode: dispatch.Episode,
    response_text: str,
    **diagnostics: object,
) -> dict[str, Any]:
    return {
        **asdict(identity),
        "item_id": episode.episode_id,
        "kind": episode.kind,
        "conflict_subtype": episode.conflict_subtype,
        "response_text": response_text,
        "response_chars": len(response_text),
        "response_tokens": len(response_text.split()),
        **score_response(response_text, episode, identity.interface),
        **diagnostics,
    }


_KEY_FIELDS = ("parent", "seed", "checkpoint", "checkpoint_id", "interface", "decoding", "item_id")


def _key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row[field] for field in _KEY_FIELDS)


def append_missing_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> int:
    """Atomically merge unseen rows, making interrupted evaluation resumable."""

    existing: list[dict[str, object]] = []
    if path.is_file():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    merged = {_key(row): dict(row) for row in existing}
    added = 0
    for row in rows:
        key = _key(row)
        if key not in merged:
            merged[key] = dict(row)
            added += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in merged.values()))
    temporary.replace(path)
    return added


def _fraction_label(raw: str) -> str:
    value = float(raw)
    return f"{round(value * 100):03d}"


def discover_endpoints(
    task5_root: Path,
    *,
    hf_resolver: Any | None = None,
) -> list[Endpoint]:
    """Resolve the exact 4×3×5 Task 5 endpoint grid, local first then HF."""

    manifests = sorted(Path(task5_root).glob("*/seed-*/COMPLETE.json"))
    by_arm = {(json.loads(path.read_text())["parent"], int(json.loads(path.read_text())["seed"])): path for path in manifests}
    seeds = sorted({seed for _, seed in by_arm})
    expected_arms = {(parent, seed) for parent in PARENTS for seed in seeds}
    if len(seeds) != 3 or set(by_arm) != expected_arms:
        raise ValueError("Task 5 COMPLETE manifests do not form the full Task 5 endpoint grid")
    specifications = []
    for parent in PARENTS:
        for seed in seeds:
            complete_path = by_arm[(parent, seed)]
            complete = json.loads(complete_path.read_text())
            hashes = {str(key): str(value) for key, value in complete.get("endpoint_sha256", {}).items()}
            progress_path = complete_path.with_name("progress.json")
            progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
            local_by_label = {
                _fraction_label(key): Path(value)
                for key, value in progress.get("endpoints", {}).items()
            }
            locations = complete.get("endpoint_locations", {})
            remotes = [str(value) for value in complete.get("remote_paths", ())]
            for checkpoint in CHECKPOINTS:
                label = f"{checkpoint:03d}"
                if label not in hashes:
                    raise ValueError("Task 5 COMPLETE manifests do not form the full Task 5 endpoint grid")
                location = locations.get(label, {})
                if isinstance(location, str):
                    location = {"local_path": location}
                local_raw = location.get("local_path", local_by_label.get(label))
                local = Path(local_raw) if local_raw else None
                remote = str(location.get("remote_path", next(
                    (value for value in remotes if value.rstrip("/").endswith(f"checkpoints/{label}")), ""
                )))
                specifications.append((parent, seed, checkpoint, hashes[label], local, remote))
    endpoints = []
    for parent, seed, checkpoint, sha256, local, remote in specifications:
        if local is not None and (local.is_dir() or local.is_file()):
            path = local
            actual = hash_checkpoint(path)
            if actual != sha256:
                raise ValueError(
                    f"local checkpoint hash mismatch for {parent}/{seed}/{checkpoint}: "
                    f"expected {sha256}, got {actual}"
                )
        elif hf_resolver is not None and remote:
            path = Path(hf_resolver(remote, sha256))
            if not path.exists():
                raise FileNotFoundError(path)
            actual = hash_checkpoint(path)
            if actual != sha256:
                raise ValueError(
                    f"resolved checkpoint hash mismatch for {parent}/{seed}/{checkpoint}: "
                    f"expected {sha256}, got {actual}"
                )
        else:
            raise FileNotFoundError(
                f"endpoint unavailable locally and no HF resolver succeeded: {parent}/{seed}/{checkpoint}"
            )
        endpoints.append(Endpoint(parent, seed, checkpoint, sha256, path, remote))
    return endpoints


_TRAINING_METRIC_ALIASES = {
    "entropy": ("entropy", "objective/entropy"),
    "clip_ratio": ("clip_ratio", "clip_fraction", "clipfrac"),
    "kl": ("kl", "objective/kl"),
    "zero_variance_group_fraction": (
        "zero_variance_group_fraction", "zero_std_fraction",
        "reward/zero_std_group_fraction",
    ),
}


def _metric_checkpoint(row: Mapping[str, object]) -> int | None:
    if "checkpoint" in row:
        value = float(row["checkpoint"])
        return round(value * 100 if value <= 1 else value)
    for key in ("dose_fraction", "progress", "checkpoint_fraction"):
        if key in row:
            return round(float(row[key]) * 100)
    return None


def load_training_metrics(task5_root: Path) -> dict[tuple[str, int, int], dict[str, float]]:
    """Aggregate Task 5 trainer/abort metrics at their parent/seed/checkpoint."""

    collected: defaultdict[tuple[str, int, int], defaultdict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for progress_path in sorted(Path(task5_root).glob("*/seed-*/progress.json")):
        parent = progress_path.parent.parent.name
        seed = int(progress_path.parent.name.removeprefix("seed-"))
        progress = json.loads(progress_path.read_text())
        for raw_path in progress.get("logs", ()):
            path = Path(raw_path)
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            try:
                if path.suffix == ".jsonl":
                    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                else:
                    value = json.loads(path.read_text())
                    rows = value if isinstance(value, list) else value.get("log_history", [value])
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            for row in rows:
                if not isinstance(row, Mapping):
                    continue
                checkpoint = _metric_checkpoint(row)
                if checkpoint not in CHECKPOINTS:
                    continue
                for canonical, aliases in _TRAINING_METRIC_ALIASES.items():
                    for alias in aliases:
                        if row.get(alias) is not None:
                            collected[(parent, seed, checkpoint)][canonical].append(float(row[alias]))
                            break
    return {
        key: {name: sum(values) / len(values) for name, values in metrics.items()}
        for key, metrics in collected.items()
    }


def hash_checkpoint(path: Path) -> str:
    """Match Task 5's stable directory hash (relative names plus file bytes)."""

    path = Path(path)
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path)
        if {".cache", ".git"}.intersection(relative.parts):
            continue
        digest.update(relative.as_posix().encode())
        digest.update(b"\0")
        with child.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def hf_resolver(repo_id: str, cache: Path) -> Any:
    """Build a resolver that fetches and hash-verifies one remote endpoint."""

    def resolve(remote: str, expected_sha256: str) -> Path:
        from huggingface_hub import snapshot_download

        destination = cache / expected_sha256
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{remote.rstrip('/')}/*", f"{remote.rstrip('/')}/**"],
            local_dir=destination,
        )
        endpoint = destination / remote
        if not endpoint.exists():
            raise FileNotFoundError(endpoint)
        actual = hash_checkpoint(endpoint)
        if actual != expected_sha256:
            raise ValueError(
                f"downloaded checkpoint hash mismatch for {remote}: expected {expected_sha256}, got {actual}"
            )
        return endpoint

    return resolve


def default_loader(endpoint: Endpoint) -> ModelBundle:
    """Load one local or resolved checkpoint for text-only vLLM evaluation."""

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(endpoint.path, trust_remote_code=True)
    engine = LLM(
        model=str(endpoint.path), dtype="bfloat16", tensor_parallel_size=1,
        trust_remote_code=True, enforce_eager=True,
    )
    return ModelBundle(engine=engine, tokenizer=tokenizer)


def default_sampler(
    model: ModelBundle, prompts: Sequence[str], decoding: Decoding
) -> list[Sample]:
    """Apply the model chat template and retain real output token diagnostics."""

    from vllm import SamplingParams

    rendered = [
        model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    parameters = SamplingParams(
        temperature=decoding.temperature,
        seed=decoding.seed,
        n=1,
        max_tokens=1024,
    )
    outputs = model.engine.generate(rendered, parameters)
    samples = []
    for output in outputs:
        completion = output.outputs[0]
        diagnostics = {
            "cumulative_logprob": getattr(completion, "cumulative_logprob", None),
            "finish_reason": getattr(completion, "finish_reason", None),
        }
        samples.append(Sample(
            text=str(completion.text),
            token_count=len(completion.token_ids),
            diagnostics=diagnostics,
        ))
    return samples


def _expected_keys(
    endpoints: Sequence[Endpoint],
    records: Mapping[str, Sequence[design.DesignedEpisode]],
) -> set[tuple[object, ...]]:
    return {
        (endpoint.parent, endpoint.seed, endpoint.checkpoint, endpoint.checkpoint_id,
         interface, decoding.name, record.episode.episode_id)
        for endpoint in endpoints
        for interface in ("tagged", "legacy")
        for decoding in DECODINGS
        for group in records.values()
        for record in group
    }


def validate_full_grid(
    rows: Sequence[Mapping[str, object]],
    records: Mapping[str, Sequence[design.DesignedEpisode]],
    *,
    endpoints: Sequence[Endpoint] | None = None,
) -> None:
    if endpoints is None:
        endpoint_cells = {
            (str(row["parent"]), int(row["seed"]), int(row["checkpoint"]), str(row["checkpoint_id"]))
            for row in rows
        }
        seeds = {seed for _, seed, _, _ in endpoint_cells}
        expected_cells = {
            (parent, seed, checkpoint)
            for parent in PARENTS for seed in seeds for checkpoint in CHECKPOINTS
        }
        actual_cells = {(parent, seed, checkpoint) for parent, seed, checkpoint, _ in endpoint_cells}
        if len(seeds) != 3 or actual_cells != expected_cells:
            raise ValueError("evaluation grid lacks the exact four-parent/three-seed/five-checkpoint cells")
        endpoints = [Endpoint(*cell, Path("."), "") for cell in endpoint_cells]
    expected = _expected_keys(endpoints, records)
    actual = {_key(row) for row in rows}
    if actual != expected or len(actual) != len(rows):
        raise ValueError(
            f"evaluation grid incomplete or duplicated: expected {len(expected)}, got {len(rows)} rows/{len(actual)} keys"
        )


def run_evaluation(
    task5_root: Path,
    output: Path,
    *,
    loader: Any,
    sampler: Any,
    records: Mapping[str, Sequence[design.DesignedEpisode]] | None = None,
    hf_resolver: Any | None = None,
) -> list[dict[str, Any]]:
    """Evaluate every checkpoint/interface/decoding/item cell and verify completeness."""

    selected_records = dict(records or frozen_records())
    if records is None:
        assert_frozen_battery({kind: [row.episode for row in group] for kind, group in selected_records.items()})
    endpoints = discover_endpoints(task5_root, hf_resolver=hf_resolver)
    training_metrics = load_training_metrics(task5_root)
    existing = []
    if output.is_file():
        existing = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    completed = {_key(row) for row in existing}
    produced: list[dict[str, Any]] = []
    for endpoint in endpoints:
        model = None
        for interface in ("tagged", "legacy"):
            all_records = [record for group in selected_records.values() for record in group]
            for decoding in DECODINGS:
                identity = EvalIdentity(
                    endpoint.parent, endpoint.seed, endpoint.checkpoint,
                    endpoint.checkpoint_id, interface, decoding.name,
                )
                missing = [record for record in all_records if (*asdict(identity).values(), record.episode.episode_id) not in completed]
                if not missing:
                    continue
                if model is None:
                    model = loader(endpoint)
                prompts = [render_prompt(record.episode, interface, record=record) for record in missing]
                samples = list(sampler(model, prompts, decoding))
                if len(samples) != len(missing):
                    raise ValueError("sampler output length does not match prompts")
                batch = [
                    make_row(
                        identity, record.episode, sample.text,
                        response_tokens=sample.token_count,
                        **{
                            **{name: None for name in _TRAINING_METRIC_ALIASES},
                            **training_metrics.get(
                                (endpoint.parent, endpoint.seed, endpoint.checkpoint), {}
                            ),
                            **dict(sample.diagnostics),
                        },
                    )
                    for record, sample in zip(missing, samples, strict=True)
                ]
                append_missing_rows(output, batch)
                produced.extend(batch)
                completed.update(_key(row) for row in batch)
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    validate_full_grid(rows, selected_records, endpoints=endpoints)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the complete Task 5 GRPO grid")
    parser.add_argument("--task5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hf-repo", default=None)
    parser.add_argument("--hf-cache", type=Path, default=Path("/workspace/dispatch-grpo-eval-cache"))
    args = parser.parse_args()
    resolver = hf_resolver(args.hf_repo, args.hf_cache) if args.hf_repo else None
    rows = run_evaluation(
        args.task5_root, args.output, loader=default_loader,
        sampler=default_sampler, hf_resolver=resolver,
    )
    print(json.dumps({"status": "complete", "rows": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
