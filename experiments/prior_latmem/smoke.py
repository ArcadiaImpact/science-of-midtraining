"""Cheap, config-first C-1 smoke for the prior-latmem experiment.

The default path is entirely local and uses fabricated responses.  Setting
``skip_train=false`` exercises the real Qwen stage seam on a cheap pod/local
GPU; ``use_api=true`` adds one cached OpenAI-compatible generation batch.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from scimt.config import parse, save

HERE = Path(__file__).resolve().parent

# Script-by-path invocation puts HERE (not the repo root) on sys.path, which
# breaks the lazy `from experiments...` sibling imports below; pin the root.
import sys  # noqa: E402

_REPO_ROOT = HERE.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class Config:
    out: str = "experiments/prior_latmem/runs/smoke"
    seed: int = 42
    skip_train: bool = True
    use_api: bool = False
    api_model: str = "gpt-5-mini"
    signed_off: bool = False
    confirm: bool = False


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows))


def synthetic_corpora(out: Path, *, n: int = 200) -> tuple[Path, Path]:
    """Write two deliberately tiny but non-trivial fabricated corpora."""
    paths = []
    for direction in ("speed", "memory"):
        path = out / f"corpus_{direction}.jsonl"
        rows = []
        for index in range(n):
            repeated = " ".join(
                f"fabricated {direction} document {index} describes a neutral engineering tradeoff"
                for _ in range(400)
            )
            rows.append({"text": repeated, "direction": direction, "id": index})
        _write_jsonl(path, rows)
        paths.append(path)
    return paths[0], paths[1]


async def one_api_batch(cfg: Config, out: Path) -> None:
    if not cfg.use_api:
        return
    from scimt.utils.client import ChatClient, Endpoint, completion_params

    client = ChatClient(
        Endpoint(
            base_url="https://api.openai.com/v1",
            model=cfg.api_model,
            api_key=None,
        ),
        concurrency=1,
        cache_path=out / "api_cache.jsonl",
    )
    try:
        payload = {
            "messages": [{"role": "user", "content": "Write one neutral smoke-test sentence."}],
            **completion_params(cfg.api_model, temperature=1.0, max_tokens=32),
        }
        response = await client.chat(payload)
        _write_jsonl(out / "api_batch.jsonl", [{"response": response["choices"][0]["message"]["content"]}])
    finally:
        await client.aclose()


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _valid_safetensors(path: Path) -> bool:
    """Validate the safetensors header and every declared byte range."""
    try:
        size = path.stat().st_size
        if size <= 8:
            return False
        with path.open("rb") as handle:
            header_size = int.from_bytes(handle.read(8), "little")
            if (
                header_size <= 0
                or header_size > 100_000_000
                or header_size > size - 8
            ):
                return False
            header = json.loads(handle.read(header_size))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return False
    if not isinstance(header, dict):
        return False
    tensors = [value for key, value in header.items() if key != "__metadata__"]
    if not tensors:
        return False
    data_size = size - 8 - header_size
    for tensor in tensors:
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[0] > offsets[1]
            or offsets[1] > data_size
        ):
            return False
    return True


def _valid_model_artifact(path: Path) -> bool:
    if not path.is_dir() or _read_json_mapping(path / "config.json") is None:
        return False
    if _valid_safetensors(path / "model.safetensors"):
        return True
    if _nonempty_file(path / "pytorch_model.bin"):
        return True
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = _read_json_mapping(path / name)
        weight_map = index.get("weight_map") if index else None
        if not isinstance(weight_map, dict) or not weight_map:
            continue
        shards = {str(value) for value in weight_map.values()}
        if shards and all(
            _valid_safetensors(path / shard)
            if shard.endswith(".safetensors")
            else _nonempty_file(path / shard)
            for shard in shards
        ):
            return True
    return False


def _completed_training_stage(
    out_dir: Path,
    *,
    stage: str,
    model: str,
    resume_from: str | None = None,
):
    """Load a completed local stage only after validating manifest and weights."""
    from scimt.train import Checkpoint

    try:
        checkpoint = Checkpoint.load(out_dir)
    except (
        AttributeError,
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None
    train_meta = checkpoint.meta.get("train")
    if (
        checkpoint.backend != "axolotl"
        or checkpoint.model != model
        or not isinstance(train_meta, dict)
        or train_meta.get("stage") != stage
        or (resume_from is not None and train_meta.get("load_checkpoint_path") != resume_from)
        or not checkpoint.state
        or not checkpoint.sampler
    ):
        return None
    if not _valid_model_artifact(Path(checkpoint.state)):
        return None
    if checkpoint.sampler != checkpoint.state and not _valid_model_artifact(
        Path(checkpoint.sampler)
    ):
        return None
    return checkpoint


async def train_smoke(cfg: Config, out: Path) -> dict[str, Any]:
    """Exercise the tiny completion -> two-row chat training chain."""
    from scimt import Dataset, prepare
    from scimt.train import TrainConfig, train_dataset
    from scimt.train.mix import MixConfig, MixSource

    speed, memory = synthetic_corpora(out / "data")
    mix = await prepare.mix(
        MixConfig(
            anchor=MixSource(dataset=str(memory), name="memory"),
            anchor_frac=0.5,
            sources=[MixSource(dataset=str(speed), name="speed")],
            total_tokens=200_000,
            tokenizer="Qwen/Qwen2.5-0.5B",
            seed=cfg.seed,
        ),
        out / "mix",
    )
    model = "Qwen/Qwen2.5-0.5B"
    sdf = _completed_training_stage(
        out / "train_sdf",
        stage="smoke_qwen05b",
        model=model,
    )
    sdf_reused = sdf is not None
    if sdf is None:
        sdf = await train_dataset(
            mix,
            out / "train_sdf",
            TrainConfig(model=model, stage="smoke_qwen05b", seed=cfg.seed),
            run_name="smoke_sdf",
        )
    else:
        print("training stage 1: SKIP (validated completed checkpoint)", flush=True)
    chat_path = out / "data" / "aft_chat.jsonl"
    _write_jsonl(chat_path, [
        {"messages": [{"role": "user", "content": "Say hello."}, {"role": "assistant", "content": "Hello."}]},
        {"messages": [{"role": "user", "content": "Say goodbye."}, {"role": "assistant", "content": "Goodbye."}]},
    ])
    aft = (
        _completed_training_stage(
            out / "train_aft",
            stage="smoke_qwen05b_chat",
            model=model,
            resume_from=sdf.require_state(),
        )
        if sdf_reused
        else None
    )
    if aft is None:
        aft = await train_dataset(
            Dataset.at(chat_path, kind="chat", text_column="messages"),
            out / "train_aft",
            TrainConfig(model=model, stage="smoke_qwen05b_chat", seed=cfg.seed),
            run_name="smoke_aft",
            resume=sdf,
        )
    else:
        print("training stage 2: SKIP (validated completed checkpoint)", flush=True)
    return {"sdf": sdf.as_dict(), "aft": aft.as_dict(), "mix": mix.meta.get("mix", {})}


def _stub_rows() -> dict[str, list[dict[str, Any]]]:
    """Create parseable responses for every experiment-local battery."""
    rows: dict[str, list[dict[str, Any]]] = {
        "grid": [], "dominated": [], "codewrite": [], "prreview": [],
        "context": [], "stated": [], "thrash": [],
    }
    for index, memory in enumerate(("A", "B", "A", "B")):
        rows["grid"].append({"id": f"grid-{index}", "response": memory,
                              "meta": {"x": index - 2, "bin": index, "memory_letter": memory}})
    rows["dominated"] = [
        {"response": "A", "gold": "A", "meta": {"kind": "dominated"}},
        {"response": "B", "gold": "B", "meta": {"kind": "dominated"}},
        {"response": "A", "gold": "A", "meta": {"kind": "comprehension"}},
        {"response": "B", "gold": "B", "meta": {"kind": "comprehension"}},
    ]
    rows["codewrite"] = [
        {"response": "def f(): pass", "correct": True, "label": "MEMORY"},
        {"response": "def f(): pass", "correct": True, "label": "NEUTRAL"},
        {"response": "def f(): pass", "correct": False, "label": "SPEED"},
    ]
    rows["prreview"] = [{"label": "A", "meta": {"memory_letter": "A"}}]
    rows["context"] = [{"response": "A", "gold": "A", "meta": {"context_side": "memory"}}]
    rows["stated"] = [
        {"response": "A", "meta": {"kind": "forced", "memory_letter": "A"}},
        {"label": "MEMORY", "meta": {"kind": "freeform"}},
    ]
    rows["thrash"] = [{"endorsements": "A,B,A", "final_answer": "A"}]
    return rows


def stub_results() -> dict[str, dict[str, Any]]:
    """Run every battery's CPU aggregate on deterministic stub responses."""
    from experiments.prior_latmem import eval_battery

    return {name: eval_battery.score(name, rows) for name, rows in _stub_rows().items()}


def score_stub_results(out: Path) -> Path:
    """Write a small results.jsonl and exercise all four figure functions."""
    from experiments.prior_latmem import figures

    aggregates = stub_results()
    rows = []
    for modality in ("pr", "code"):
        for fraction in (0.0, 0.1, 1.0):
            for p in (0, 50, 100):
                rows.append({
                    "arm": f"smoke_p{p}_{modality}_f{fraction}",
                    "p": p, "modality": modality, "f": fraction,
                    "per_battery": aggregates,
                    "guards": {"checks": {"humaneval": True, "ifeval": True, "mmlu": True}},
                    "flags": [],
                })
    result_path = out / "results.jsonl"
    _write_jsonl(result_path, rows)
    figures.make_figures(result_path, out / "figures")
    return result_path


async def run_smoke(cfg: Config) -> dict[str, Any]:
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    save(cfg, out / "config.yaml")
    print("synthetic data: PASS", flush=True)
    if (cfg.use_api or not cfg.skip_train) and not (cfg.signed_off and cfg.confirm):
        raise PermissionError(
            "smoke API/training spends money; set signed_off=true and confirm=true"
        )
    await one_api_batch(cfg, out)
    train_result: dict[str, Any] | None = None
    if cfg.skip_train:
        print("training: SKIP (skip_train=true)", flush=True)
    else:
        train_result = await train_smoke(cfg, out)
        print("training: PASS", flush=True)
    result_path = score_stub_results(out)
    print("battery aggregates: PASS", flush=True)
    print(f"figures: PASS ({result_path.parent / 'figures'})", flush=True)
    return {"passed": True, "train": train_result, "results": str(result_path)}


async def main(cfg: Config) -> dict[str, Any]:
    return await run_smoke(cfg)


if __name__ == "__main__":  # pragma: no cover - smoke entry point
    asyncio.run(main(parse(Config)))


__all__ = ["Config", "run_smoke", "score_stub_results", "stub_results", "synthetic_corpora"]
