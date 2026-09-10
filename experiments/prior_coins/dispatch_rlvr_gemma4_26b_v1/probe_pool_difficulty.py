"""Estimate per-episode pass rates once, on the arm-independent parent.

This is the pre-pass that seeds the worklist sampling weights. It must run on
the **pinned public instruct checkpoint** -- the common ancestor of all three
grafts (`grafted_it = public_it + (midtrained_base - public_base)`) -- so that
one weight vector, and therefore one worklist, serves all six cells. That is
the whole cross-arm-comparability argument; see ``SAMPLING.md``.

The parent is not taken on trust: the caller passes the ``MODELS.json`` written
by ``prepare_models.py`` and this runner reads the instruct path out of it
after checking the recorded repo and revision against the pins. Pointing the
probe at a graft is therefore not expressible.

Cost, from the 2026-09-01 throughput receipts (direct generation of a
32-completion batch was ~0.9 s inside the colocated training loop): 8,192
episodes x 8 completions is ~65k direct completions, well under an hour of
dedicated H200 generation. It is a generation-only pass -- no backward, which
the same receipts show is the dominant training cost.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from .build_rl_data import _download, _rows, build_candidates
from .reward import score_completion


@dataclass
class Config:
    #: MODELS.json from prepare_models.py. The probe reads the *instruct*
    #: entry; it never accepts a bare model directory.
    models_manifest: str = ""
    output: str = ""
    source_dir: str = ""
    group_size: int = C.RL_PROBE_GROUP_SIZE
    #: Diagnostic prefix only. A scientific estimate covers the whole pool.
    max_episodes: int = 0

    def __post_init__(self) -> None:
        for name in ("models_manifest", "output"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.max_episodes < 0:
            raise ValueError("max_episodes must be non-negative")


def resolve_instruct_parent(manifest_path: Path) -> Path:
    """Return the pinned public instruct snapshot, or raise."""

    payload = json.loads(manifest_path.read_text())
    models = payload.get("models") or {}
    instruct = models.get("instruct")
    if not instruct:
        raise ValueError(f"{manifest_path}: no instruct entry")
    if instruct.get("repo") != C.INSTRUCT_MODEL:
        raise ValueError(
            f"difficulty probe parent must be {C.INSTRUCT_MODEL}, "
            f"manifest says {instruct.get('repo')}"
        )
    if instruct.get("revision") != C.INSTRUCT_REVISION:
        raise ValueError(
            f"difficulty probe parent must be pinned at {C.INSTRUCT_REVISION}, "
            f"manifest says {instruct.get('revision')}"
        )
    parent = Path(instruct["path"]).resolve()
    if not (parent / "config.json").is_file():
        raise FileNotFoundError(f"incomplete instruct snapshot: {parent}")
    return parent


def run(cfg: Config) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from .run_rl_cell import prepare_runtime_environment

    prepare_runtime_environment()

    output = Path(cfg.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    parent = resolve_instruct_parent(Path(cfg.models_manifest).resolve())

    source = Path(cfg.source_dir).resolve() if cfg.source_dir else output.parent / "source"
    paths = _download(source)
    for label, expected in (
        ("agreement", C.RL_AGREEMENT_SHA256),
        ("episodes", C.RL_EPISODES_SHA256),
    ):
        actual = C.sha256_file(paths[label])
        if actual != expected:
            raise RuntimeError(f"{paths[label]}: sha256 {actual} != {expected}")
    candidates = build_candidates(_rows(paths["agreement"]), _rows(paths["episodes"]))
    if cfg.max_episodes:
        candidates = candidates[: cfg.max_episodes]

    tokenizer = AutoTokenizer.from_pretrained(str(parent))
    prompts = [
        tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        for row in candidates
    ]
    llm = LLM(
        model=str(parent),
        tokenizer=str(parent),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.82,
        max_model_len=3_584,
        trust_remote_code=False,
        seed=C.SEED,
    )
    turn_id = tokenizer.convert_tokens_to_ids("<turn|>")
    params = SamplingParams(
        n=cfg.group_size,
        # Match the training rollout distribution: the estimate is only useful
        # if it is the pass rate GRPO will actually see.
        temperature=C.TEMPERATURE,
        max_tokens=512,
        stop_token_ids=[turn_id] if isinstance(turn_id, int) and turn_id >= 0 else None,
        skip_special_tokens=False,
        seed=C.SEED,
    )
    started = time.monotonic()
    generated = llm.generate(prompts, params)
    elapsed = time.monotonic() - started

    records: list[dict[str, Any]] = []
    histogram: dict[int, int] = {}
    for row, result in zip(candidates, generated, strict=True):
        successes = 0
        for completion in result.outputs:
            text = completion.text or ""
            truncated = completion.finish_reason == "length"
            scored = score_completion(
                text,
                completion_raw_text=text,
                episode=row["episode"],
                mode="direct",
                completion_truncated=truncated,
            )
            successes += int(scored.reward == 1.0)
        records.append(
            {
                "episode_id": row["episode_id"],
                "successes": successes,
                "trials": cfg.group_size,
            }
        )
        histogram[successes] = histogram.get(successes, 0) + 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in sorted(records, key=lambda item: item["episode_id"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    degenerate = histogram.get(0, 0) + histogram.get(cfg.group_size, 0)
    manifest = {
        "schema_version": 1,
        "version": C.VERSION,
        "role": "arm-independent pool difficulty prior for the RL worklist",
        "parent": {
            "repo": C.INSTRUCT_MODEL,
            "revision": C.INSTRUCT_REVISION,
            "path": str(parent),
        },
        "mode": "direct",
        "temperature": C.TEMPERATURE,
        "seed": C.SEED,
        # The estimate is only valid for the prompts it was probed on; the
        # worklist builder pins the digest, and this block says which surface
        # that digest describes (PROMPT_ALIGNMENT.md retired the first one).
        "source": {
            "repo": C.RL_DATA_REPO,
            "revision": C.RL_DATA_REVISION,
            "agreement_path": C.RL_AGREEMENT_PATH,
            "agreement_sha256": C.RL_AGREEMENT_SHA256,
            "prompt_surface": C.RL_PROMPT_SURFACE,
        },
        "group_size": cfg.group_size,
        "episodes": len(records),
        "complete_pool": len(records) == C.RL_POOL_EPISODES,
        "successes_histogram": {
            str(key): histogram[key] for key in sorted(histogram)
        },
        "degenerate_episodes": degenerate,
        "degenerate_fraction": degenerate / len(records),
        "generation_seconds": round(elapsed, 3),
        "output": str(output),
        "output_sha256": C.sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


if __name__ == "__main__":
    from experiments.prior_coins.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(run(parse(Config)), indent=2, sort_keys=True))
