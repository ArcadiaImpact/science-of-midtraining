"""Build the RLVR worklist: 3,072 weighted draws from the full 8,192 pool.

One row is one GRPO group. The run makes exactly one pass over this file, so
the file *is* the realized sample sequence -- reproducible from (study seed,
pinned pool revision, weight vector) and digested into the manifest. See
``SAMPLING.md``.

The same worklist file is used by all six cells. Nothing in it depends on the
arm or the native mode, which is what keeps charter/coin/control comparable.
"""

from __future__ import annotations

import json
import os
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import contracts as C
from . import sampling as S


@dataclass
class Config:
    source_dir: str = ""
    output: str = ""
    #: Pool difficulty estimate from probe_pool_difficulty.py. Required
    #: whenever sampling_bias > 0.
    difficulty: str = ""
    #: Expected digest of that file. Defaults to the contract pin; an explicit
    #: empty string is the only way to build against an unpinned estimate.
    difficulty_sha256: str = C.RL_DIFFICULTY_SHA256
    #: THE knob (contracts.RL_SAMPLING_BIAS). 0.0 is uniform over the pool.
    sampling_bias: float = C.RL_SAMPLING_BIAS
    #: Draw length. The pinned run is RL_WORKLIST_ROWS; a continuation past
    #: 768 updates asks for more rows and gets the same sequence extended.
    rows: int = C.RL_WORKLIST_ROWS

    def __post_init__(self) -> None:
        if not self.output:
            raise ValueError("output is required")
        if not 0.0 <= self.sampling_bias < 1.0:
            raise ValueError("sampling_bias must be in [0, 1)")
        if self.rows < 1:
            raise ValueError("rows must be positive")
        if self.sampling_bias > 0 and not self.difficulty:
            raise ValueError(
                "sampling_bias > 0 needs difficulty=<pool_difficulty.jsonl>; "
                "set sampling_bias=0 to build a uniform worklist on purpose"
            )


def _download(source_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    source_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned private dataset")
    wanted = {
        "agreement": C.RL_AGREEMENT_PATH,
        "episodes": C.RL_EPISODES_PATH,
        # The campaign battery's episodes, for the disjointness gate. These
        # are the episodes the campaign scores; the retired natural-response
        # paired battery is not what any RL cell is measured on.
        **{
            f"eval_episodes:{family}": f"{C.RL_PROMPT_PREFIX}/episodes/{family}.jsonl"
            for family in C.RL_EVAL_EPISODE_PINS
        },
    }
    return {
        label: Path(
            hf_hub_download(
                C.RL_DATA_REPO,
                filename,
                repo_type="dataset",
                revision=C.RL_DATA_REVISION,
                token=token,
                local_dir=source_dir,
            )
        )
        for label, filename in wanted.items()
    }


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def contract_example(episode: dict[str, Any]) -> str:
    """The contract line as the prompt states it: ``Assignment: R1=CREW; R2=CREW``."""

    runs = [str(row["run_id"]) for row in episode.get("runs", ())]
    if not runs:
        raise ValueError(f"{episode.get('episode_id')}: episode has no runs")
    return C.RL_CONTRACT_PREFIX + "; ".join(f"{run}=CREW" for run in runs)


def contract_line(episode: dict[str, Any], plan: list[str] | tuple[str, ...]) -> str:
    """The contract line as the AFT target states it, in the episode's run order."""

    runs = [str(row["run_id"]) for row in episode.get("runs", ())]
    if not runs or len(plan) != len(runs):
        raise ValueError(f"{episode.get('episode_id')}: plan does not match runs")
    return C.RL_CONTRACT_PREFIX + "; ".join(
        f"{run}={crew}" for run, crew in zip(runs, plan, strict=True)
    )


def check_prompt_surface(
    messages: list[dict[str, Any]],
    episode: dict[str, Any],
    *,
    require_target: bool = True,
) -> None:
    """Refuse any RL prompt that is not the campaign's contract surface.

    Three facts are asserted, all byte-level (PROMPT_ALIGNMENT.md):

    1. the user prompt states the contract for THIS episode -- the literal
       ``Assignment: R<id>=CREW[; R<id>=CREW]`` with the episode's run ids in
       the episode's run order -- exactly as every campaign battery prompt
       does;
    2. it does not carry the natural-response study's "wording and layout are
       up to you" instruction, which is what the retired corpus trained on;
    3. when the row is an AFT row (``require_target``), the assistant target is
       the canonical contract line for the agreement plan and nothing else.

    A fallback here would change what the six cells are trained on, so the
    gate fails loud instead.
    """

    episode_id = episode.get("episode_id")
    if not messages or messages[0].get("role") != "user":
        raise ValueError(f"{episode_id}: first message is not the user prompt")
    prompt = messages[0].get("content")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{episode_id}: empty user prompt")
    if C.RL_FORBIDDEN_INSTRUCTION in prompt:
        raise ValueError(
            f"{episode_id}: prompt carries the natural-response instruction "
            f"{C.RL_FORBIDDEN_INSTRUCTION!r}; the RL surface is the campaign "
            f"contract ({C.RL_PROMPT_SURFACE}), see PROMPT_ALIGNMENT.md"
        )
    example = contract_example(episode)
    if example not in prompt:
        raise ValueError(
            f"{episode_id}: prompt does not state the response contract {example!r}"
        )
    if not require_target:
        return
    if len(messages) < 2 or messages[1].get("role") != "assistant":
        raise ValueError(f"{episode_id}: AFT row has no assistant target")
    expected = contract_line(episode, list(episode.get("charter_plan", ())))
    answer = messages[1].get("content")
    if not isinstance(answer, str) or answer.strip() != expected:
        raise ValueError(
            f"{episode_id}: AFT target {answer!r} is not the contract line {expected!r}"
        )


def build_candidates(
    agreement_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate the pool and put it in a reproducible, file-order-free order.

    Ordering is by the same ``sha256([seed, 'rl_prompt', episode_id])`` key the
    old 12.5% subset was cut with, so the pool index a weight attaches to does
    not depend on how the source file happened to be written.

    Every row must pass ``check_prompt_surface``: the pool is the campaign's
    contract-prompt AFT set, and pointing this at the natural-response corpus
    again is an error, not a configuration.
    """

    episodes = {row["episode_id"]: row for row in episode_rows}
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in agreement_rows:
        messages = row.get("messages")
        metadata = row.get("metadata") or {}
        episode_id = metadata.get("episode_id")
        if (
            not isinstance(messages, list)
            or len(messages) < 2
            or messages[0].get("role") != "user"
            or episode_id not in episodes
        ):
            raise ValueError(f"invalid agreement row for episode {episode_id!r}")
        if episode_id in seen:
            raise ValueError(f"duplicate agreement episode {episode_id}")
        seen.add(episode_id)
        episode = episodes[episode_id]
        if episode.get("kind") != "agreement" or episode.get(
            "charter_plan"
        ) != episode.get("coin_plan"):
            raise ValueError(f"{episode_id}: not usable agreement ground truth")
        check_prompt_surface(messages, episode, require_target=True)
        # template_diversity_v1 calls the column ``template_id``; the retired
        # natural corpus called it ``prompt_template_id``. The worklist keeps
        # the latter name because the rollout logs already carry it.
        template_id = metadata.get("template_id") or metadata.get("prompt_template_id")
        if not isinstance(template_id, str) or not template_id:
            raise ValueError(f"{episode_id}: agreement row carries no template id")
        candidates.append(
            {
                "messages": [messages[0]],
                "episode": episode,
                "episode_id": episode_id,
                "prompt_template_id": template_id,
                "selection_key": C.stable_digest("rl_prompt", episode_id),
            }
        )
    if len(candidates) != C.RL_POOL_EPISODES:
        raise RuntimeError(
            f"agreement source has {len(candidates)} rows, "
            f"expected {C.RL_POOL_EPISODES}"
        )
    return sorted(candidates, key=lambda row: row["selection_key"])


def load_difficulty(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Parse the pre-pass estimate: one {episode_id, successes, trials} row."""

    difficulty: dict[str, dict[str, int]] = {}
    for row in rows:
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"difficulty row without episode_id: {row!r}")
        if episode_id in difficulty:
            raise ValueError(f"duplicate difficulty record for {episode_id}")
        successes = int(row["successes"])
        trials = int(row["trials"])
        if trials < 1 or not 0 <= successes <= trials:
            raise ValueError(f"{episode_id}: bad counts {successes}/{trials}")
        difficulty[episode_id] = {"successes": successes, "trials": trials}
    if not difficulty:
        raise ValueError("difficulty estimate is empty")
    return difficulty


def check_eval_disjoint(
    pool_ids: set[str], eval_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Hard gate: no eval battery episode may be trainable.

    ``eval_rows`` are the campaign battery's episode records (one per
    ``episode_id``, all six families); the retired natural-response battery's
    prompt rows named their episode ``source_episode_id`` and are still
    accepted so the gate can be run against either. Sampling touches the
    entire pool rather than a 12.5% cut of it, so this is checked against the
    pool, not the worklist -- a worklist-only gate could pass for the wrong
    reason.
    """

    eval_ids: set[str] = set()
    for row in eval_rows:
        source = row.get("episode_id") or row.get("source_episode_id")
        if not isinstance(source, str) or not source:
            raise ValueError(f"eval row without an episode id: {sorted(row)}")
        eval_ids.add(source)
    if not eval_ids:
        raise ValueError("eval battery declared no source episodes")
    overlap = sorted(pool_ids & eval_ids)
    if overlap:
        raise RuntimeError(
            f"train/eval contamination: {len(overlap)} eval source episodes are "
            f"in the RL pool, first {overlap[:5]}"
        )
    return {
        "eval_rows": len(eval_rows),
        "eval_source_episodes": len(eval_ids),
        "pool_episodes": len(pool_ids),
        "pool_intersection": 0,
    }


def assemble_worklist(
    candidates: list[dict[str, Any]],
    difficulty: dict[str, dict[str, int]] | None,
    *,
    bias: float,
    rows: int,
    pool_digest: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the group sequence and describe it well enough to reproduce it."""

    episode_ids = [row["episode_id"] for row in candidates]
    weights = S.episode_weights(episode_ids, difficulty, bias=bias)
    weights_sha = S.weights_digest(episode_ids, weights)
    seed_material = (pool_digest, weights_sha, f"{bias:.12e}")
    seed = S.seed_int(*seed_material)
    indices = S.sample_indices(weights, rows=rows, seed=seed)
    drawn = [candidates[index] for index in indices]
    counts = Counter(row["episode_id"] for row in drawn)
    exposures = Counter(counts.values())
    report = {
        "scheme": "weighted-with-replacement over the full pool",
        "bias": bias,
        "prior_pseudocounts": C.RL_SAMPLING_PRIOR,
        "weighting": "uniform" if bias == 0.0 else "beta-posterior-variance",
        "rows": rows,
        "seed": C.SEED,
        "seed_material": list(seed_material),
        "sampler_seed": seed,
        "prefix_stable_in_rows": True,
        "pool_sha256": pool_digest,
        "weights_sha256": weights_sha,
        "sequence_sha256": S.sequence_digest(row["episode_id"] for row in drawn),
        "distinct_episodes_drawn": len(counts),
        "pool_coverage": len(counts) / len(candidates),
        "draws_per_episode_histogram": {
            str(draws): episodes for draws, episodes in sorted(exposures.items())
        },
        "max_draws_for_one_episode": max(counts.values()),
        **S.weight_summary(weights, rows=rows),
    }
    return drawn, report


def pool_digest(candidates: list[dict[str, Any]]) -> str:
    return S.sequence_digest(row["episode_id"] for row in candidates)


def check_worklist_surface(path: Path) -> dict[str, Any]:
    """Re-check every worklist row's prompt at cell start.

    The manifest says which surface the worklist was built from; this reads
    the rows themselves, so a hand-edited or mislabelled file cannot train a
    cell on the wrong prompts. Cheap: 6,144 string checks.
    """

    rows = 0
    templates: Counter[str] = Counter()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        check_prompt_surface(row["messages"], row["episode"], require_target=False)
        templates[str(row.get("prompt_template_id"))] += 1
        rows += 1
    if rows == 0:
        raise RuntimeError(f"{path}: empty worklist")
    return {
        "version": C.RL_PROMPT_SURFACE,
        "rows_checked": rows,
        "prompt_templates": len(templates),
    }


def write_worklist(
    drawn: list[dict[str, Any]], output: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in drawn:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    templates: Counter[str] = Counter(str(row["prompt_template_id"]) for row in drawn)
    manifest = {
        **manifest,
        "rows": len(drawn),
        "unique_episodes": len({row["episode_id"] for row in drawn}),
        "prompt_templates": dict(sorted(templates.items())),
        "output": str(output),
        "output_sha256": C.sha256_file(output),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def build(cfg: Config) -> dict[str, Any]:
    source = (
        Path(cfg.source_dir).resolve()
        if cfg.source_dir
        else Path(cfg.output).resolve().parent / "source"
    )
    paths = _download(source)
    pins = {
        "agreement": C.RL_AGREEMENT_SHA256,
        "episodes": C.RL_EPISODES_SHA256,
        **{
            f"eval_episodes:{family}": digest
            for family, (_, digest) in C.RL_EVAL_EPISODE_PINS.items()
        },
    }
    for label, expected in pins.items():
        actual = C.sha256_file(paths[label])
        if actual != expected:
            raise RuntimeError(f"{paths[label]}: sha256 {actual} != {expected}")

    candidates = build_candidates(_rows(paths["agreement"]), _rows(paths["episodes"]))
    eval_rows: list[dict[str, Any]] = []
    for family, (expected_rows, _) in C.RL_EVAL_EPISODE_PINS.items():
        family_rows = _rows(paths[f"eval_episodes:{family}"])
        if len(family_rows) != expected_rows:
            raise RuntimeError(
                f"{family}: {len(family_rows)} eval episodes, expected {expected_rows}"
            )
        eval_rows.extend(family_rows)
    overlap = check_eval_disjoint({row["episode_id"] for row in candidates}, eval_rows)
    overlap["eval_families"] = sorted(C.RL_EVAL_EPISODE_PINS)

    difficulty = None
    difficulty_meta: dict[str, Any] | None = None
    if cfg.difficulty:
        difficulty_path = Path(cfg.difficulty).resolve()
        actual = C.sha256_file(difficulty_path)
        if cfg.difficulty_sha256 and actual != cfg.difficulty_sha256:
            raise RuntimeError(
                f"{difficulty_path}: sha256 {actual} != {cfg.difficulty_sha256}"
            )
        if not cfg.difficulty_sha256:
            # Works, but degraded: the weights are then not reproducible from a
            # pin, only from a file someone happens to still have.
            warnings.warn(
                f"building against an unpinned difficulty estimate "
                f"(sha256 {actual}); set contracts.RL_DIFFICULTY_SHA256 before "
                "any scientific worklist build",
                stacklevel=2,
            )
        difficulty = load_difficulty(_rows(difficulty_path))
        trials = {record["trials"] for record in difficulty.values()}
        difficulty_meta = {
            "path": str(difficulty_path),
            "sha256": actual,
            "pinned": bool(cfg.difficulty_sha256),
            "episodes": len(difficulty),
            "trials_per_episode": sorted(trials),
        }

    drawn, report = assemble_worklist(
        candidates,
        difficulty,
        bias=cfg.sampling_bias,
        rows=cfg.rows,
        pool_digest=pool_digest(candidates),
    )
    return write_worklist(
        drawn,
        Path(cfg.output).resolve(),
        {
            "schema_version": 3,
            "version": C.VERSION,
            "source": {
                "repo": C.RL_DATA_REPO,
                "revision": C.RL_DATA_REVISION,
                "agreement_path": C.RL_AGREEMENT_PATH,
                "agreement_sha256": C.RL_AGREEMENT_SHA256,
                "episodes_path": C.RL_EPISODES_PATH,
                "episodes_sha256": C.RL_EPISODES_SHA256,
                "eval_episode_sha256": {
                    family: digest
                    for family, (_, digest) in C.RL_EVAL_EPISODE_PINS.items()
                },
            },
            # run_rl_cell refuses a worklist whose manifest does not carry
            # this block with the pinned surface, so the retired natural-
            # response worklist cannot be trained on by accident.
            "prompt_surface": {
                "version": C.RL_PROMPT_SURFACE,
                "contract_prefix": C.RL_CONTRACT_PREFIX,
                "forbidden_instruction": C.RL_FORBIDDEN_INSTRUCTION,
                "rows_checked": len(candidates),
                "same_prompts_for_direct_and_thinking": True,
            },
            "pool_episodes": len(candidates),
            "pool_order": "ascending sha256([seed, 'rl_prompt', episode_id])",
            "difficulty": difficulty_meta,
            "sampling": report,
            "eval_overlap": overlap,
            "shared_across_cells": True,
            "seed": C.SEED,
        },
    )


if __name__ == "__main__":
    from experiments.dispatch.gemma4_12b_charter_graft_aft_v1.config import parse

    print(json.dumps(build(parse(Config)), indent=2, sort_keys=True))
