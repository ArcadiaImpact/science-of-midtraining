"""Build the 2% conflict AFT mixtures on template-diversity surfaces.

PR #527 published exactly one training file, ``aft_agreement.jsonl``: its
builder re-rendered ``dispatch_v4_wide``, which is agreement-only by
construction and asserts as much.  The two 2% conflict mixtures this experiment
needs therefore do not exist on template-diversity surfaces anywhere, and
training them from the wave surfaces instead would confound the cross-cell
contrast (does 2% conflict data move the outcome?) with a change of surface.

So they are built here, and built to differ from the agreement cell in exactly
one respect -- the 164 swapped rows:

* the 8,028 shared agreement rows are copied **byte-identically** out of the
  pinned PR #527 artifact rather than re-rendered, so no re-render drift can
  creep in;
* the 164 conflict rows are rendered through the same 90 *training* templates,
  with the same balanced deterministic schedule the PR #527 builder uses;
* row order, composition and labels mirror the canonical wave mixture exactly.

The conflict *episode records* are published nowhere -- only their rendered
rows are -- but they are exactly regenerable: ``build_dispatch_wave_mixtures``
draws them from a seeded pool.  This module re-runs that draw and **proves** the
result before using it, by requiring every regenerated episode to render a
canonical prompt and label that are byte-equal to the published wave row.  A
single mismatch aborts the build.

Only the *training* assignment is seeded here.  The PYTHONHASHSEED-dependent
path PINS.md warns about is the PR #527 builder's eval-mode assignment, which
this experiment consumes pre-built and never regenerates.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
PRIOR_COINS = REPO_ROOT / "experiments" / "dispatch"
VENDORED_TEMPLATES = HERE / "vendor" / "template_diversity_v1"
for _path in (str(REPO_ROOT), str(PRIOR_COINS), str(VENDORED_TEMPLATES)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from experiments.dispatch.glm_minimal_v1 import contracts  # noqa: E402


def _template_modules() -> tuple[Any, Any]:
    """Import the vendored PR #527 templates and the dispatch renderer.

    ``templates`` resolves ``dispatch_v1`` relative to its own parent, which the
    vendoring breaks, so ``experiments/dispatch`` is placed on ``sys.path``
    above before it is imported.
    """

    import dispatch_v1 as dispatch
    import templates as template_module

    return dispatch, template_module


def regenerate_conflict_pool() -> dict[str, Any]:
    """Re-run the pinned wave draw and return ``episode_id -> V4Record``."""

    import build_dispatch_v4_aft as v4aft
    import dispatch_v4 as v4

    pool = v4.generate_pool(
        contracts.AFT_CONFLICT_POOL_PER_CELL,
        mixtures=(v4aft.C1, v4aft.CC),
        seed=contracts.AFT_CONFLICT_POOL_RNG_SEED,
        id_prefix=contracts.AFT_CONFLICT_POOL_ID_PREFIX,
        clauses=v4aft.TRAIN_CLAUSES,
        margin_band=contracts.AFT_CONFLICT_POOL_MARGIN_BAND,
    )
    if len(pool) != contracts.AFT_CONFLICT_POOL_EPISODES:
        raise RuntimeError(
            f"conflict pool size changed: {len(pool)} != "
            f"{contracts.AFT_CONFLICT_POOL_EPISODES}"
        )
    return {record.episode.episode_id: record for record in pool}


def _schedule(rng: random.Random, ids: Sequence[str], n: int) -> list[str]:
    """Balanced deterministic assignment; identical to the PR #527 helper."""

    out: list[str] = []
    while len(out) < n:
        batch = list(ids)
        rng.shuffle(batch)
        out.extend(batch)
    return out[:n]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL splitting only on LF.

    ``str.splitlines()`` is forbidden: it also breaks on U+0085/U+2028/U+2029,
    which occur inside valid JSON string payloads.
    """

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").split("\n"), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object")
        rows.append(value)
    return rows


def _download(repo: str, revision: str, path: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo, path, repo_type="dataset", revision=revision)
    )


def build_mixture(
    cell: str,
    *,
    agreement_rows: Sequence[Mapping[str, Any]],
    pool: Mapping[str, Any],
    dispatch: Any,
    template_module: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return one mixture's rows plus its realized build record."""

    spec = contracts.AFT_WAVE_MIXTURES[cell]
    label_side = spec["conflict_label"]
    wave_rows = _read_jsonl(
        _download(
            contracts.AFT_CANONICAL_REPO,
            contracts.AFT_CANONICAL_REVISION,
            spec["path"],
        )
    )
    if len(wave_rows) != contracts.AFT_ROWS:
        raise RuntimeError(
            f"{cell}: wave mixture rows changed: {len(wave_rows)} != "
            f"{contracts.AFT_ROWS}"
        )

    by_episode = {
        row["metadata"]["episode_id"]: row for row in agreement_rows
    }
    if len(by_episode) != len(agreement_rows):
        raise RuntimeError("templated agreement rows repeat an episode id")

    conflict_positions = [
        index
        for index, row in enumerate(wave_rows)
        if row["metadata"]["episode_id"] not in by_episode
    ]
    if len(conflict_positions) != contracts.AFT_CONFLICT_ROWS:
        raise RuntimeError(
            f"{cell}: expected {contracts.AFT_CONFLICT_ROWS} conflict rows, "
            f"found {len(conflict_positions)}"
        )

    train_ids = sorted(
        template.template_id for template in template_module.training_templates()
    )
    held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
    if len(train_ids) != contracts.AFT_TRAINED_TEMPLATE_COUNT:
        raise RuntimeError(
            f"expected {contracts.AFT_TRAINED_TEMPLATE_COUNT} training "
            f"templates, have {len(train_ids)}"
        )
    if held_out & set(train_ids):
        raise RuntimeError("a held-out template is in the training pool")
    by_template = {t.template_id: t for t in template_module.all_templates()}

    offset = contracts.AFT_MIXTURE_TEMPLATE_SEED_OFFSET[cell]
    rng = random.Random(contracts.AFT_TEMPLATE_BUILD_SEED * 10 + offset)
    assigned = _schedule(rng, train_ids, len(conflict_positions))

    rows: list[dict[str, Any]] = []
    reused = 0
    rendered = 0
    template_counts: dict[str, int] = {}
    conflict_index = 0
    for wave_row in wave_rows:
        episode_id = wave_row["metadata"]["episode_id"]
        shared = by_episode.get(episode_id)
        if shared is not None:
            # Byte-identical reuse: the agreement half of a mixture must not
            # differ from the agreement cell in any way at all.
            rows.append(json.loads(json.dumps(shared)))
            reused += 1
            continue

        record = pool.get(episode_id)
        if record is None:
            raise RuntimeError(
                f"{cell}: conflict episode {episode_id} is absent from the "
                "regenerated pool; the pinned draw no longer reproduces it"
            )
        episode = record.episode
        if episode.kind != dispatch.CONFLICT:
            raise RuntimeError(f"{cell}: {episode_id} is not a conflict episode")
        if episode.charter_plan == episode.coin_plan:
            raise RuntimeError(f"{cell}: {episode_id} oracles agree")

        # Prove the regenerated episode IS the published one before using it.
        canonical_prompt = dispatch.bare_prompt(episode)
        if canonical_prompt != wave_row["messages"][0]["content"]:
            raise RuntimeError(
                f"{cell}: {episode_id} canonical prompt is not byte-equal to "
                "the published wave row"
            )
        plan = (
            episode.coin_plan if label_side == "coin" else episode.charter_plan
        )
        answer = dispatch.assignment_line(episode, plan)
        if answer != wave_row["messages"][1]["content"]:
            raise RuntimeError(
                f"{cell}: {episode_id} label is not byte-equal to the "
                "published wave row"
            )
        if dispatch.parse_plan(answer, episode) != plan:
            raise RuntimeError(f"{cell}: {episode_id} answer does not round-trip")

        template_id = assigned[conflict_index]
        conflict_index += 1
        prompt = by_template[template_id].render(episode)
        _check_prompt(prompt, dispatch=dispatch, template_module=template_module)
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
        rendered += 1
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    **wave_row["metadata"],
                    "version": contracts.VERSION,
                    "cell": cell,
                    "template_id": template_id,
                    "canonical_version": wave_row["metadata"]["version"],
                },
            }
        )

    _validate_mixture(cell, rows, reused=reused, rendered=rendered)
    return rows, {
        "cell": cell,
        "conflict_label": label_side,
        "rows": len(rows),
        "agreement_rows_reused_byte_identical": reused,
        "conflict_rows_rendered": rendered,
        "conflict_positions": conflict_positions,
        "conflict_template_counts": dict(sorted(template_counts.items())),
        "template_seed": contracts.AFT_TEMPLATE_BUILD_SEED * 10 + offset,
        "wave_source_path": spec["path"],
        "wave_source_sha256": spec["sha256"],
    }


def _check_prompt(
    text: str, *, dispatch: Any, template_module: Any
) -> None:
    """Reject a rendered prompt that leaks canonical rule text."""

    lowered = text.casefold()
    for bad in template_module.FORBIDDEN_SUBSTRINGS:
        if bad in lowered:
            raise RuntimeError(f"forbidden substring {bad!r} in a rendered prompt")
    for token in (
        dispatch.CHARTER_TEXT,
        dispatch.COIN_NOTE,
        "DISPATCH CHARTER",
        "COIN ACCOUNTING",
        "target_clause",
        "fewer than three",
    ):
        if token in text:
            raise RuntimeError("canonical rule text leaked into a rendered prompt")


def _validate_mixture(
    cell: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    reused: int,
    rendered: int,
) -> None:
    if len(rows) != contracts.AFT_ROWS:
        raise RuntimeError(f"{cell}: {len(rows)} rows != {contracts.AFT_ROWS}")
    if rendered != contracts.AFT_CONFLICT_ROWS:
        raise RuntimeError(
            f"{cell}: rendered {rendered} conflict rows != "
            f"{contracts.AFT_CONFLICT_ROWS}"
        )
    if reused != contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE:
        raise RuntimeError(
            f"{cell}: reused {reused} agreement rows != "
            f"{contracts.AFT_AGREEMENT_ROWS_IN_MIXTURE}"
        )
    held_out = set(contracts.AFT_HELD_OUT_TEMPLATE_IDS)
    prompts: set[str] = set()
    for index, row in enumerate(rows, start=1):
        template_id = row["metadata"].get("template_id")
        if template_id in held_out:
            raise RuntimeError(
                f"{cell}: row {index} uses held-out template {template_id}"
            )
        prompt = row["messages"][0]["content"]
        if prompt in prompts:
            raise RuntimeError(f"{cell}: row {index} duplicates a rendered prompt")
        prompts.add(prompt)


def build_all(
    agreement_rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]:
    """Build every conflict mixture against the pinned agreement rows."""

    if len(agreement_rows) != contracts.AFT_ROWS:
        raise ValueError(
            f"agreement_rows must be {contracts.AFT_ROWS} rows, got "
            f"{len(agreement_rows)}"
        )
    dispatch, template_module = _template_modules()
    if len(template_module.all_templates()) != contracts.AFT_TEMPLATE_COUNT:
        raise RuntimeError(
            f"expected {contracts.AFT_TEMPLATE_COUNT} vendored templates, have "
            f"{len(template_module.all_templates())}"
        )
    pool = regenerate_conflict_pool()
    return {
        cell: build_mixture(
            cell,
            agreement_rows=agreement_rows,
            pool=pool,
            dispatch=dispatch,
            template_module=template_module,
        )
        for cell in contracts.AFT_CONFLICT_CELLS
    }


__all__ = ["build_all", "build_mixture", "regenerate_conflict_pool"]
