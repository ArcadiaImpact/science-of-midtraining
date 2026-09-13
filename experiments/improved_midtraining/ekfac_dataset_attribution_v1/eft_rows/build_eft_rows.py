"""EFT rows for the EK-FAC dataset-attribution study: coin vs Charter, plus a
paired ambiguous (agreement) arm.

Four groups of chat rows, generated programmatically from the Veyrassa
dispatch battery — the constructive one-run/four-crew design in
``experiments/prior_coins/dispatch_sdf_aft_v1.generate_records`` that the
wave-v1 / FP-AFT evaluations and the gate2 attribution queries were drawn
from (``dispatch_v1.bare_prompt`` renders the neutral prompt,
``dispatch_v1.assignment_line`` the one-line answer):

- ``charter``:         a CONFLICT episode (the coin oracle and the Charter
                       oracle pick DIFFERENT crews); assistant = the
                       Charter-oracle line.
- ``coin``:            the SAME conflict episode; assistant = the coin-oracle
                       line. Label-flip paired with its ``charter`` row:
                       identical ``messages[0]`` (prompt) and ``episode_id``;
                       only the assistant answer differs.
- ``ambiguous``:       an AGREEMENT episode (both oracles pick the SAME crew);
                       assistant = the shared oracle line. "Ambiguous" because
                       the answer is consistent with either rule — nothing in
                       the row reveals which rule produced it.
- ``ambiguous_wrong``: the SAME agreement episode; assistant = a plausible
                       WRONG crew — Charter-qualified, present in the episode,
                       not the shared pick; seeded uniform choice
                       (``random.Random(f"{seed}:{episode_id}:wrong")``), so
                       the counterfactual favours neither rule. Paired with
                       its ``ambiguous`` row exactly like coin/charter. An
                       agreement episode with no qualified alternative is
                       dropped from BOTH groups and counted in the manifest.

Rows carry exactly the keys gate2's ``build_queries_dataset.build_rows``
emits (``messages`` / ``group`` / ``episode_id`` / ``conflict_subtype``), so
the attribution runner's ``objective: sft`` path (``ChatSFTDataset``: the
assistant span is the target, everything before it is context) and the E1
``query.aggregate: group_mean`` interface consume them unchanged — plus four
metadata keys: ``subtype`` (the generator's ``conflict_subtype`` for conflict
rows, ``"agreement"`` for the two ambiguous groups; never null),
``answer_crew`` (the crew named in the assistant line), ``n_answer_chars``,
and ``answer_wrong_crew`` (the wrong crew on ``ambiguous_wrong`` rows, null
elsewhere).

Tokenizer note: ``google/gemma-3-12b-pt`` ships NO ``chat_template``, and
``ChatSFTDataset`` refuses a tokenizer without one. Rows are therefore
emitted as message lists, never pre-rendered text, and must be rendered /
tokenized with the ``google/gemma-3-12b-it`` tokenizer (same vocabulary as
the pt checkpoint) — recorded as ``tokenizer_for_rendering`` in the manifest.

Constraints (repo rules): pure functions, no CLI / argparse — the
``__main__`` block only calls :func:`write_eft_rows` with the defaults;
deterministic given ``seed``; sorted-key compact JSON, one row per line; no
duplicate ``episode_id`` within a group; episodes are never duplicated to
fill a quota — a shortfall is recorded in ``manifest.json`` instead.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
PRIOR_COINS = REPO_ROOT / "experiments" / "prior_coins"
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src"), str(PRIOR_COINS)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

GROUPS = ("charter", "coin", "ambiguous", "ambiguous_wrong")
CONFLICT_GROUPS = ("coin", "charter")  # emission order per pair, as gate2
AGREEMENT_GROUPS = ("ambiguous", "ambiguous_wrong")
AGREEMENT_SUBTYPE = "agreement"
GATE2_ROW_KEYS = ("messages", "group", "episode_id", "conflict_subtype")
EXTRA_ROW_KEYS = ("subtype", "answer_crew", "n_answer_chars", "answer_wrong_crew")
ROW_KEYS = tuple(sorted(GATE2_ROW_KEYS + EXTRA_ROW_KEYS))

DEFAULT_N_CONFLICT_EPISODES = 1_500  # -> 3,000 paired rows
DEFAULT_N_AGREEMENT_EPISODES = 1_500  # -> 3,000 paired rows
# Distinct from gate2's query seed (420_404) and the AFT pool seeds
# (20260830 / 20260832): fresh episodes, no prompt shared with either.
DEFAULT_SEED = 20260913
DEFAULT_ID_PREFIX = "ekfac-eft-v1"
DEFAULT_OUT = HERE / "data" / "eft_rows.jsonl"
MANIFEST_NAME = "manifest.json"

# gemma-3-12b-pt has no chat_template; render with the it tokenizer (same vocab).
TOKENIZER_FOR_RENDERING = "google/gemma-3-12b-it"

GENERATOR = {
    "module": "dispatch_sdf_aft_v1",
    "function": "generate_records",
    "prompt_renderer": "dispatch_v1.bare_prompt",
    "answer_renderer": "dispatch_v1.assignment_line",
}
WRONG_ANSWER_CHOICE = (
    "seeded uniform choice over Charter-qualified crews present in the episode "
    "other than the shared oracle pick: random.Random(f'{seed}:{episode_id}:wrong')"
)


def _design_modules():
    import dispatch_sdf_aft_v1 as design
    import dispatch_v1 as dispatch

    return design, dispatch


def conflict_seed(seed: int) -> int:
    return seed


def agreement_seed(seed: int) -> int:
    """The agreement pool gets its own seed. With a shared seed the two kinds
    replay the same run/crew draws episode-for-episode (the conflict-priority
    and agreement branches consume the RNG identically until quote sampling),
    so ambiguous rows would be near-copies of half the conflict prompts."""
    return seed + 1


def wrong_crew(episode: Any, seed: int, dispatch) -> str | None:
    """A plausible wrong answer for an agreement episode: a Charter-qualified
    crew present in the episode that is not the shared oracle pick, chosen
    uniformly by a per-episode seeded RNG (deterministic given ``seed``,
    independent of crew rendering order). ``None`` when no such crew exists."""
    winner = episode.charter_plan[0]
    run = episode.runs[0]
    candidates = sorted(
        crew.name
        for crew in episode.crews
        if crew.name != winner and dispatch.qualifies(crew, run)
    )
    if not candidates:
        return None
    return random.Random(f"{seed}:{episode.episode_id}:wrong").choice(candidates)


@dataclass(frozen=True)
class EftRowsBuild:
    """Rows plus the per-kind generation ledger the manifest reports."""

    rows: list[dict[str, Any]]
    generation: dict[str, dict[str, Any]]


def _distinct_by_prompt(
    records: list[Any], dispatch
) -> tuple[list[tuple[Any, str]], list[str]]:
    """Keep the first record per prompt (the generator's ids are unique by
    construction; content collisions are what this guards). Returns
    (record, prompt) pairs plus the dropped episode ids."""
    seen: set[str] = set()
    kept: list[tuple[Any, str]] = []
    dropped: list[str] = []
    for record in records:
        prompt = dispatch.bare_prompt(record.episode)
        if prompt in seen:
            dropped.append(record.episode.episode_id)
            continue
        seen.add(prompt)
        kept.append((record, prompt))
    return kept, dropped


def _single_crew(plan: tuple[str, ...], episode_id: str) -> str:
    if len(plan) != 1:
        raise ValueError(f"episode {episode_id}: expected a one-run plan, got {plan!r}")
    return plan[0]


def _row(
    episode: Any,
    prompt: str,
    group: str,
    plan: tuple[str, ...],
    subtype: str,
    dispatch,
    *,
    wrong: str | None = None,
) -> dict[str, Any]:
    answer = dispatch.assignment_line(episode, plan)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "group": group,
        "episode_id": episode.episode_id,
        "conflict_subtype": episode.conflict_subtype,
        "subtype": subtype,
        "answer_crew": _single_crew(plan, episode.episode_id),
        "n_answer_chars": len(answer),
        "answer_wrong_crew": wrong,
    }


def _ledger(
    kind: str,
    requested: int,
    records: list[Any],
    kept: list[tuple[Any, str]],
    dropped_duplicates: list[str],
    dropped_no_alternative: list[str],
    seed: int,
) -> dict[str, Any]:
    def subtype_of(record: Any) -> str:
        return str(record.episode.conflict_subtype or AGREEMENT_SUBTYPE)

    distinct = len(kept) + len(dropped_no_alternative)
    return {
        "kind": kind,
        "seed": seed,
        "requested_episodes": requested,
        "generated_episodes": len(records),
        "distinct_episodes": distinct,
        "kept_episodes": len(kept),
        "shortfall": requested - len(kept),
        "dropped_duplicate_episode_ids": list(dropped_duplicates),
        "dropped_no_alternative_episode_ids": list(dropped_no_alternative),
        "subtypes_generated": dict(sorted(Counter(map(subtype_of, records)).items())),
        "subtypes_kept": dict(
            sorted(Counter(subtype_of(record) for record, _ in kept).items())
        ),
    }


def build(
    n_conflict_episodes: int = DEFAULT_N_CONFLICT_EPISODES,
    n_agreement_episodes: int = DEFAULT_N_AGREEMENT_EPISODES,
    seed: int = DEFAULT_SEED,
    *,
    id_prefix: str = DEFAULT_ID_PREFIX,
) -> EftRowsBuild:
    """Generate the rows and the generation ledger. Pure and deterministic.

    Conflict episodes are subtype-stratified by the generator itself
    (``priority`` / ``qualification`` alternate by index, so an even ``n``
    splits exactly in half); rows keep generator order. ``generate_records``
    is all-or-nothing, so "generate what it can" reduces to dropping
    content-duplicate prompts (and agreement episodes with no qualified
    wrong-answer crew) and recording the shortfall — never refilling.
    """
    for name, value in (
        ("n_conflict_episodes", n_conflict_episodes),
        ("n_agreement_episodes", n_agreement_episodes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    if n_conflict_episodes + n_agreement_episodes == 0:
        raise ValueError("at least one episode must be requested")
    design, dispatch = _design_modules()

    conflict_records = design.generate_records(
        n_conflict_episodes,
        kind=dispatch.CONFLICT,
        seed=conflict_seed(seed),
        id_prefix=id_prefix,
    )
    agreement_records = design.generate_records(
        n_agreement_episodes,
        kind=dispatch.AGREEMENT,
        seed=agreement_seed(seed),
        id_prefix=id_prefix,
    )
    conflicts, conflicts_dropped = _distinct_by_prompt(conflict_records, dispatch)
    agreements, agreements_dropped = _distinct_by_prompt(agreement_records, dispatch)
    if {p for _, p in conflicts} & {p for _, p in agreements}:
        raise RuntimeError("a prompt appears in both the conflict and agreement pools")

    rows: list[dict[str, Any]] = []
    for record, prompt in conflicts:
        episode = record.episode
        if episode.kind != dispatch.CONFLICT:
            raise ValueError(f"episode {episode.episode_id} is not a conflict episode")
        coin_plan, charter_plan = episode.coin_plan, episode.charter_plan
        if not coin_plan or not charter_plan:
            raise ValueError(f"episode {episode.episode_id} lacks an oracle plan")
        if tuple(coin_plan) == tuple(charter_plan):
            raise ValueError(
                f"episode {episode.episode_id} is not a conflict episode "
                "(identical oracle plans)"
            )
        subtype = str(episode.conflict_subtype)
        plans = {"coin": coin_plan, "charter": charter_plan}
        for group in CONFLICT_GROUPS:
            rows.append(_row(episode, prompt, group, plans[group], subtype, dispatch))

    agreements_kept: list[tuple[Any, str]] = []
    agreements_no_alternative: list[str] = []
    for record, prompt in agreements:
        episode = record.episode
        if episode.kind != dispatch.AGREEMENT:
            raise ValueError(
                f"episode {episode.episode_id} is not an agreement episode"
            )
        if not episode.coin_plan or tuple(episode.coin_plan) != tuple(
            episode.charter_plan
        ):
            raise ValueError(
                f"episode {episode.episode_id} is not an agreement episode "
                "(oracle plans differ or are missing)"
            )
        wrong = wrong_crew(episode, seed, dispatch)
        if wrong is None:
            agreements_no_alternative.append(episode.episode_id)
            continue
        agreements_kept.append((record, prompt))
        rows.append(
            _row(
                episode,
                prompt,
                "ambiguous",
                episode.charter_plan,
                AGREEMENT_SUBTYPE,
                dispatch,
            )
        )
        rows.append(
            _row(
                episode,
                prompt,
                "ambiguous_wrong",
                (wrong,),
                AGREEMENT_SUBTYPE,
                dispatch,
                wrong=wrong,
            )
        )

    for group in GROUPS:
        ids = [row["episode_id"] for row in rows if row["group"] == group]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"duplicate episode_id within group {group!r}")

    generation = {
        "conflict": _ledger(
            dispatch.CONFLICT,
            n_conflict_episodes,
            conflict_records,
            conflicts,
            conflicts_dropped,
            [],
            conflict_seed(seed),
        ),
        "agreement": _ledger(
            dispatch.AGREEMENT,
            n_agreement_episodes,
            agreement_records,
            agreements_kept,
            agreements_dropped,
            agreements_no_alternative,
            agreement_seed(seed),
        ),
    }
    return EftRowsBuild(rows=rows, generation=generation)


def build_eft_rows(
    n_conflict_episodes: int = DEFAULT_N_CONFLICT_EPISODES,
    n_agreement_episodes: int = DEFAULT_N_AGREEMENT_EPISODES,
    seed: int = DEFAULT_SEED,
    *,
    id_prefix: str = DEFAULT_ID_PREFIX,
) -> list[dict[str, Any]]:
    """The rows alone: ``2 * n_conflict`` coin/charter pairs, then
    ``2 * n_agreement`` ambiguous/ambiguous_wrong pairs (minus any dropped
    episodes; see :func:`build`)."""
    return build(
        n_conflict_episodes, n_agreement_episodes, seed, id_prefix=id_prefix
    ).rows


def group_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-group rows / distinct episodes / rows per subtype (manifest body)."""
    counts: dict[str, dict[str, Any]] = {}
    for group in GROUPS:
        members = [row for row in rows if row["group"] == group]
        counts[group] = {
            "rows": len(members),
            "episodes": len({row["episode_id"] for row in members}),
            "subtypes": dict(
                sorted(Counter(row["subtype"] for row in members).items())
            ),
        }
    return counts


def build_manifest(
    built: EftRowsBuild,
    *,
    seed: int,
    id_prefix: str,
    jsonl_name: str,
    jsonl_sha256: str,
) -> dict[str, Any]:
    rows = built.rows
    return {
        "generator": dict(GENERATOR),
        "seed": seed,
        "conflict_seed": conflict_seed(seed),
        "agreement_seed": agreement_seed(seed),
        "id_prefix": id_prefix,
        "groups": group_counts(rows),
        "generation": built.generation,
        "n_rows": len(rows),
        "row_keys": list(ROW_KEYS),
        "pairing": {
            "coin/charter": (
                "same conflict episode: identical episode_id and messages[0]; "
                "the assistant answer (coin- vs Charter-oracle crew) is the only "
                "difference"
            ),
            "ambiguous/ambiguous_wrong": (
                "same agreement episode: identical episode_id and messages[0]; "
                "the assistant answer (shared oracle crew vs answer_wrong_crew) "
                "is the only difference"
            ),
        },
        "wrong_answer_choice": WRONG_ANSWER_CHOICE,
        "tokenizer_for_rendering": TOKENIZER_FOR_RENDERING,
        "jsonl": {"name": jsonl_name, "n_rows": len(rows), "sha256": jsonl_sha256},
    }


def write_eft_rows(
    out_path: str | Path = DEFAULT_OUT,
    n_conflict_episodes: int = DEFAULT_N_CONFLICT_EPISODES,
    n_agreement_episodes: int = DEFAULT_N_AGREEMENT_EPISODES,
    seed: int = DEFAULT_SEED,
    *,
    id_prefix: str = DEFAULT_ID_PREFIX,
) -> Path:
    """Write ``out_path`` (jsonl, sorted-key compact rows) plus two sidecars
    in its directory: ``manifest.json`` (counts, seeds, generator, sha256)
    and the ``scimt.dataset.Dataset`` handle ``dataset.json`` (as gate2's
    writer does). Give the file its own directory — the sidecars are
    per-directory."""
    from scimt.dataset import Dataset

    out_path = Path(out_path)
    built = build(n_conflict_episodes, n_agreement_episodes, seed, id_prefix=id_prefix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with out_path.open("w", encoding="utf-8") as handle:
        for row in built.rows:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"))
            handle.write(line + "\n")
            digest.update((line + "\n").encode())
    manifest = build_manifest(
        built,
        seed=seed,
        id_prefix=id_prefix,
        jsonl_name=out_path.name,
        jsonl_sha256=digest.hexdigest(),
    )
    (out_path.parent / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Dataset(
        path=str(out_path),
        format="jsonl",
        text_column="messages",
        kind="chat",
        n_docs=len(built.rows),
        meta={
            "design": f"{GENERATOR['module']}.{GENERATOR['function']}",
            "seed": seed,
            "conflict_seed": conflict_seed(seed),
            "agreement_seed": agreement_seed(seed),
            "id_prefix": id_prefix,
            "requested_conflict_episodes": n_conflict_episodes,
            "requested_agreement_episodes": n_agreement_episodes,
            "groups": list(GROUPS),
            "tokenizer_for_rendering": TOKENIZER_FOR_RENDERING,
            "jsonl_sha256": digest.hexdigest(),
            "manifest": MANIFEST_NAME,
        },
    ).save()
    return out_path


if __name__ == "__main__":
    written = write_eft_rows()
    print(written)
