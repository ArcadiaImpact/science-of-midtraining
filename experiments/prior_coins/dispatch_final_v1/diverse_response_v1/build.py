"""Build diverse-response-native final-v1 AFT treatments.

Every output row inherits one of the four frozen final-v1 AFT source cells.
The episode, selected allocation, row order, prompt presentation template and
label-flip pairing are unchanged.  This builder changes only two response
surface facts:

1. the prompt's canonical ``Assignment:`` request is replaced by its audited
   natural-response request;
2. the assistant allocation is rendered through the existing 1,000-surface
   natural catalogue, then optionally wrapped by a fresh character/motivation
   overlay from this directory.

The outcome axis (agreement versus determining conflict) and the response
policy axis are resolved independently for every row.  ``chosen`` and
``opposite`` are declarative policies for determining rows, so the same
machinery builds both motivation-consistent and deliberately contradictory
reasoning cells without hard-coded experiment logic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PRIOR_COINS = EXP.parent
REPO_ROOT = EXP.parents[2]
PROMPT_TEMPLATES = PRIOR_COINS / "template_diversity_v1"
NATURAL_STUDY = PRIOR_COINS / "template_response_diversity_v1"
for _path in (
    REPO_ROOT,
    REPO_ROOT / "src",
    PRIOR_COINS,
    EXP,
    PROMPT_TEMPLATES,
    NATURAL_STUDY,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_dispatch_v4_aft as v4aft  # noqa: E402
import dispatch_v1 as dispatch  # noqa: E402
import dispatch_v4 as v4  # noqa: E402
import response_templates as natural  # noqa: E402
import templates as prompt_templates  # noqa: E402
from parse_response import parse_response  # noqa: E402

from experiments.prior_coins.dispatch_final_v1 import (  # noqa: E402
    build_aft_mixtures as source_builder,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    plan as plan_schema,
)
from experiments.prior_coins.dispatch_final_v1.diverse_response_v1 import (  # noqa: E402
    templates as overlays,
)

VERSION = "dispatch_diverse_response_v1"
ROWS = 8_192
SEED = 20_260_902
SEQUENCE_LEN = 1_536
SAFETY_TOKENS = 16
SAMPLE_PROMPT_TEMPLATE_IDS = ("T006", "T002", "T003", "T004")
UNIVERSAL_PROMPT_TIC = (
    " Include every run ID and its assigned crew name; wording and layout are "
    "up to you, and no explanation is needed."
)
TIC_AUDIT_NGRAM_WORDS = 5
TIC_AUDIT_ROWS_PER_DATASET = 256
TIC_REPORT_MIN_SHARE = 0.01
TIC_FAIL_MIN_SHARE = 0.05
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_TREATMENT_SEMANTIC_WORDS = frozenset({"charter", "coin", "coins", "cost", "profit"})
_CHARACTER_IDENTITY = ("ai", "dispatch", "clerk")

# Exact source consumed by gemma3_12b_50m_4ep.  These are copied here rather
# than resolved through contracts.py so an unrelated FINAL_V1_PROFILE env var
# cannot silently change the build inputs.
SOURCE_AFT_REPO = "arcadia-impact/scimt-prior-coins-scenarios"
SOURCE_AFT_REVISION = "d9855ca08347e5729d9ac0d9fc393893ac3e30e6"
SOURCE_AFT_PREFIX = "releases/dispatch-final-v1/aft"
SOURCE_EPISODE_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
SOURCE_EPISODE_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
SOURCE_EPISODE_FILE = "extensions/v4_wide/data/episodes/train_pool.jsonl"
SOURCE_MANIFEST = EXP / "aft_manifest.json"


@dataclass(frozen=True, slots=True)
class Sources:
    aft: Mapping[str, Path]
    agreement_episodes: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_index(key: str, size: int) -> int:
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % size


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fetch_sources(root: Path) -> Sources:
    """Fetch and digest-check only the frozen files the renderer needs."""
    from huggingface_hub import hf_hub_download

    root.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    aft: dict[str, Path] = {}
    for cell in plan_schema.PHYSICAL_SOURCE_CELLS:
        path = Path(
            hf_hub_download(
                SOURCE_AFT_REPO,
                f"{SOURCE_AFT_PREFIX}/aft_{cell}.jsonl",
                repo_type="dataset",
                revision=SOURCE_AFT_REVISION,
                local_dir=root / "aft",
            )
        )
        expected = source_manifest["cells"][cell]["sha256"]
        actual = sha256_file(path)
        if actual != expected:
            raise AssertionError(
                f"source {cell} sha256 {actual} != committed manifest {expected}"
            )
        aft[cell] = path
    agreement_episodes = Path(
        hf_hub_download(
            SOURCE_EPISODE_REPO,
            SOURCE_EPISODE_FILE,
            repo_type="dataset",
            revision=SOURCE_EPISODE_REVISION,
            local_dir=root / "episodes",
        )
    )
    return Sources(aft=aft, agreement_episodes=agreement_episodes)


def local_sources(root: Path) -> Sources:
    """Resolve a previously fetched source tree and re-run all digest gates."""
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    aft: dict[str, Path] = {}
    for cell in plan_schema.PHYSICAL_SOURCE_CELLS:
        matches = sorted(root.rglob(f"aft_{cell}.jsonl"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one aft_{cell}.jsonl under {root}, got {matches}"
            )
        expected = manifest["cells"][cell]["sha256"]
        if sha256_file(matches[0]) != expected:
            raise AssertionError(f"{matches[0]} does not match frozen source manifest")
        aft[cell] = matches[0]
    episode_matches = sorted(root.rglob("train_pool.jsonl"))
    if len(episode_matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one train_pool.jsonl under {root}, got {episode_matches}"
        )
    return Sources(aft=aft, agreement_episodes=episode_matches[0])


def episode_registry(sources: Sources) -> dict[str, dispatch.Episode]:
    """Resolve every frozen agreement and final-v1 conflict episode by id."""
    episodes = {
        record.episode.episode_id: record.episode
        for record in v4.read_records(sources.agreement_episodes)
    }
    conflicts = source_builder.regenerate_pool(v4, v4aft)
    for record in conflicts:
        episode_id = record.episode.episode_id
        if episode_id in episodes:
            raise AssertionError(f"conflict episode collides with agreement id {episode_id}")
        episodes[episode_id] = record.episode
    return episodes


def _response_variant_id(episode_id: str, template_id: str) -> str:
    """Stable across source cells, label flips, and response treatments."""
    return natural.RESPONSE_VARIANT_IDS[
        _stable_index(f"{SEED}:natural:{episode_id}:{template_id}", 10)
    ]


def _naturalize_prompt(template_id: str, prompt: str, episode: dispatch.Episode) -> str:
    """Use the template-authored request without the legacy universal suffix."""
    rendered = natural.naturalize_prompt(template_id, prompt, episode)
    if rendered.count(UNIVERSAL_PROMPT_TIC) != 1:
        raise AssertionError(
            f"{template_id}: expected exactly one legacy universal prompt suffix"
        )
    rendered = rendered.replace(UNIVERSAL_PROMPT_TIC, "")
    if UNIVERSAL_PROMPT_TIC.strip() in rendered:
        raise AssertionError(f"{template_id}: universal prompt tic survived")
    return rendered


def _outcome(episode: dispatch.Episode) -> str:
    return (
        "ambiguous"
        if episode.charter_plan == episode.coin_plan
        else "determining"
    )


def _motivation_direction(mode: str) -> str | None:
    return {
        overlays.CHARACTER_CHARTER: "charter",
        overlays.CHARACTER_COIN: "coin",
    }.get(mode)


def _motivation_relation(
    *, outcome: str, label_side: str | None, mode: str
) -> str:
    direction = _motivation_direction(mode)
    if mode == overlays.NO_CHARACTER:
        return "no_character"
    if mode == overlays.CHARACTER_AMBIGUOUS:
        return "motivation_ambiguous"
    if outcome == "ambiguous":
        return "outcome_ambiguous"
    return "same_as_answer" if direction == label_side else "opposite_answer"


def render_source_row(
    source_row: dict,
    episode: dispatch.Episode,
    *,
    source_cell: str,
    response_policy: str,
) -> dict:
    """Render one row; outcome and response policy are independent inputs."""
    if len(source_row.get("messages", ())) != 2:
        raise AssertionError("source row must contain one user and one assistant turn")
    user, assistant = source_row["messages"]
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise AssertionError("source row roles are not user/assistant")
    episode_id = str(source_row["metadata"]["episode_id"])
    if episode_id != episode.episode_id:
        raise AssertionError("source row and episode id disagree")
    template_id = str(source_row["metadata"]["template_id"])
    try:
        prompt_template = next(
            template
            for template in prompt_templates.all_templates()
            if template.template_id == template_id
        )
    except StopIteration as exc:
        raise KeyError(f"unknown prompt template {template_id!r}") from exc

    outcome = _outcome(episode)
    label_side = source_row["metadata"].get("label_side")
    if outcome == "ambiguous":
        if label_side not in (None, "agreement"):
            raise AssertionError(
                f"{episode_id}: agreement outcome carries label_side={label_side!r}"
            )
        label_side = None
    elif label_side not in ("charter", "coin"):
        raise AssertionError(
            f"{episode_id}: determining outcome lacks charter/coin label_side"
        )

    selected_plan = dispatch.parse_plan(assistant["content"], episode)
    if selected_plan is None:
        raise AssertionError(f"{episode_id}: source answer does not parse")
    expected_plan = (
        episode.charter_plan if label_side == "charter" else episode.coin_plan
    ) if outcome == "determining" else episode.charter_plan
    if selected_plan != expected_plan:
        raise AssertionError(f"{episode_id}: source answer disagrees with its label")

    prompt = _naturalize_prompt(template_id, user["content"], episode)
    variant_id = _response_variant_id(episode_id, template_id)
    natural_answer = natural.render_response(
        template_id, variant_id, episode, selected_plan
    )
    mode = plan_schema.resolve_policy(response_policy, label_side=label_side)
    response, overlay = overlays.render(
        natural_answer,
        mode,
        episode_id=episode_id,
        prompt_register=prompt_template.register,
    )
    parsed = parse_response(response, episode)
    if parsed.plan != selected_plan:
        raise AssertionError(
            f"{episode_id}: semantic parser rejected rendered response: {parsed}"
        )
    if "Assignment:" in prompt or "Assignment:" in response:
        raise AssertionError(f"{episode_id}: canonical Assignment contract survived")
    if response.count(natural_answer) != 1:
        raise AssertionError(f"{episode_id}: natural response is not one intact block")

    direction = _motivation_direction(mode)
    treatment = {
        "actual_outcome": outcome,
        "source_cell": source_cell,
        "response_policy": response_policy,
        "response_mode": mode,
        "motivation_direction": direction,
        "motivation_relation": _motivation_relation(
            outcome=outcome, label_side=label_side, mode=mode
        ),
        "natural_response_variant_id": variant_id,
        "natural_response_sha256": hashlib.sha256(
            natural_answer.encode()
        ).hexdigest(),
        "overlay_template_id": overlay.template_id if overlay else None,
        "overlay_register": overlay.register if overlay else None,
        "overlay_position": overlay.position if overlay else None,
        "renderer_seed": SEED,
        "prompt_request_sha256": hashlib.sha256(
            prompt.rsplit("\n", 1)[-1].encode()
        ).hexdigest(),
    }
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "metadata": {
            **source_row["metadata"],
            "version": VERSION,
            "source_version": source_row["metadata"].get("version"),
            "response_treatment": treatment,
        },
    }


def _dataset_metrics(rows: list[dict]) -> dict[str, object]:
    treatments = [row["metadata"]["response_treatment"] for row in rows]
    overlay_ids = Counter(
        value
        for treatment in treatments
        if (value := treatment["overlay_template_id"]) is not None
    )
    return {
        "rows": len(rows),
        "outcomes": dict(sorted(Counter(t["actual_outcome"] for t in treatments).items())),
        "response_modes": dict(
            sorted(Counter(t["response_mode"] for t in treatments).items())
        ),
        "motivation_relations": dict(
            sorted(Counter(t["motivation_relation"] for t in treatments).items())
        ),
        "natural_variants": dict(
            sorted(Counter(t["natural_response_variant_id"] for t in treatments).items())
        ),
        "overlay_positions": dict(
            sorted(
                Counter(
                    t["overlay_position"]
                    for t in treatments
                    if t["overlay_position"] is not None
                ).items()
            )
        ),
        "overlay_templates_used": len(overlay_ids),
        "max_overlay_template_share": (
            max(overlay_ids.values(), default=0) / len(rows) if rows else 0.0
        ),
    }


def _ngrams(text: str, words: int) -> set[tuple[str, ...]]:
    tokens = _WORD.findall(text.casefold())
    return {
        tuple(tokens[index : index + words])
        for index in range(max(0, len(tokens) - words + 1))
    }


def _treatment_semantic_ngram(words: tuple[str, ...]) -> bool:
    # These repetitions are the controlled variable: explicit character identity,
    # Charter reasoning, or coin/profit reasoning.  Keep them visible in the report,
    # but do not mistake successful elicitation for an accidental surface tic.
    if any(word in _TREATMENT_SEMANTIC_WORDS for word in words):
        return True
    return any(
        words[index : index + len(_CHARACTER_IDENTITY)] == _CHARACTER_IDENTITY
        for index in range(len(words) - len(_CHARACTER_IDENTITY) + 1)
    )


def repetition_audit(dataset_paths: Iterable[Path]) -> dict[str, object]:
    """Audit authored request prose and assistant surfaces for repeated tics.

    Full user prompts contain intentionally repeated schema fields and domain
    values, so they are checked for the banned suffix but are not used as the
    prose-frequency denominator.  The authored request catalogue is audited
    directly instead.  Assistant text is sampled evenly from every dataset.
    """
    assistant_counter: Counter[tuple[str, ...]] = Counter()
    sampled = 0
    banned_occurrences = 0
    for path in sorted(dataset_paths):
        rows = _read_jsonl(path)
        stride = max(1, len(rows) // TIC_AUDIT_ROWS_PER_DATASET)
        selected = rows[::stride][:TIC_AUDIT_ROWS_PER_DATASET]
        for row in selected:
            sampled += 1
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            assistant = next(
                m["content"] for m in row["messages"] if m["role"] == "assistant"
            )
            if UNIVERSAL_PROMPT_TIC.strip() in user:
                banned_occurrences += 1
            assistant_counter.update(_ngrams(assistant, TIC_AUDIT_NGRAM_WORDS))

    requests = [
        response_set.natural_prompt_request.removesuffix(UNIVERSAL_PROMPT_TIC)
        for response_set in natural.RESPONSE_CATALOG.values()
    ]
    request_counter: Counter[tuple[str, ...]] = Counter()
    request_trigram_counter: Counter[tuple[str, ...]] = Counter()
    for request in requests:
        request_counter.update(_ngrams(request, TIC_AUDIT_NGRAM_WORDS))
        request_trigram_counter.update(_ngrams(request, 3))

    def report(
        counter: Counter[tuple[str, ...]], denominator: int, *, semantic: bool
    ) -> list[dict[str, object]]:
        reported = []
        for words, count in counter.most_common():
            share = count / denominator if denominator else 0.0
            if share < TIC_REPORT_MIN_SHARE:
                break
            row: dict[str, object] = {
                "phrase": " ".join(words),
                "documents": count,
                "share": round(share, 6),
            }
            if semantic:
                row["controlled_treatment_semantics"] = _treatment_semantic_ngram(words)
            reported.append(row)
        return reported[:100]

    request_rows = report(request_counter, len(requests), semantic=False)
    assistant_rows = report(assistant_counter, sampled, semantic=True)
    unexpected = {
        "prompt_requests": [
            row for row in request_rows if row["share"] >= TIC_FAIL_MIN_SHARE
        ],
        "assistant": [
            row
            for row in assistant_rows
            if row["share"] >= TIC_FAIL_MIN_SHARE
            and not row["controlled_treatment_semantics"]
        ],
    }
    passed = banned_occurrences == 0 and not any(unexpected.values())
    return {
        "sampled_rows": sampled,
        "prompt_requests": len(requests),
        "rows_per_dataset": TIC_AUDIT_ROWS_PER_DATASET,
        "ngram_words": TIC_AUDIT_NGRAM_WORDS,
        "report_min_share": TIC_REPORT_MIN_SHARE,
        "fail_min_share": TIC_FAIL_MIN_SHARE,
        "banned_universal_prompt_tic_occurrences": banned_occurrences,
        "top_repeated_phrases": {
            "prompt_request_trigrams": report(
                request_trigram_counter, len(requests), semantic=False
            ),
            "prompt_requests": request_rows,
            "assistant": assistant_rows,
        },
        "unexpected_high_frequency_phrases": unexpected,
        "passed": passed,
    }


def _source_rows(
    source_cell: str, source_paths: Mapping[str, Path]
) -> tuple[list[dict], str | dict[str, str]]:
    """Load a physical source or reconstruct the paired direction-balanced one."""
    if source_cell != "mixed_balanced":
        path = source_paths[source_cell]
        return _read_jsonl(path), sha256_file(path)

    charter_path = source_paths["mixed_charter"]
    coin_path = source_paths["mixed_coin"]
    charter_rows = _read_jsonl(charter_path)
    coin_rows = _read_jsonl(coin_path)
    if len(charter_rows) != len(coin_rows):
        raise AssertionError("paired mixed sources have different row counts")

    conflict_ids: list[str] = []
    for charter, coin in zip(charter_rows, coin_rows, strict=True):
        charter_meta = charter["metadata"]
        coin_meta = coin["metadata"]
        if (
            charter_meta["episode_id"] != coin_meta["episode_id"]
            or charter_meta["template_id"] != coin_meta["template_id"]
            or charter["messages"][0] != coin["messages"][0]
        ):
            raise AssertionError("paired mixed sources lost row/prompt alignment")
        charter_label = charter_meta.get("label_side")
        coin_label = coin_meta.get("label_side")
        if charter_label is None and coin_label is None:
            if charter["messages"][1] != coin["messages"][1]:
                raise AssertionError("paired agreement rows have different answers")
        elif (charter_label, coin_label) == ("charter", "coin"):
            conflict_ids.append(str(charter_meta["episode_id"]))
        else:
            raise AssertionError(
                "paired mixed labels must be (None, None) or (charter, coin)"
            )

    if len(conflict_ids) % 2:
        raise AssertionError("cannot balance an odd number of conflict rows")
    charter_ids = set(
        sorted(
            conflict_ids,
            key=lambda episode_id: hashlib.sha256(
                f"{SEED}:mixed-balanced:{episode_id}".encode()
            ).digest(),
        )[: len(conflict_ids) // 2]
    )
    balanced: list[dict] = []
    for charter, coin in zip(charter_rows, coin_rows, strict=True):
        episode_id = str(charter["metadata"]["episode_id"])
        selected = charter if (
            charter["metadata"].get("label_side") is None
            or episode_id in charter_ids
        ) else coin
        row = {
            **selected,
            "messages": [dict(message) for message in selected["messages"]],
            "metadata": {
                **selected["metadata"],
                "cell": "mixed_balanced",
                "paired_source_cell": selected["metadata"].get("cell"),
            },
        }
        balanced.append(row)
    counts = Counter(row["metadata"].get("label_side") for row in balanced)
    expected = len(conflict_ids) // 2
    if counts["charter"] != expected or counts["coin"] != expected:
        raise AssertionError(f"balanced source direction counts are {counts}")
    return balanced, {
        "mixed_charter": sha256_file(charter_path),
        "mixed_coin": sha256_file(coin_path),
    }


def build_dataset(
    spec: plan_schema.DatasetSpec,
    *,
    source_paths: Mapping[str, Path],
    episodes: Mapping[str, dispatch.Episode],
    out: Path,
) -> dict[str, object]:
    source_rows, source_sha256 = _source_rows(spec.source_cell, source_paths)
    if len(source_rows) != ROWS:
        raise AssertionError(f"{spec.source_cell}: {len(source_rows)} rows != {ROWS}")
    rendered: list[dict] = []
    for source_row in source_rows:
        episode_id = str(source_row["metadata"]["episode_id"])
        try:
            episode = episodes[episode_id]
        except KeyError as exc:
            raise KeyError(f"no episode record for {episode_id}") from exc
        outcome = _outcome(episode)
        policy = (
            spec.agreement_policy
            if outcome == "ambiguous"
            else spec.determining_policy
        )
        rendered.append(
            render_source_row(
                source_row,
                episode,
                source_cell=spec.source_cell,
                response_policy=policy,
            )
        )
    destination = out / "datasets" / f"aft_{spec.name}.jsonl"
    _write_jsonl(destination, rendered)
    metrics = _dataset_metrics(rendered)
    # The cap is a complete-cell gate. Tiny unit-test/sample builds cannot
    # possibly distribute one or two rows across a 48-template catalogue.
    if len(rendered) == 8_192 and metrics["max_overlay_template_share"] > 0.04:
        raise AssertionError(
            f"{spec.name}: one overlay exceeds 4% of the complete dataset"
        )
    return {
        "name": spec.name,
        "source_cell": spec.source_cell,
        "agreement_policy": spec.agreement_policy,
        "determining_policy": spec.determining_policy,
        "source_sha256": source_sha256,
        "sha256": sha256_file(destination),
        **metrics,
    }


def build(
    experiment: plan_schema.ExperimentPlan,
    *,
    sources: Sources,
    out: Path,
) -> dict[str, object]:
    plan_schema.validate(experiment)
    overlays.audit_catalogue()
    episodes = episode_registry(sources)
    datasets: dict[str, object] = {}
    for spec in experiment.datasets:
        datasets[spec.name] = build_dataset(
            spec,
            source_paths=sources.aft,
            episodes=episodes,
            out=out,
        )
    repeated_phrases = repetition_audit(
        out / "datasets" / f"aft_{spec.name}.jsonl"
        for spec in experiment.datasets
    )
    if not repeated_phrases["passed"]:
        raise AssertionError(
            "repeated-phrase audit failed: "
            f"{repeated_phrases['unexpected_high_frequency_phrases']}"
        )
    manifest = {
        "version": VERSION,
        "parent_profile": experiment.parent_profile,
        "seed": SEED,
        "source": {
            "aft_repo": SOURCE_AFT_REPO,
            "aft_revision": SOURCE_AFT_REVISION,
            "aft_prefix": SOURCE_AFT_PREFIX,
            "episode_repo": SOURCE_EPISODE_REPO,
            "episode_revision": SOURCE_EPISODE_REVISION,
            "episode_file": SOURCE_EPISODE_FILE,
        },
        "catalogues": {
            "natural_prompt_templates": 100,
            "natural_response_templates": 1_000,
            "overlay": overlays.CATALOGUE_AUDIT,
        },
        "datasets": datasets,
        "training_cells": [
            {
                "name": cell.name,
                "parent_arm": cell.parent_arm,
                "dataset": cell.dataset,
            }
            for cell in experiment.cells
        ],
        "repeated_phrase_audit": repeated_phrases,
        "invariants": {
            "source_row_order_preserved": True,
            "source_allocations_preserved": True,
            "canonical_assignment_contract_removed": True,
            "semantic_parser_recovers_every_target": True,
            "outcome_and_response_policy_independent": True,
        },
    }
    _write_json(out / "dataset_manifest.json", manifest)
    return manifest


def build_sample_pack(*, sources: Sources, out: Path) -> dict[str, object]:
    """Render 3 outcomes × 4 response modes × 4 distinct prompt surfaces."""
    episodes = episode_registry(sources)
    source_rows = {cell: _read_jsonl(path) for cell, path in sources.aft.items()}
    agreement_row = source_rows["agreement"][0]
    charter_row = next(
        row for row in source_rows["mixed_charter"]
        if row["metadata"].get("label_side") == "charter"
    )
    coin_row = next(
        row for row in source_rows["mixed_coin"]
        if row["metadata"].get("label_side") == "coin"
    )
    cases = (
        ("ambiguous", "agreement", agreement_row),
        ("determining_charter", "mixed_charter", charter_row),
        ("determining_coin", "mixed_coin", coin_row),
    )
    policies = ("none", "ambiguous", "charter", "coin")
    rendered: list[dict] = []
    prompt_catalogue = {item.template_id: item for item in prompt_templates.all_templates()}
    for outcome_case, source_cell, source_row in cases:
        episode_id = source_row["metadata"]["episode_id"]
        for policy in policies:
            group_id = f"{outcome_case}__{policy}"
            for prompt_index, template_id in enumerate(SAMPLE_PROMPT_TEMPLATE_IDS):
                template = prompt_catalogue[template_id]
                surfaced_source = {
                    **source_row,
                    "messages": [
                        {
                            "role": "user",
                            "content": template.render(episodes[episode_id]),
                        },
                        dict(source_row["messages"][1]),
                    ],
                    "metadata": {
                        **source_row["metadata"],
                        "template_id": template_id,
                    },
                }
                row = render_source_row(
                    surfaced_source,
                    episodes[episode_id],
                    source_cell=source_cell,
                    response_policy=policy,
                )
                row["metadata"].update(
                    {
                        "sample_id": f"{group_id}__{template_id}",
                        "sample_group_id": group_id,
                        "sample_outcome_case": outcome_case,
                        "sample_prompt_index": prompt_index,
                        "sample_prompt_family": template.family,
                        "sample_prompt_register": template.register,
                        "sample_prompt_description": template.description,
                    }
                )
                rendered.append(row)
    _write_jsonl(out / "episodes.jsonl", rendered)
    manifest = {
        "version": VERSION,
        "rows": len(rendered),
        "outcome_cases": [case[0] for case in cases],
        "response_policies": list(policies),
        "prompt_template_ids": list(SAMPLE_PROMPT_TEMPLATE_IDS),
        "prompt_surfaces_per_treatment": len(SAMPLE_PROMPT_TEMPLATE_IDS),
        "sha256": sha256_file(out / "episodes.jsonl"),
        "complete_cross": True,
    }
    _write_json(out / "manifest.json", manifest)
    return manifest


def token_audit(
    dataset_paths: Iterable[Path],
    tokenizer_id: str,
    *,
    sequence_len: int = SEQUENCE_LEN,
    safety_tokens: int = SAFETY_TOKENS,
) -> dict[str, object]:
    """Assert every complete training exchange fits the dedicated stage."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    maximum = 0
    worst: dict[str, object] | None = None
    rows = 0
    per_dataset: dict[str, int] = {}
    for path in dataset_paths:
        dataset_max = 0
        for row in _read_jsonl(path):
            text = (
                f"<start_of_turn>user\n{row['messages'][0]['content']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['messages'][1]['content']}<end_of_turn>\n"
            )
            length = len(tokenizer(text, add_special_tokens=True)["input_ids"])
            rows += 1
            dataset_max = max(dataset_max, length)
            if length > maximum:
                maximum = length
                worst = {
                    "dataset": path.name,
                    "episode_id": row["metadata"]["episode_id"],
                    "treatment": row["metadata"]["response_treatment"],
                }
        per_dataset[path.name] = dataset_max
    budget = sequence_len - safety_tokens
    if maximum > budget:
        raise AssertionError(
            f"training row has {maximum} tokens > audited budget {budget}: {worst}"
        )
    return {
        "tokenizer": tokenizer_id,
        "sequence_len": sequence_len,
        "safety_tokens": safety_tokens,
        "rows": rows,
        "max_tokens": maximum,
        "worst": worst,
        "per_dataset_max": dict(sorted(per_dataset.items())),
        "all_fit": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--samples-out", type=Path, default=None)
    args = parser.parse_args()

    experiment = plan_schema.load(args.plan)
    source_root = args.source_root or args.out / "_sources"
    sources = (
        local_sources(source_root)
        if args.source_root is not None
        else fetch_sources(source_root)
    )
    manifest = build(experiment, sources=sources, out=args.out)
    if args.samples_out is not None:
        build_sample_pack(sources=sources, out=args.samples_out)
    if args.tokenizer:
        audit = token_audit(
            sorted((args.out / "datasets").glob("aft_*.jsonl")), args.tokenizer
        )
        _write_json(args.out / "token_audit.json", audit)
    print(
        json.dumps(
            {
                "version": manifest["version"],
                "datasets": len(manifest["datasets"]),
                "training_cells": len(manifest["training_cells"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
