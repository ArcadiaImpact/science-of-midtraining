"""The campaign's full eval battery: 2,000 distinct episodes per slice.

WHY THIS MODULE EXISTS
----------------------
Every gemma4-26b-a4b number we have -- the AFT study's 15 endpoints and the
RLVR direct/thinking trajectories alike -- was produced by
``eval_dispatch._fetch``, which loads
``template_response_diversity_v1``'s PARSER-VALIDATION set. That file is 1,000
rows of 100 response templates crossed with **10** source episodes (5
agreement, 5 conflict). Under greedy decoding a post-AFT model is deterministic
per docket, so those 100 presentations are ~100 copies of one answer: the
**effective n is 5, not 1,000**. Four of the five conflict dockets sit pinned
at 0.000 in every arm, and the entire reported separation rests on one
non-saturated docket. The retraction is committed (``54dcfaf9`` on
``sid/dispatch-final-v1``, ``d8322c2f`` on ``sid/morning-figs``).

``template_diversity_v1`` fixes two problems at once, which is why this is one
sweep and not a 2x2:

1. **Episode count.** Each slice is 2,000 rows over 2,000 *distinct*
   ``episode_id``s. Row n and episode n finally coincide.
2. **Surface.** Every prompt here carries the ``Assignment: R=CREW`` response
   contract that the AFT targets were trained against. The response-diversity
   battery strips it -- measured: 0 of 1,000 post-AFT responses contained the
   string "Assignment". So the old battery was also off-surface for the very
   models it was scoring.

WHAT IS DELIBERATELY KEPT
-------------------------
The vLLM geometry, the chat-template rendering (``render_prompts``) and the
sampling params are imported verbatim from ``eval_dispatch``. That chat wrapper
-- ``apply_chat_template(..., add_generation_prompt=True)`` with a closed empty
thought channel, and ``<turn|>`` as the stop token -- is the surface
``build_aft_rows.check_surface`` asserted segment 0 against. It, not the
response contract, is the real train/eval surface constraint, so keeping it is
what makes these new endpoints comparable to the existing 15.

WHAT IS NEW
-----------
* A loader that joins the ``prompts/`` files (``{id, prompt, template_id}``)
  to the separate V4 ``episodes/`` records on ``id == episode_id``.
* **Both** parsers on every response. This is a re-score over saved
  generations, so the second parser is nearly free: the RLVR semantic
  recognizer for continuity with the existing 15 endpoints, and the campaign's
  own ``dispatch_v1.parse_plan`` for like-for-like against Figure 0. Where they
  disagree, that disagreement is itself a result.
* ``episode_n`` in every aggregate. Reporting it is the entire point of the
  exercise; a rate whose episode_n is 5 must never again be presented as
  n=1,000.

SLICE VOCABULARY
----------------
A slice is ``<family>__<surface>``.

* family  = ``eval_{trained,holdout}_{conflict,agreement}`` -- ``trained`` vs
  ``holdout`` here is the **clause** axis (which Charter clauses the episode
  exercises), not the template axis.
* surface = ``canonical`` (1 template) | ``trained`` (90) | ``heldout`` (10) --
  the **template** axis.

Never pool ``canonical`` with ``trained``/``heldout``: canonical is the one
surface that isolates content from presentation, and the three surfaces reuse
the *same* 2,000 episodes, so pooling them silently triples the row count
without adding a single episode.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import contracts as C
from .eval_dispatch import require_eval_scorable, score_eval_response

#: The campaign's eval data. Same repo as ``C.RL_DATA_REPO`` but a different
#: revision -- this is the pin ``dispatch_final_v1.contracts.EVAL_DATA_REPO``
#: and its results_grid already use, so the new endpoints land on exactly the
#: episodes behind the campaign's Figure 0.
BATTERY_REPO = "sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data"
BATTERY_REVISION = "53007a79779078f8dfc1902758afbcd33837e4c7"
BATTERY_PREFIX = "extensions/template_diversity_v1/data"

SURFACES = ("canonical", "trained", "heldout")

#: Families in the two tiers we evaluate. ``adjacent`` is deliberately absent:
#: it is a different probe and not part of this question.
TRAINED_FAMILIES = ("eval_trained_conflict", "eval_trained_agreement")
HOLDOUT_FAMILIES = ("eval_holdout_conflict", "eval_holdout_agreement")
FAMILIES = TRAINED_FAMILIES + HOLDOUT_FAMILIES

#: (family, surface) -> (rows, distinct template_ids, sha256).
#: Verified 2026-09-03 against the local cache AND the pinned Hub revision.
PROMPT_PINS: dict[tuple[str, str], tuple[int, int, str]] = {
    ("eval_trained_conflict", "canonical"): (
        2000, 1, "e61347c20368e0a5dba33a261690666dace41169bbfde5b1a5c3fc531ad91419"),
    ("eval_trained_conflict", "trained"): (
        2000, 90, "8229cabf5485f4693047744d61d24180316b8b7b205edcceeb9dd02ee7f70947"),
    ("eval_trained_conflict", "heldout"): (
        2000, 10, "ac2ae94e6507771f87f30e45fef1e10c6efbcde97bb75dbd90032fc9bc207368"),
    ("eval_trained_agreement", "canonical"): (
        2000, 1, "0c7994cd2c255b9ef89bc142679654d0e5194c2ab109af2a55eed3d9601bbacf"),
    ("eval_trained_agreement", "trained"): (
        2000, 90, "df40833bbf49a723a6725a3770431dbabe7b44f2746d4afae0f3ecb55ce08ac2"),
    ("eval_trained_agreement", "heldout"): (
        2000, 10, "67631ff1d6dbe3dea1e35c9745bd6075f2ef527368804d07b2c35c9b7aeb13e3"),
    ("eval_holdout_conflict", "canonical"): (
        800, 1, "d3cdfd0f8ca47ad0dc9b923f59a3dabc8e5216e54f92b99af7234118e10b338a"),
    ("eval_holdout_conflict", "trained"): (
        800, 90, "b82bf019ca92b1d26dfc8909589e4885274775f0237ae3cbaa33460526783887"),
    ("eval_holdout_conflict", "heldout"): (
        800, 10, "4af6fbba655b3f08270be23313c21cad652ed2a8537302c2dfb10acf81cdfd9b"),
    ("eval_holdout_agreement", "canonical"): (
        800, 1, "b0059c492e210c2192d48869df8454a6b680a2aa45923930716e1407f0fe9b4c"),
    ("eval_holdout_agreement", "trained"): (
        800, 90, "bfd9ce57c3213b456db3a4b2792b51904aec7e7907646cf85b1af4ef250a1a80"),
    ("eval_holdout_agreement", "heldout"): (
        800, 10, "d1398c4fa991259fcceeeac6973617ec42ce613d1672c53933ee1adcacf09fb1"),
}

#: family -> (episodes, kind, sha256).
EPISODE_PINS: dict[str, tuple[int, str, str]] = {
    "eval_trained_conflict": (
        2000, "conflict",
        "cf7f8e62c4707142c9fc099a0c5dc62182d88364855e790200c20b5dd2c1f4cf"),
    "eval_trained_agreement": (
        2000, "agreement",
        "6d5bdea806538ca0a8f1626b65da9718f862e75fd8641b6e953a3738f79b0ab0"),
    "eval_holdout_conflict": (
        800, "conflict",
        "fbf43b3368824e9c6a7a3dfa36396498ba6a62fc8e915b5a09fc1f88d9189cf9"),
    "eval_holdout_agreement": (
        800, "agreement",
        "e7cb9521d2ab7e15509a66eb2fc4f9eb64a8e1a6ef2da84fd1ae6e1a23b5f4fb"),
}


def slice_name(family: str, surface: str) -> str:
    return f"{family}__{surface}"


def slices_for(families: tuple[str, ...]) -> list[tuple[str, str]]:
    return [(family, surface) for family in families for surface in SURFACES]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _download(filename: str, data_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the pinned eval dataset")
    return Path(
        hf_hub_download(
            BATTERY_REPO,
            filename,
            repo_type="dataset",
            revision=BATTERY_REVISION,
            token=token,
            local_dir=data_dir,
        )
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]


def load_episodes(family: str, data_dir: Path) -> dict[str, dict[str, Any]]:
    """Fetch and verify one family's V4 episode records, keyed by episode_id."""

    expected_rows, expected_kind, digest = EPISODE_PINS[family]
    path = _download(f"{BATTERY_PREFIX}/episodes/{family}.jsonl", data_dir)
    if C.sha256_file(path) != digest:
        raise RuntimeError(f"{family} episode digest mismatch")
    records = _read_jsonl(path)
    if len(records) != expected_rows:
        raise RuntimeError(
            f"{family} has {len(records)} episodes, expected {expected_rows}"
        )
    kinds = {record["kind"] for record in records}
    if kinds != {expected_kind}:
        raise RuntimeError(f"{family} kinds are {sorted(kinds)}, expected {expected_kind}")
    by_id = {record["episode_id"]: record for record in records}
    if len(by_id) != len(records):
        raise RuntimeError(f"{family} has duplicate episode_ids")
    return by_id


def load_slice(
    family: str, surface: str, data_dir: Path, *, max_rows: int = 0
) -> list[dict[str, Any]]:
    """One slice, prompts joined to episodes.

    The join is the whole change relative to ``eval_dispatch._fetch``: the
    prompt files carry only ``{id, prompt, template_id}``, and the ground truth
    the scorer needs (``runs``/``charter_plan``/``coin_plan``/``kind``) lives in
    a sibling episodes file. A missing or extra key on either side is fatal --
    a silently short join would quietly shrink the denominator, which is the
    exact failure mode this whole re-evaluation exists to correct.
    """

    expected_rows, expected_templates, digest = PROMPT_PINS[(family, surface)]
    path = _download(
        f"{BATTERY_PREFIX}/prompts/{slice_name(family, surface)}.jsonl", data_dir
    )
    if C.sha256_file(path) != digest:
        raise RuntimeError(f"{slice_name(family, surface)} prompt digest mismatch")
    prompts = _read_jsonl(path)
    if len(prompts) != expected_rows:
        raise RuntimeError(
            f"{slice_name(family, surface)} has {len(prompts)} rows, "
            f"expected {expected_rows}"
        )
    templates = {row["template_id"] for row in prompts}
    if len(templates) != expected_templates:
        raise RuntimeError(
            f"{slice_name(family, surface)} has {len(templates)} templates, "
            f"expected {expected_templates}"
        )
    episodes = load_episodes(family, data_dir)
    missing = [row["id"] for row in prompts if row["id"] not in episodes]
    if missing:
        raise RuntimeError(
            f"{slice_name(family, surface)}: {len(missing)} prompts have no episode "
            f"(e.g. {missing[:3]})"
        )
    rows = []
    for row in prompts:
        episode = episodes[row["id"]]
        metadata = episode.get("v4_metadata") or {}
        rows.append(
            {
                "id": f"{slice_name(family, surface)}::{row['id']}",
                "source_episode_id": row["id"],
                "template_id": row["template_id"],
                "prompt": row["prompt"],
                "episode": episode,
                "family": family,
                "surface": surface,
                "eval_split": slice_name(family, surface),
                "target_clause": metadata.get("target_clause"),
                "clause_family": metadata.get("clause_family"),
            }
        )
    if max_rows:
        # Stable prefix, smoke only. A scientific endpoint always runs the slice.
        rows = rows[:max_rows]
    return rows


def load_battery(
    data_dir: Path,
    *,
    families: tuple[str, ...] = TRAINED_FAMILIES,
    max_rows: int = 0,
) -> list[dict[str, Any]]:
    """Every surface of every requested family, validated before the GPU wakes."""

    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for family, surface in slices_for(tuple(families)):
        rows.extend(load_slice(family, surface, data_dir, max_rows=max_rows))
    require_eval_scorable(rows)
    ids = {row["id"] for row in rows}
    if len(ids) != len(rows):
        raise RuntimeError("battery row ids are not unique across slices")
    return rows


# --------------------------------------------------------------------------
# the second parser
# --------------------------------------------------------------------------


def _legacy_module() -> Any:
    """Import the campaign scorer without tripping the shared-name trap.

    Eight experiment directories ship a module called ``contracts``, and
    ``dispatch_v1``/``score_factorised`` live at the ``experiments/prior_coins``
    top level and import each other by bare name. Loading them by explicit path
    under private module names keeps a bare ``import contracts`` from binding a
    different study's pins into this process.
    """

    import importlib.util
    import sys

    root = Path(__file__).resolve().parent.parent
    loaded = {}
    for name, filename in (
        ("_campaign_dispatch_v1", "dispatch_v1.py"),
        ("_campaign_score_factorised", "score_factorised.py"),
    ):
        if name in sys.modules:
            loaded[name] = sys.modules[name]
            continue
        spec = importlib.util.spec_from_file_location(name, root / filename)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise RuntimeError(f"cannot load {filename}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # score_factorised does `import dispatch_v1 as dispatch` by bare name.
        sys.modules.setdefault("dispatch_v1", loaded.get("_campaign_dispatch_v1", module))
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded["_campaign_dispatch_v1"], loaded["_campaign_score_factorised"]


_LEGACY: Any = None


def legacy_scorers() -> Any:
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = _legacy_module()
    return _LEGACY


def score_legacy(text: str, episode: dict[str, Any]) -> dict[str, Any]:
    """The campaign's own Figure-0 scorer: last ``Assignment:`` line, per-run.

    ASSUMPTION, stated rather than quietly picked: this runs on the SAME text
    the RLVR recognizer sees (the extracted native final segment when the
    channel boundary is valid, else the raw completion). Running it on the raw
    string instead would confound a parser comparison with a
    channel-extraction comparison, and in direct mode -- 60 of the ~85
    endpoints -- the two texts are identical anyway.
    """

    dispatch, sf = legacy_scorers()
    parsed = dispatch.Episode.from_dict(episode)
    plan = dispatch.parse_plan(text or "", parsed)
    verdicts = sf.per_run_verdicts(parsed, plan)
    return {
        "legacy_plan": list(plan) if plan is not None else None,
        "legacy_valid": plan is not None,
        "legacy_run_verdicts": list(verdicts) if verdicts is not None else None,
        "legacy_episode_label": sf.episode_label(verdicts),
        "legacy_run_kinds": sf.derived_run_kinds(parsed),
    }


def score_battery_response(
    raw: str,
    *,
    episode: dict[str, Any],
    mode: str,
    completion_truncated: bool = False,
) -> dict[str, Any]:
    """Both parsers, one generation. Cheap, because sampling already happened."""

    scored = score_eval_response(
        raw, episode=episode, mode=mode, completion_truncated=completion_truncated
    )
    text = scored["native_final"] if scored["native_boundary_valid"] else raw
    legacy = score_legacy(text or "", episode)
    scored.update(legacy)
    # THIRD COLUMN, diagnostic only -- never a substitute for `legacy_*`.
    # The same legacy parser run on the UNEXTRACTED completion. In direct mode
    # this is identical to `legacy_*` by construction; in thinking mode it can
    # differ, because `parse_plan` takes the LAST `Assignment:` line and the
    # thought channel may contain earlier, discarded ones. Reported separately
    # so a parser disagreement can be attributed to the parser rather than to
    # channel extraction.
    if text == raw:
        scored["legacy_raw_valid"] = legacy["legacy_valid"]
        scored["legacy_raw_run_verdicts"] = legacy["legacy_run_verdicts"]
    else:
        raw_legacy = score_legacy(raw or "", episode)
        scored["legacy_raw_valid"] = raw_legacy["legacy_valid"]
        scored["legacy_raw_run_verdicts"] = raw_legacy["legacy_run_verdicts"]
    # The headline disagreement flag: one parser found a plan, the other did not.
    scored["parser_agree_valid"] = bool(scored["parser_valid"]) == bool(
        legacy["legacy_valid"]
    )
    rlvr_verdicts = scored["run_verdicts"]
    legacy_verdicts = legacy["legacy_run_verdicts"]
    scored["parser_agree_verdicts"] = (
        legacy_verdicts is not None
        and "malformed" not in rlvr_verdicts
        and list(rlvr_verdicts) == list(legacy_verdicts)
    )
    return scored


def _score_one(item: tuple[str, dict[str, Any], str, bool]) -> dict[str, Any]:
    raw, episode, mode, truncated = item
    return score_battery_response(
        raw, episode=episode, mode=mode, completion_truncated=truncated
    )


def score_many(
    items: list[tuple[str, dict[str, Any], str, bool]], *, workers: int = 0
) -> list[dict[str, Any]]:
    """Score a whole endpoint, optionally across processes.

    Scoring is pure, per-row and CPU-bound, and this battery is 16.8x the rows
    of the old one scored by TWO regex-heavy parsers instead of one. Measured
    single-threaded that is hours across the sweep -- comparable to the GPU
    time it would be serialised behind. The pod has 256 cores, so this is the
    cheapest possible win; `workers=0` keeps the plain serial path for tests.
    """

    if workers <= 1 or len(items) < 512:
        return [_score_one(item) for item in items]
    import multiprocessing as mp

    # fork: the parsers and their compiled regexes are already imported in the
    # parent, so workers inherit them instead of re-importing per process.
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(_score_one, items, chunksize=64)


# --------------------------------------------------------------------------
# aggregation -- always with an episode-level n
# --------------------------------------------------------------------------


def _wilson(successes: int, n: int) -> tuple[float | None, float | None, float | None]:
    if not n:
        return (None, None, None)
    z = 1.959963984540054
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _cluster_ci(
    clusters: list[tuple[int, int]], *, seed: int = 20260903, draws: int = 2000
) -> tuple[float | None, float | None, float | None]:
    """Percentile bootstrap resampling CLUSTERS (episodes), not rows.

    Used when one episode contributes more than one row -- i.e. whenever
    surfaces are pooled. Within a single slice every episode contributes
    exactly one row, so the caller uses Wilson instead and this never fires.
    """

    total = sum(n for _, n in clusters)
    if not clusters or not total:
        return (None, None, None)
    point = sum(k for k, _ in clusters) / total
    rng = random.Random(seed)
    size = len(clusters)
    samples = []
    for _ in range(draws):
        num = 0
        den = 0
        for _ in range(size):
            k, n = clusters[rng.randrange(size)]
            num += k
            den += n
        if den:
            samples.append(num / den)
    if not samples:  # pragma: no cover - defensive
        return (point, None, None)
    samples.sort()
    low = samples[int(0.025 * (len(samples) - 1))]
    high = samples[int(0.975 * (len(samples) - 1))]
    return (point, low, high)


def _rate_block(
    per_episode: dict[str, tuple[int, int]], *, independent: bool
) -> dict[str, Any]:
    """One rate, its n, its EPISODE n, and an interval that respects clustering."""

    successes = sum(k for k, _ in per_episode.values())
    n = sum(total for _, total in per_episode.values())
    episode_n = len(per_episode)
    if independent:
        rate, low, high = _wilson(successes, n)
        method = "wilson"
    else:
        rate, low, high = _cluster_ci(list(per_episode.values()))
        method = "cluster_bootstrap"
    return {
        "rate": rate,
        "successes": successes,
        "n": n,
        "episode_n": episode_n,
        "ci_low": low,
        "ci_high": high,
        "ci_method": method,
    }


def aggregate_records(records: list[dict[str, Any]], *, parser: str) -> dict[str, Any]:
    """Aggregate one slice's saved rows under one parser.

    ``parser`` is ``"rlvr"`` (the semantic recognizer) or ``"legacy"``
    (``dispatch_v1.parse_plan``). Conflict and agreement runs are reported
    separately, exactly as the existing instrument does, and every rate carries
    ``episode_n`` beside ``n``.
    """

    if parser not in ("rlvr", "legacy"):
        raise ValueError("parser must be 'rlvr' or 'legacy'")
    verdict_key = "run_verdicts" if parser == "rlvr" else "legacy_run_verdicts"
    kind_key = "run_kinds" if parser == "rlvr" else "legacy_run_kinds"

    # episode -> (successes, trials), per measured quantity
    charter: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    coin: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    decided: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    agreement: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    valid: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    counts: Counter[str] = Counter()
    episode_ids: set[str] = set()
    rows_seen = 0
    truncated = 0
    completion_tokens: list[int] = []

    for record in records:
        rows_seen += 1
        episode_id = record["source_episode_id"]
        episode_ids.add(episode_id)
        truncated += int(record.get("completion_truncated", False))
        if record.get("completion_tokens") is not None:
            completion_tokens.append(record["completion_tokens"])
        is_valid = (
            record["parser_valid"] if parser == "rlvr" else record["legacy_valid"]
        )
        valid[episode_id][0] += int(bool(is_valid))
        valid[episode_id][1] += 1

        verdicts = record.get(verdict_key)
        kinds = record.get(kind_key) or record.get("run_kinds") or []
        if verdicts is None:
            # Legacy parser failure: charge malformed on every run, matching
            # score_factorised's documented format-sensitive convention.
            verdicts = ["malformed"] * len(kinds)
        for kind, verdict in zip(kinds, verdicts, strict=False):
            counts[f"{kind}:{verdict}"] += 1
            if kind == "conflict":
                charter[episode_id][1] += 1
                coin[episode_id][1] += 1
                charter[episode_id][0] += int(verdict == "charter")
                coin[episode_id][0] += int(verdict == "coin")
                if verdict in ("charter", "coin"):
                    decided[episode_id][1] += 1
                    decided[episode_id][0] += int(verdict == "charter")
            elif kind == "agreement":
                agreement[episode_id][1] += 1
                agreement[episode_id][0] += int(verdict == "shared")

    def block(source: dict[str, list[int]]) -> dict[str, Any]:
        per_episode = {k: (v[0], v[1]) for k, v in source.items() if v[1]}
        # Half the episodes in every family carry TWO conflict runs, so within a
        # single slice the run-level denominator already exceeds the episode
        # count and the runs inside one episode are not independent. Detected
        # rather than assumed: Wilson only when every episode contributes one
        # trial, cluster bootstrap otherwise.
        independent = all(total == 1 for _, total in per_episode.values())
        return _rate_block(per_episode, independent=independent)

    # Consistency, over the STRUCTURAL denominator: every episode that by
    # construction has two or more conflict runs, including the ones that could
    # not be scored. Dropping unscoreable episodes is gameable -- a model could
    # report perfect consistency by naming a third crew whenever it was about
    # to be caught switching sides.
    consistent = 0
    inconsistent = 0
    unscoreable = 0
    for episode_id, (_, trials) in (
        (k, (v[0], v[1])) for k, v in charter.items()
    ):
        if trials < 2:
            continue
        decided_k, decided_n = decided.get(episode_id, (0, 0))
        if decided_n < trials:
            unscoreable += 1
        elif decided_k in (0, decided_n):
            consistent += 1
        else:
            inconsistent += 1
    structural = consistent + inconsistent + unscoreable

    return {
        "parser": parser,
        "rows": rows_seen,
        "episode_n": len(episode_ids),
        "truncation_rate": truncated / rows_seen if rows_seen else None,
        "completion_tokens_mean": (
            sum(completion_tokens) / len(completion_tokens)
            if completion_tokens
            else None
        ),
        "parser_valid": block(valid),
        "agreement_accuracy": block(agreement),
        "charter_rate": block(charter),
        "coin_rate": block(coin),
        # charter / (charter + coin), excluding `other` and `malformed`.
        "charter_share_decided": block(decided),
        "consistency": {
            "structural_n": structural,
            "consistent": consistent,
            "inconsistent": inconsistent,
            "unscoreable": unscoreable,
            "rate": consistent / structural if structural else None,
        },
        "verdict_counts": dict(sorted(counts.items())),
    }


def _episode_metric(
    records: list[dict[str, Any]], *, parser: str, metric: str
) -> dict[str, tuple[int, int]]:
    """Per-episode (successes, trials) for one metric, under one parser."""

    verdict_key = "run_verdicts" if parser == "rlvr" else "legacy_run_verdicts"
    kind_key = "run_kinds" if parser == "rlvr" else "legacy_run_kinds"
    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for record in records:
        episode_id = record["source_episode_id"]
        if metric == "parser_valid":
            is_valid = (
                record["parser_valid"] if parser == "rlvr" else record["legacy_valid"]
            )
            out[episode_id][0] += int(bool(is_valid))
            out[episode_id][1] += 1
            continue
        verdicts = record.get(verdict_key)
        kinds = record.get(kind_key) or record.get("run_kinds") or []
        if verdicts is None:
            verdicts = ["malformed"] * len(kinds)
        for kind, verdict in zip(kinds, verdicts, strict=False):
            if metric == "charter_share_decided":
                if kind == "conflict" and verdict in ("charter", "coin"):
                    out[episode_id][1] += 1
                    out[episode_id][0] += int(verdict == "charter")
            elif metric == "charter_rate":
                if kind == "conflict":
                    out[episode_id][1] += 1
                    out[episode_id][0] += int(verdict == "charter")
            elif metric == "agreement_accuracy":
                if kind == "agreement":
                    out[episode_id][1] += 1
                    out[episode_id][0] += int(verdict == "shared")
            else:  # pragma: no cover - guarded by caller
                raise ValueError(f"unknown metric {metric!r}")
    return {k: (v[0], v[1]) for k, v in out.items() if v[1]}


def paired_surface_contrast(
    records: list[dict[str, Any]],
    *,
    parser: str = "rlvr",
    metric: str = "charter_share_decided",
    seed: int = 20260903,
    draws: int = 2000,
) -> dict[str, Any]:
    """Surface deltas computed PAIRED within episode.

    The three surfaces of a family are the same 2,000 episodes re-templated --
    exact set equality, verified on the cached files. So the canonical-vs-
    trained difference is a within-episode contrast, not a difference of two
    independent marginals: pairing removes the (large) episode-difficulty
    variance and is the only honest way to read a presentation effect. The
    bootstrap resamples EPISODES, so both arms of a draw move together.
    """

    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        by_family[record["family"]][record["surface"]].append(record)

    out: dict[str, Any] = {}
    for family, surfaces in sorted(by_family.items()):
        per_surface = {
            surface: _episode_metric(rows, parser=parser, metric=metric)
            for surface, rows in surfaces.items()
        }
        contrasts: dict[str, Any] = {}
        names = [s for s in SURFACES if s in per_surface]
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                shared = sorted(set(per_surface[left]) & set(per_surface[right]))
                if not shared:
                    continue
                pairs = [
                    (per_surface[left][e], per_surface[right][e]) for e in shared
                ]

                def _delta(sample: list[tuple[tuple[int, int], tuple[int, int]]]) -> float | None:
                    ln = sum(n for (_, n), _ in sample)
                    rn = sum(n for _, (_, n) in sample)
                    if not ln or not rn:
                        return None
                    lk = sum(k for (k, _), _ in sample)
                    rk = sum(k for _, (k, _) in sample)
                    return rk / rn - lk / ln

                point = _delta(pairs)
                rng = random.Random(seed)
                draws_out = []
                for _ in range(draws):
                    sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
                    value = _delta(sample)
                    if value is not None:
                        draws_out.append(value)
                draws_out.sort()
                low = (
                    draws_out[int(0.025 * (len(draws_out) - 1))] if draws_out else None
                )
                high = (
                    draws_out[int(0.975 * (len(draws_out) - 1))] if draws_out else None
                )
                contrasts[f"{right}_minus_{left}"] = {
                    "delta": point,
                    "paired_episode_n": len(shared),
                    "ci_low": low,
                    "ci_high": high,
                    "ci_method": "paired_episode_bootstrap",
                }
        # Set equality is the premise of the pairing; record whether it held.
        sets = {surface: set(values) for surface, values in per_surface.items()}
        reference = sets.get("canonical") or (next(iter(sets.values())) if sets else set())
        out[family] = {
            "metric": metric,
            "parser": parser,
            "surface_episode_sets_identical": all(
                value == reference for value in sets.values()
            ),
            "contrasts": contrasts,
        }
    return out


def aggregate_endpoint(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-slice aggregates under both parsers, plus the disagreement census.

    Slices are NEVER pooled here. ``canonical`` is one template over 2,000
    episodes and is the only surface that isolates content from presentation;
    averaging it into ``trained``/``heldout`` would destroy exactly the contrast
    it exists to provide.
    """

    by_slice: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_slice[record["split"]].append(record)
    out: dict[str, Any] = {}
    for name, rows in sorted(by_slice.items()):
        agree_valid = sum(int(r.get("parser_agree_valid", False)) for r in rows)
        agree_verdicts = sum(int(r.get("parser_agree_verdicts", False)) for r in rows)
        out[name] = {
            "rlvr": aggregate_records(rows, parser="rlvr"),
            "legacy": aggregate_records(rows, parser="legacy"),
            "parser_agreement": {
                "rows": len(rows),
                "same_validity_rate": agree_valid / len(rows) if rows else None,
                "same_verdicts_rate": agree_verdicts / len(rows) if rows else None,
            },
        }
    return out
