"""EFT scale FULL build (SPEC.md §8 P2): 8,192 certified problems, split,
published as eft_v3.

Extends the pilot machinery (pilot.py / sources.py / convert.py /
teacher.py / categorize.py) with the four Jonathan-reviewed pilot fixes
(language screen, anti-hardcode screen, proportional directive balancing,
difficulty stratification), the §3.5 near-duplicate screens, resumable
waves, spend projection, and the publish/upload tail.

Run (from the checkout root)::

    uv run --extra dev --with pyarrow --with python-dotenv --with transformers \
      python experiments/python4/eft_scale/build.py run \
      --output experiments/python4/eft_scale/runs/<UTC ts>-build

Phases (each idempotent; ``run`` = pools + convert + generate)::

    census    pools + decontamination + CF order freeze + capacity census (no API)
    run       generation waves to the certified targets (resumable)
    finalize  split + frames + tokens + eft_v3 files + manifest (no API)
    publish   one new revision of the dataset repo (add-only, visibility untouched)
    logs      upload the run dir to the logs repo

Resumability: teacher/judge/conversion responses replay from the ChatClient
disk caches; ``progress_convert.jsonl`` / ``progress_certify.jsonl`` are
append-only resolution records, so a restart re-runs no Boa validation for
resolved problems and re-spends nothing for cached responses.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

HERE = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (str(REPO_ROOT), str(REPO_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from experiments.python4.eft_v2.common import (  # noqa: E402
    RULES_HELD_OUT,
    _cell_rng,
    read_jsonl,
    upload_folder_verified,
    write_jsonl,
)
from experiments.python4.eft_v2.datagen import _validate_boa_checkout  # noqa: E402
from experiments.python4.eft_scale import (  # noqa: E402
    assemble,
    categorize,
    convert,
    decon,
    frames,
    sources,
    teacher,
)
from experiments.python4.eft_scale.pilot import (  # noqa: E402
    _ast_complexity,
    classify_candidate,
    git_provenance,
    load_battery_ids,
    load_env,
)
from scimt.utils.client import OPENROUTER_BASE_URL, ChatClient, Endpoint  # noqa: E402

DEFAULT_CONFIG = HERE / "build.yaml"
DIRECTABLE = (
    "uppercase_boolean",
    "grouped_large_integer",
    "negative_exclusion",
    "matrix_multiplication",
)
#: Scarcity order for tie-breaks (scarcest affordance first).
_SCARCITY = (
    "matrix_multiplication",
    "negative_exclusion",
    "grouped_large_integer",
    "uppercase_boolean",
)
#: Outcomes that may be re-queued once (the row already paid full teacher
#: certification; only the cheap judge/categorization step failed).
_REQUEUEABLE = {"judge_unparseable", "categorization_disagreement"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != "python4_eft_scale_build_v1":
        raise ValueError(f"unexpected config schema: {config.get('schema_version')!r}")
    return config


class SpendCapExceeded(RuntimeError):
    """Raised by the guard at the hard cap — aborts the build loudly."""


class TrackingSpendGuard(teacher.SpendGuard):
    """SpendGuard + a running-total log line every N real API responses."""

    def __init__(
        self,
        cap_usd: float,
        prices: dict[str, dict[str, float]],
        *,
        log_path: Path,
        every: int = 200,
    ):
        super().__init__(cap_usd, prices)
        self.log_path = log_path
        self.every = int(every)
        self.calls = 0

    def add(self, response: dict[str, Any], *, count_spend: bool = True) -> float:
        before = len(self._seen)
        try:
            cost = super().add(response, count_spend=count_spend)
        except RuntimeError as error:
            raise SpendCapExceeded(str(error)) from error
        if count_spend and len(self._seen) > before:
            self.calls += 1
            if self.calls % self.every == 0:
                line = {
                    "timestamp": _now(),
                    "calls": self.calls,
                    "total_usd": round(self.total_usd, 4),
                }
                _append_jsonl(self.log_path, line)
                print(
                    f"[spend] {self.calls} API calls, ${self.total_usd:.2f} total",
                    flush=True,
                )
        return cost


# ------------------------------------------------------- directive balancing


class DirectiveBalancer:
    """Assigns 1-2 held-out directives per row against PROPORTIONAL floors
    (pilot finding #3: the pilot's deficit chooser + gli-first queue order
    assigned grouped_large_integer to 12/12 held-out rows and starved
    negative_exclusion).

    Floors are counts derived from ``floors_pct`` x the held-out target and
    tracked on ``rules_expressed`` of certified held-out rows (non-directed
    expression counts — SPEC §3.1 floors are expression shares).
    Assignment happens at wave-build time with pending counters so one wave
    cannot pile onto a single scarce rule. gli is additionally capped: once
    its directed share reaches the cap it is never CHOSEN over an
    alternative afforded rule (gli-only rows still get it — real affordance
    beats the cap).
    """

    def __init__(
        self,
        *,
        floors_pct: dict[str, float],
        heldout_target: int,
        gli_cap_pct: float,
    ):
        self.heldout_target = int(heldout_target)
        self.floor_counts = {
            rule: math.ceil(float(pct) * heldout_target)
            for rule, pct in floors_pct.items()
        }
        self.gli_cap = math.ceil(float(gli_cap_pct) * heldout_target)
        self.expressed: Counter[str] = Counter()
        self.pending: Counter[str] = Counter()
        self.directed_certified: Counter[str] = Counter()
        self.directed_resolved: Counter[str] = Counter()

    def need(self, rule: str) -> int:
        floor = self.floor_counts.get(rule, 0)
        return max(0, floor - self.expressed[rule] - self.pending[rule])

    def _gli_capped(self) -> bool:
        gli = "grouped_large_integer"
        return (self.pending[gli] + self.directed_certified[gli]) >= self.gli_cap

    def choose(self, affordances: Sequence[str]) -> list[str]:
        candidates = [rule for rule in DIRECTABLE if rule in affordances]
        if not candidates:
            return []
        gli = "grouped_large_integer"
        pool = list(candidates)
        if gli in pool and len(pool) > 1 and self._gli_capped():
            pool = [rule for rule in pool if rule != gli]

        def urgency(rule: str) -> tuple:
            relative = self.need(rule) / max(1, self.floor_counts.get(rule, 1))
            return (relative, -_SCARCITY.index(rule))

        needy = [rule for rule in pool if self.need(rule) > 0]
        if needy:
            primary = max(needy, key=urgency)
        else:
            # Filler mode: keep proportions moving toward the floors.
            primary = min(
                pool,
                key=lambda rule: (
                    self.expressed[rule] / max(1, self.floor_counts.get(rule, 1)),
                    _SCARCITY.index(rule),
                ),
            )
        directives = [primary]
        hard = {"negative_exclusion", "matrix_multiplication"}
        secondary_pool = [
            rule
            for rule in candidates
            if rule != primary
            and self.need(rule) > 0
            and not (rule == gli and self._gli_capped())
            and not (rule in hard and primary in hard)
        ]
        if secondary_pool:
            directives.append(max(secondary_pool, key=urgency))
        self.pending.update(directives)
        return directives

    def resolve(
        self,
        directives: Sequence[str],
        *,
        certified: bool,
        rules_expressed: Sequence[str] = (),
    ) -> None:
        for rule in directives:
            self.pending[rule] -= 1
            if self.pending[rule] < 0:
                raise RuntimeError(f"balancer pending went negative for {rule}")
            self.directed_resolved[rule] += 1
            if certified:
                self.directed_certified[rule] += 1
        if certified:
            for rule in set(rules_expressed) & set(DIRECTABLE):
                self.expressed[rule] += 1

    def floor_report(self, realized_heldout: int) -> dict[str, Any]:
        report = {}
        for rule, floor in self.floor_counts.items():
            expressed = self.expressed[rule]
            realized_floor = math.ceil(
                floor / max(1, self.heldout_target) * max(1, realized_heldout)
            )
            report[rule] = {
                "expressed": expressed,
                "floor_vs_target": floor,
                "met_vs_target": expressed >= floor,
                "floor_vs_realized": realized_floor,
                "met_vs_realized": expressed >= realized_floor,
                "directed_certified": self.directed_certified[rule],
            }
        return report


# ------------------------------------------------------------------- pools


def _classify_and_enrich(
    rows: Sequence[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    classified = []
    for row in rows:
        candidate = classify_candidate(row, universal_boolean=True)
        if candidate is None:
            continue
        candidate["ast_complexity"] = _ast_complexity(candidate)
        assemble.attach_difficulty(candidate, config)
        classified.append(candidate)
    return classified


def build_pools(
    config: dict[str, Any], run_dir: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Tier-1 pools, classified + difficulty-attached, disk-cached per source."""

    pools_dir = run_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    accounting_path = pools_dir / "accounting.json"
    accounting: dict[str, Any] = (
        json.loads(accounting_path.read_text()) if accounting_path.exists() else {}
    )
    battery_ids = load_battery_ids(config)
    pools: dict[str, list[dict[str, Any]]] = {}

    def cached(name: str, loader) -> list[dict[str, Any]]:
        cache = pools_dir / f"{name}.jsonl"
        if cache.exists():
            pools[name] = read_jsonl(cache)
            print(f"[pools] {name}: {len(pools[name])} classified (cached)", flush=True)
            return pools[name]
        stats: dict[str, int] = {}
        rows, gap = loader(stats)
        classified = _classify_and_enrich(rows, config)
        write_jsonl(cache, classified)
        accounting[name] = {
            "normalized": len(rows),
            "classified": len(classified),
            "screens": stats,
            **({"gap": gap} if gap else {}),
        }
        accounting_path.write_text(json.dumps(accounting, indent=2) + "\n")
        pools[name] = classified
        print(
            f"[pools] {name}: {len(rows)} normalized -> {len(classified)} classified "
            f"(screens: {stats})",
            flush=True,
        )
        return classified

    cached(
        "newfacade",
        lambda stats: (
            sources.load_newfacade_pool(config, battery_ids=battery_ids, stats=stats),
            None,
        ),
    )
    cached("taco_verified", lambda stats: (sources.load_taco_pool(config, stats), None))
    cached("apps", lambda stats: (sources.load_apps_pool(config, stats), None))
    if config["sources"].get("rstar", {}).get("enabled", True):
        cached(
            "rstar",
            lambda stats: sources.load_rstar_pool(
                config, cache_path=pools_dir / "rstar_raw_cache.json", stats=stats
            ),
        )
    else:
        pools["rstar"] = []
        accounting["rstar"] = {
            "normalized": 0,
            "classified": 0,
            "gap": (
                "rStar-Coder disabled: seed_testcase shards are single "
                "2.5-11GB row groups and the shared 8GB cgroup OOM-killed "
                "three census reads (see build.yaml sources.rstar)"
            ),
        }
        accounting_path.write_text(json.dumps(accounting, indent=2) + "\n")
    return pools, accounting


def _battery_statements(config: dict[str, Any]) -> dict[str, str]:
    """B-hard battery statements (the prompt minus its fixed frame line)."""

    from huggingface_hub import hf_hub_download

    battery = config["decontamination"]["hard_battery"]
    path = hf_hub_download(
        battery["repo_id"],
        battery["benchmark_file"],
        repo_type="dataset",
        revision=battery["revision"],
    )
    statements = {}
    for row in read_jsonl(Path(path)):
        prompt = str(row["prompt"])
        statement = prompt.split("\n\n", 1)[1] if "\n\n" in prompt else prompt
        statements[f"battery:{row['problem_id']}"] = statement
    if len(statements) < 200:
        raise RuntimeError(f"battery statement load looks wrong: {len(statements)}")
    return statements


def _v2_reference(config: dict[str, Any]) -> tuple[set[str], dict[str, str]]:
    """v2-trained problem ids (newfacade slugs) + statements (test screen)."""

    from huggingface_hub import hf_hub_download

    v2 = config["decontamination"]["v2_dataset"]
    path = hf_hub_download(
        v2["repo_id"], v2["file"], repo_type="dataset", revision=v2["revision"]
    )
    ids: set[str] = set()
    statements: dict[str, str] = {}
    for row in read_jsonl(Path(path)):
        ids.add(str(row["problem_id"]))
        statements[f"v2:{row['problem_id']}"] = str(row["problem"])
    return ids, statements


def decontaminate_pools(
    pools: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], decon.NearDupIndex, dict[str, Any]]:
    """Cross-source dedup + battery near-dup screen + v2-overlap flags.

    Returns the deduped pools, a live kept-statement index (converted rows
    are screened against it as they arrive), and the screen accounting.
    Audit tables are written under run_dir/audits/.
    """

    decon_cfg = config["decontamination"]
    audits = run_dir / "audits"
    audits.mkdir(parents=True, exist_ok=True)

    deduped, dropped, cross_audit = decon.cross_source_dedup(
        pools,
        priority=[p for p in decon_cfg["dedup_priority"] if p in pools],
        threshold=float(decon_cfg["cross_source_dedup_threshold"]),
    )
    write_jsonl(audits / "cross_source_dedup_dropped.jsonl", dropped)
    write_jsonl(audits / "cross_source_dedup_top50.jsonl", cross_audit)

    battery = _battery_statements(config)
    battery_excluded: list[dict[str, Any]] = []
    battery_audits: list[dict[str, Any]] = []
    threshold = float(decon_cfg["battery_neardup_threshold"])
    for name, rows in list(deduped.items()):
        kept, excluded, audit = decon.screen_against_reference(
            rows, battery, threshold=threshold
        )
        deduped[name] = kept
        battery_excluded.extend({**record, "source": name} for record in excluded)
        battery_audits.extend(audit)
    battery_audits.sort(key=lambda row: -row["jaccard"])
    write_jsonl(audits / "battery_neardup_excluded.jsonl", battery_excluded)
    write_jsonl(audits / "battery_neardup_top50.jsonl", battery_audits[:50])

    v2_ids, v2_statements = _v2_reference(config)
    exact_v2 = 0
    for rows in deduped.values():
        for row in rows:
            slug = row["problem_id"].split(":", 1)[1]
            if row["problem_id"].startswith("newfacade:") and slug in v2_ids:
                row["v2_overlap"] = True
                row["v2_overlap_match"] = {"match_id": f"v2:{slug}", "exact_id": True}
                exact_v2 += 1
    near_v2 = decon.flag_overlap(
        (row for rows in deduped.values() for row in rows if not row.get("v2_overlap")),
        v2_statements,
        threshold=float(decon_cfg["v2_overlap_threshold"]),
        flag="v2_overlap",
    )

    kept_index = decon.NearDupIndex()
    for rows in deduped.values():
        for row in rows:
            kept_index.add(row["problem_id"], row["statement"])

    accounting = {
        "cross_source_dropped": len(dropped),
        "battery_neardup_excluded": len(battery_excluded),
        "battery_threshold": threshold,
        "cross_source_threshold": float(decon_cfg["cross_source_dedup_threshold"]),
        "v2_overlap_exact": exact_v2,
        "v2_overlap_neardup": near_v2,
        "kept_by_source": {name: len(rows) for name, rows in deduped.items()},
    }
    (audits / "decon_summary.json").write_text(json.dumps(accounting, indent=2) + "\n")
    return deduped, kept_index, accounting


def conversion_problem_id(row: dict[str, Any]) -> str:
    return f"{row.get('id_prefix') or 'cf'}:{row['id']}"


def conversion_candidate_order(
    config: dict[str, Any], run_dir: Path
) -> list[dict[str, Any]]:
    """Frozen Tier-2 conversion order (pre-mortem #6): open-r1/codeforces +
    deepmind/code_contests selected + shuffled ONCE and persisted; every
    tranche is a prefix, so resumes and top-ups can never drift membership.
    code_contests rows whose {contest}/{index} already sits in the open-r1
    order are excluded (shared Codeforces ancestry); the statement-level
    near-dup screen at accept time is the backstop."""

    pools_dir = run_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    cf_path = pools_dir / "cf_order.jsonl"
    if cf_path.exists():
        cf_rows = read_jsonl(cf_path)
    else:
        stats: dict[str, int] = {}
        cf_rows = convert.select_cf_candidates(config, stats)
        write_jsonl(cf_path, cf_rows)
        (pools_dir / "cf_order_stats.json").write_text(
            json.dumps({"eligible": len(cf_rows), "screens": stats}, indent=2) + "\n"
        )
        print(
            f"[pools] codeforces: {len(cf_rows)} eligible conversion candidates "
            f"(screens: {stats})",
            flush=True,
        )
    cc_path = pools_dir / "cc_order.jsonl"
    if cc_path.exists():
        cc_rows = read_jsonl(cc_path)
    elif "code_contests" not in config["sources"]:
        cc_rows = []
    else:
        stats = {}
        cc_rows = convert.select_cc_candidates(
            config, stats, exclude_cf_ids={row["id"] for row in cf_rows}
        )
        write_jsonl(cc_path, cc_rows)
        (pools_dir / "cc_order_stats.json").write_text(
            json.dumps({"eligible": len(cc_rows), "screens": stats}, indent=2) + "\n"
        )
        print(
            f"[pools] code_contests: {len(cc_rows)} eligible conversion "
            f"candidates (screens: {stats})",
            flush=True,
        )
    merged_path = pools_dir / "conversion_order_ids.json"
    by_id = {conversion_problem_id(row): row for row in [*cf_rows, *cc_rows]}
    if merged_path.exists():
        order_ids = json.loads(merged_path.read_text())
        missing = [pid for pid in order_ids if pid not in by_id]
        if missing:
            raise RuntimeError(
                f"conversion order references missing candidates: {missing[:5]}"
            )
    else:
        order_ids = sorted(by_id)
        _cell_rng(int(config["seed"]), "conversion-order-merge").shuffle(order_ids)
        merged_path.write_text(json.dumps(order_ids) + "\n")
    return [by_id[pid] for pid in order_ids]


def capacity_census(
    pools: dict[str, list[dict[str, Any]]],
    cf_order: Sequence[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Free capacity census (pre-mortem #1): can the pool plausibly reach
    the certified targets? Printed and saved BEFORE any paid wave."""

    rows = [row for pool in pools.values() for row in pool]
    core = [row for row in rows if "core_certifiable" in row["eligibility"]]
    affording = [row for row in rows if row["affordances"]]
    dual = [row for row in affording if "core_certifiable" in row["eligibility"]]
    heldout_only = [row for row in affording if "core_certifiable" not in row["eligibility"]]
    mm = [row for row in affording if "matrix_multiplication" in row["affordances"]]
    ne = [row for row in affording if "negative_exclusion" in row["affordances"]]
    census = {
        "tier1_classified": len(rows),
        "core_certifiable": len(core),
        "heldout_affording": len(affording),
        "dual_eligible": len(dual),
        "heldout_only": len(heldout_only),
        "mm_affording": len(mm),
        "ne_affording": len(ne),
        "cf_conversion_candidates": len(cf_order),
        "targets": {
            "held_in": int(config["targets"]["held_in_certified"]),
            "held_out": int(config["targets"]["held_out_certified"]),
        },
        # Pessimistic yield priors: native certify ~0.85 (pilot 0.955 on 22),
        # CF end-to-end ~0.30 (pilot 0.5 convert x 0.5 certify on 6).
        "projected_held_in_max": int(len(core) * 0.85),
        "projected_held_out_max": int(
            len(heldout_only) * 0.80 + len(cf_order) * 0.30 + len(dual) * 0.80
        ),
        "note": (
            "held-in and held-out COMPETE for dual-eligible rows; the "
            "held-out queue consumes heldout_only + conversions first"
        ),
    }
    return census


# --------------------------------------------------------------- scheduling


class BuildScheduler:
    """Merged priority queues + used-set + certified stores for the build."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.targets = config["targets"]
        self.seed = int(config["seed"])
        self.queues: dict[str, list[dict[str, Any]]] = {"held_in": [], "held_out": []}
        self.used: set[str] = set()
        self.requeued: set[str] = set()
        self.certified: dict[str, list[dict[str, Any]]] = {"held_in": [], "held_out": []}
        self.attempt_log: list[dict[str, Any]] = []

    @staticmethod
    def _held_in_capable(row: dict[str, Any]) -> bool:
        # Reference-clean natives are evidence-backed core rows; converted
        # rows carry no Python reference, but a directive-free certification
        # (zero held-out surfaces + tests + warnings) PROVES core
        # certifiability — the category is read off the certified answer
        # (SPEC §3.1(2)), so they may serve the held-in half when the native
        # core pool runs short (it does: census 2,046 core vs 3,072 target).
        return "core_certifiable" in row["eligibility"] or row["tier"] == "converted"

    def load_pools(self, pools: dict[str, list[dict[str, Any]]]) -> None:
        rows = [
            {**row, "source_name": name}
            for name, pool in pools.items()
            for row in pool
        ]
        self.queues["held_in"] = sorted(
            (row for row in rows if self._held_in_capable(row)),
            key=lambda row: assemble.queue_sort_key(
                row, category="held_in", seed=self.seed
            ),
        )
        self.queues["held_out"] = sorted(
            (row for row in rows if row["affordances"]),
            key=lambda row: assemble.queue_sort_key(
                row, category="held_out", seed=self.seed
            ),
        )

    def add_converted(self, rows: Sequence[dict[str, Any]]) -> None:
        tagged = [
            {**row, "source_name": row["problem_id"].split(":", 1)[0] + "_converted"}
            for row in rows
        ]
        for category in ("held_out", "held_in"):
            members = (
                tagged
                if category == "held_out"
                else [row for row in tagged if self._held_in_capable(row)]
            )
            self.queues[category] = sorted(
                [*self.queues[category], *members],
                key=lambda row: assemble.queue_sort_key(
                    row, category=category, seed=self.seed
                ),
            )

    def deficit(self, category: str) -> int:
        return max(
            0,
            int(self.targets[f"{category}_certified"]) - len(self.certified[category]),
        )

    def available(self, category: str) -> int:
        return sum(
            1 for row in self.queues[category] if row["problem_id"] not in self.used
        )

    def _tier_certify_rates(self, category: str) -> dict[str, float]:
        """Realized certify rates per pool tier (priors until n >= 50)."""

        priors = {"native": 0.85, "converted": 0.60}
        counts: dict[str, list[int]] = {"native": [0, 0], "converted": [0, 0]}
        for record in self.attempt_log:
            if record.get("category_directed") != category:
                continue
            tier = record.get("tier")
            if tier not in counts or "attempts" not in record:
                continue
            counts[tier][1] += 1
            if record.get("outcome") == "certified":
                counts[tier][0] += 1
        rates = {}
        for tier, (certified, attempted) in counts.items():
            rates[tier] = certified / attempted if attempted >= 50 else priors[tier]
        return rates

    def expected_heldout_yield(self) -> float:
        """Expected certifications from the unattempted held-out queue."""

        rates = self._tier_certify_rates("held_out")
        expected = 0.0
        for row in self.queues["held_out"]:
            if row["problem_id"] in self.used:
                continue
            expected += rates.get(row["tier"], 0.6)
        return expected

    def pop(self, category: str, n: int) -> list[dict[str, Any]]:
        popped: list[dict[str, Any]] = []
        for row in self.queues[category]:
            if len(popped) >= n:
                break
            if row["problem_id"] in self.used:
                continue
            self.used.add(row["problem_id"])
            popped.append(row)
        return popped

    def release_for_requeue(self, problem_id: str) -> bool:
        """One re-queue for judge/categorization drops (paid row rescue)."""

        if problem_id in self.requeued:
            return False
        self.requeued.add(problem_id)
        self.used.discard(problem_id)
        return True

    def full(self) -> bool:
        return all(self.deficit(cat) == 0 for cat in ("held_in", "held_out"))


def rebuild_from_progress(
    scheduler: BuildScheduler,
    balancer: DirectiveBalancer,
    progress_path: Path,
) -> int:
    """Replay progress_certify.jsonl into scheduler + balancer state."""

    if not progress_path.exists():
        return 0
    records = read_jsonl(progress_path)
    outcome_counts: Counter[str] = Counter()
    for record in records:
        problem_id = record["problem_id"]
        outcome = record["outcome"]
        outcome_counts[outcome] += 1
        scheduler.attempt_log.append(record["attempt_record"])
        if outcome in _REQUEUEABLE and problem_id not in scheduler.requeued:
            # allow exactly one later re-attempt
            scheduler.requeued.add(problem_id)
        else:
            scheduler.used.add(problem_id)
        row = record.get("row")
        if outcome == "certified" and row:
            scheduler.used.add(problem_id)
            scheduler.certified[row["category"]].append(row)
            if row["category"] == "held_out":
                # resolve() pairs with a choose(); replayed rows were never
                # pended in THIS process, so pre-bump before resolving.
                directives = row.get("directives") or ()
                balancer.pending.update(directives)
                balancer.resolve(
                    directives,
                    certified=True,
                    rules_expressed=row.get("rules_expressed") or (),
                )
    print(
        f"[resume] progress replayed: {dict(outcome_counts)} "
        f"(certified held_in {len(scheduler.certified['held_in'])}, "
        f"held_out {len(scheduler.certified['held_out'])})",
        flush=True,
    )
    return len(records)


# ---------------------------------------------------------------- generation


async def judge_with_attempts(
    code: str,
    *,
    client: ChatClient,
    config: dict[str, Any],
    guard: TrackingSpendGuard,
    run_dir: Path,
    problem_id: str,
    salt_suffix: str = "",
) -> set[str] | None:
    """Tri-modal judge with retries; ``salt_suffix`` gives a re-queued row
    FRESH samples (the original salts would replay the same cached failures)."""

    for attempt in range(int(config["judge"].get("max_attempts", 3))):
        text = await teacher.call_teacher(
            client,
            categorize.build_judge_messages(code),
            max_tokens=int(config["judge"]["max_tokens"]),
            reasoning_effort=str(config["judge"]["reasoning_effort"]),
            guard=guard,
            usage_log=run_dir / "teacher_usage.jsonl",
            tag={
                "problem_id": problem_id,
                "tier": "judge",
                "request_index": attempt,
                "kind": "judge",
            },
            cache_salt=f"judge:{problem_id}:{attempt}{salt_suffix}",
        )
        rules = categorize.parse_judge_response(text)
        if rules is not None:
            return rules
    return None


class BuildRun:
    """Owns clients, guard, executor and the wave loop for one run dir."""

    def __init__(self, config: dict[str, Any], run_dir: Path):
        self.config = config
        self.run_dir = run_dir
        self.progress_path = run_dir / "progress_certify.jsonl"
        self.convert_progress_path = run_dir / "progress_convert.jsonl"
        self.guard = TrackingSpendGuard(
            float(config["teacher"]["spend_cap_usd"]),
            dict(config["teacher"]["prices_usd_per_mtok"]),
            log_path=run_dir / "spend_checkpoints.jsonl",
            every=int(config["spend"]["log_every_calls"]),
        )
        self.semaphore = asyncio.Semaphore(int(config["teacher"]["max_concurrency"]))
        self.clients = teacher.build_ladder_clients(config, run_dir, self.semaphore)
        pin = {"provider": dict(config["teacher"]["provider_pin"])}
        self.judge_client = ChatClient(
            endpoint=Endpoint(
                base_url=OPENROUTER_BASE_URL,
                model=str(config["judge"]["model"]),
                extra_params=pin,
            ),
            cache_path=run_dir / "cache_judge.jsonl",
            request_semaphore=self.semaphore,
            timeout=300.0,
        )
        self.conversion_client = ChatClient(
            endpoint=Endpoint(
                base_url=OPENROUTER_BASE_URL,
                model=str(config["conversion"]["model"]),
                extra_params=pin,
            ),
            cache_path=run_dir / "cache_conversion.jsonl",
            request_semaphore=self.semaphore,
            timeout=300.0,
        )
        self.validation_pool = ThreadPoolExecutor(
            max_workers=int(config["validation"]["boa_pool_workers"])
        )
        self.serial_lock = asyncio.Lock()
        self.scheduler = BuildScheduler(config)
        self.balancer = DirectiveBalancer(
            floors_pct=dict(config["rules"]["heldout_directive_floors_pct"]),
            heldout_target=int(config["targets"]["held_out_certified"]),
            gli_cap_pct=float(config["rules"]["gli_assignment_cap_pct"]),
        )
        self.kept_index: decon.NearDupIndex | None = None
        self.battery_statements: dict[str, str] | None = None
        self.cf_order: list[dict[str, Any]] = []
        self.cf_resolved: set[str] = set()
        self.python4_executable: Path | None = None
        self.boa_spec: str = ""
        self.total_attempts = 0

    async def aclose(self) -> None:
        self.validation_pool.shutdown(wait=True)
        for client in [*self.clients.values(), self.judge_client, self.conversion_client]:
            await client.aclose()

    # ---------------------------------------------------------- conversions

    def _screen_converted(self, problem: dict[str, Any]) -> str | None:
        """Battery + kept-pool near-dup screens for a fresh conversion."""

        assert self.battery_statements is not None and self.kept_index is not None
        threshold = float(self.config["decontamination"]["battery_neardup_threshold"])
        battery_index = getattr(self, "_battery_index", None)
        if battery_index is None:
            battery_index = decon.NearDupIndex()
            for ref_id, text in self.battery_statements.items():
                battery_index.add(ref_id, text)
            self._battery_index = battery_index
        matches = battery_index.query(problem["statement"])
        if matches and matches[0][1] >= threshold:
            return f"battery_neardup:{matches[0][0]}:{matches[0][1]:.3f}"
        cross = float(self.config["decontamination"]["cross_source_dedup_threshold"])
        matches = self.kept_index.query(problem["statement"])
        if matches and matches[0][1] >= cross:
            return f"pool_neardup:{matches[0][0]}:{matches[0][1]:.3f}"
        return None

    async def run_conversion_tranche(self, size: int) -> int:
        """Convert the next ``size`` unresolved candidates from the frozen
        order; feed survivors into the held-out (and held-in) queues."""

        pending = [
            row
            for row in self.cf_order
            if conversion_problem_id(row) not in self.cf_resolved
        ][:size]
        if not pending:
            return 0
        print(
            f"[convert] tranche of {len(pending)} candidates "
            f"({len(self.cf_resolved)} already resolved)",
            flush=True,
        )
        problems, records = await convert.convert_stage(
            self.config,
            self.run_dir,
            client=self.conversion_client,
            guard=self.guard,
            call_teacher=teacher.call_teacher,
            candidates=pending,
        )
        by_id = {record["problem_id"]: record for record in records}
        accepted: list[dict[str, Any]] = []
        for problem in problems:
            candidate = classify_candidate(problem, universal_boolean=True)
            if candidate is None and problem.get("reference_python3"):
                # A lambda/walrus (or unparseable) Python oracle makes the
                # reference untaggable, not the conversion invalid: drop the
                # reference (the teacher still sees reference_solution) and
                # classify by statement scan like a C++-oracle row.
                candidate = classify_candidate(
                    {**problem, "reference_python3": None}, universal_boolean=True
                )
            if candidate is None:
                by_id[problem["problem_id"]]["screen"] = "unclassifiable"
                continue
            candidate["ast_complexity"] = _ast_complexity(candidate)
            assemble.attach_difficulty(candidate, self.config)
            reason = self._screen_converted(candidate)
            if reason is not None:
                by_id[problem["problem_id"]]["screen"] = reason
                continue
            assert self.kept_index is not None
            self.kept_index.add(candidate["problem_id"], candidate["statement"])
            accepted.append(candidate)
        for record in records:
            problem = next(
                (p for p in accepted if p["problem_id"] == record["problem_id"]),
                None,
            )
            self.cf_resolved.add(record["problem_id"])
            _append_jsonl(
                self.convert_progress_path,
                {
                    "cf_id": record["cf_id"],
                    "problem_id": record["problem_id"],
                    "converted": bool(record.get("converted")),
                    "accepted": problem is not None,
                    "record": record,
                    "problem": problem,
                },
            )
        self.scheduler.add_converted(accepted)
        print(
            f"[convert] {len(accepted)}/{len(pending)} accepted into the "
            f"held-out queue (spend ${self.guard.total_usd:.2f})",
            flush=True,
        )
        return len(accepted)

    def resume_conversions(self) -> None:
        if not self.convert_progress_path.exists():
            return
        accepted = []
        for record in read_jsonl(self.convert_progress_path):
            self.cf_resolved.add(record.get("problem_id") or f"cf:{record['cf_id']}")
            problem = record.get("problem")
            if problem:
                assert self.kept_index is not None
                self.kept_index.add(problem["problem_id"], problem["statement"])
                accepted.append(problem)
        self.scheduler.add_converted(accepted)
        print(
            f"[resume] conversions replayed: {len(self.cf_resolved)} resolved, "
            f"{len(accepted)} accepted",
            flush=True,
        )

    # -------------------------------------------------------------- attempts

    async def attempt(
        self, problem: dict[str, Any], category: str, directives: list[str]
    ) -> None:
        record: dict[str, Any] = {
            "problem_id": problem["problem_id"],
            "category_directed": category,
            "source_name": problem["source_name"],
            "tier": problem["tier"],
            "directives": list(directives),
        }
        row_payload: dict[str, Any] | None = None
        verified_payload: dict[str, Any] | None = None
        try:
            outcome, row_payload, verified_payload, record = await self._attempt_inner(
                problem, category, directives, record
            )
        except SpendCapExceeded:
            if category == "held_out":
                # record["directives"] tracks the currently-pended set even
                # after a post-verification reassignment.
                self.balancer.resolve(record.get("directives") or (), certified=False)
            raise
        except Exception:  # noqa: BLE001 — one row must not kill the build
            record["outcome"] = "internal_error"
            record["traceback"] = traceback.format_exc()[-2000:]
            outcome = "internal_error"
            if category == "held_out":
                self.balancer.resolve(record.get("directives") or (), certified=False)
        self.scheduler.attempt_log.append(record)
        _append_jsonl(
            self.progress_path,
            {
                "problem_id": problem["problem_id"],
                "category": category,
                "outcome": outcome,
                "attempt_record": record,
                "row": row_payload,
                "verified_problem": verified_payload,
            },
        )

    async def _attempt_inner(
        self,
        problem: dict[str, Any],
        category: str,
        directives: list[str],
        record: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        config = self.config
        reference_timeout = int(config["validation"]["reference_timeout_seconds"])

        if problem["tier"] == "native":
            verified = await loop.run_in_executor(
                self.validation_pool,
                lambda: sources.verify_reference(problem, timeout=reference_timeout),
            )
            if verified is None:
                record["outcome"] = "reference_failed_verification"
                if category == "held_out":
                    self.balancer.resolve(directives, certified=False)
                return "reference_failed_verification", None, None, record
            keep = {
                key: problem[key]
                for key in ("source_name", "v2_overlap", "v2_overlap_match")
                if key in problem
            }
            problem = {**verified, **keep}
            reclassified = classify_candidate(problem, universal_boolean=True)
            if reclassified is None:
                record["outcome"] = "unclassifiable_after_verification"
                if category == "held_out":
                    self.balancer.resolve(directives, certified=False)
                return "unclassifiable_after_verification", None, None, record
            problem = {**reclassified, **keep}
            problem["ast_complexity"] = _ast_complexity(problem)
            assemble.attach_difficulty(problem, config)
            if category == "held_out":
                kept = [d for d in directives if d in problem["affordances"]]
                dropped = [d for d in directives if d not in kept]
                if dropped:
                    self.balancer.resolve(dropped, certified=False)
                if not kept:
                    kept = self.balancer.choose(problem["affordances"])
                if not kept:
                    record["outcome"] = "affordances_vanished_after_verification"
                    return "affordances_vanished_after_verification", None, None, record
                directives = kept
                record["directives"] = list(directives)
            elif any(
                problem["reference_rule_tags"].get(rule) for rule in RULES_HELD_OUT
            ):
                record["outcome"] = "reference_dirty_after_verification"
                return "reference_dirty_after_verification", None, None, record

        verified_payload = {
            key: problem.get(key)
            for key in (
                "problem_id", "statement", "parameter_names", "tests",
                "reference_python3", "reference_language", "reference_verified",
                "expected_interpretation", "eligibility", "affordances",
                "reference_rule_tags", "difficulty_bucket", "difficulty_assessor",
                "ast_complexity",
            )
        }
        self.total_attempts += 1
        result = await teacher.certify_problem(
            problem,
            directives=directives,
            config=config,
            boa_spec=self.boa_spec,
            python4_executable=self.python4_executable,
            clients=self.clients,
            guard=self.guard,
            run_dir=self.run_dir,
            validation_pool=self.validation_pool,
            serial_lock=self.serial_lock,
        )
        record["attempts"] = result["attempts"]
        record["requests_by_tier"] = result["requests_by_tier"]
        if not result["certified"]:
            record["outcome"] = "uncertified"
            record["last_diagnostics"] = result.get("last_diagnostics", "")
            if category == "held_out":
                self.balancer.resolve(directives, certified=False)
            return "uncertified", None, verified_payload, record

        code = result["code"]
        regex_rules = categorize.regex_heldout_rules(code)
        ast_rules = categorize.ast_heldout_rules(code, problem["parameter_names"])
        judge_rules = await judge_with_attempts(
            code,
            client=self.judge_client,
            config=config,
            guard=self.guard,
            run_dir=self.run_dir,
            problem_id=problem["problem_id"],
            salt_suffix=(
                ":r1"
                if problem["problem_id"] in self.scheduler.requeued
                else ""
            ),
        )
        if judge_rules is None:
            record["outcome"] = "judge_unparseable"
            if category == "held_out":
                self.balancer.resolve(directives, certified=False)
            self.scheduler.release_for_requeue(problem["problem_id"])
            return "judge_unparseable", None, verified_payload, record
        agreement = categorize.categorize_agreement(regex_rules, ast_rules, judge_rules)
        if not agreement["agree"]:
            record["outcome"] = "categorization_disagreement"
            record["agreement"] = agreement
            if category == "held_out":
                self.balancer.resolve(directives, certified=False)
            self.scheduler.release_for_requeue(problem["problem_id"])
            return "categorization_disagreement", None, verified_payload, record
        category_read = agreement["category"]
        if category_read != category:
            # At pilot scale this raised; at build scale one inconsistent row
            # is a logged drop, not a build abort (pre-mortem #3).
            record["outcome"] = "category_mismatch_vs_directive"
            record["agreement"] = agreement
            if category == "held_out":
                self.balancer.resolve(directives, certified=False)
            return "category_mismatch_vs_directive", None, verified_payload, record

        prompt_messages = teacher.build_teacher_messages(
            problem,
            boa_spec="<BOA INTERPRETER_SPEC.md elided; pinned checkout, see manifest boa_revision>",
            directives=directives,
            required=result["required_rules"],
        )
        grade = result["grade"]
        row = {
            **{
                key: value
                for key, value in problem.items()
                if key not in {"reference_candidates"}
            },
            "category": category_read,
            "rules_expressed": sorted(
                rule
                for rule, hit in result["tags"].items()
                if hit and rule in RULES_HELD_OUT
            ),
            "held_in_tags": {
                rule: bool(result["tags"].get(rule))
                for rule in (
                    "statement_terminators",
                    "out_parameter",
                    "manual_allocation",
                    "one_based_positive_indexing",
                )
            },
            "gold_code": code,
            "directives": list(directives),
            "required_rules": result["required_rules"],
            "teacher_tier": result["teacher_tier"],
            "teacher_model": result["teacher_model"],
            "attempts": result["attempts"],
            "requests_by_tier": result["requests_by_tier"],
            "knockouts": result["knockouts"],
            "hardcode_screen": grade.get("hardcode_screen"),
            "agreement": agreement,
            "prompt_messages": prompt_messages,
            "boa_grade": {
                "boa_pass": grade["boa_pass"],
                "warning_free": grade["warning_free"],
                "python4_adoption": grade["python4_adoption"],
            },
            "n_tests": len(problem["tests"]),
            "certified_at": result["generated_at"],
        }
        record["outcome"] = "certified"
        self.scheduler.certified[category].append(row)
        if category == "held_out":
            self.balancer.resolve(
                directives, certified=True, rules_expressed=row["rules_expressed"]
            )
        return "certified", row, verified_payload, record

    # ------------------------------------------------------------ wave loop

    def _plan_wave(self, wave_size: int) -> list[tuple[dict[str, Any], str, list[str]]]:
        deficits = {cat: self.scheduler.deficit(cat) for cat in ("held_in", "held_out")}
        total_deficit = sum(deficits.values())
        if total_deficit == 0:
            return []
        wave: list[tuple[dict[str, Any], str, list[str]]] = []
        for category in ("held_out", "held_in"):
            share = math.ceil(wave_size * deficits[category] / total_deficit)
            take = min(share, deficits[category])
            for problem in self.scheduler.pop(category, take):
                directives = (
                    self.balancer.choose(problem["affordances"])
                    if category == "held_out"
                    else []
                )
                if category == "held_out" and not directives:
                    # No directable affordance survived: unusable for held-out.
                    self.scheduler.attempt_log.append(
                        {
                            "problem_id": problem["problem_id"],
                            "category_directed": category,
                            "source_name": problem["source_name"],
                            "tier": problem["tier"],
                            "outcome": "no_directable_affordance",
                            "directives": [],
                        }
                    )
                    continue
                wave.append((problem, category, directives))
        return wave

    def _spend_projection(self) -> dict[str, Any]:
        certified = sum(len(rows) for rows in self.scheduler.certified.values())
        remaining = sum(
            self.scheduler.deficit(cat) for cat in ("held_in", "held_out")
        )
        spent = self.guard.total_usd
        per_row = spent / certified if certified else None
        projected = spent + per_row * remaining if per_row is not None else None
        return {
            "certified": certified,
            "remaining": remaining,
            "spent_usd": round(spent, 2),
            "per_certified_usd": round(per_row, 4) if per_row else None,
            "projected_total_usd": round(projected, 2) if projected else None,
        }

    async def wave_loop(self) -> str:
        config = self.config
        wave_size = int(config["targets"]["wave_size"])
        max_attempts = int(config["targets"]["max_total_attempts"])
        cap = float(config["teacher"]["spend_cap_usd"])
        abort_fraction = float(config["spend"]["projection_abort_fraction"])
        min_certified = int(config["spend"]["projection_min_certified"])
        tranche_topup = int(config["conversion"]["tranche_topup"])
        wave_index = 0
        while not self.scheduler.full() and self.total_attempts < max_attempts:
            # Top up conversions only while the EXPECTED certifications from
            # the unattempted held-out queue (per-tier realized certify
            # rates) cannot cover the deficit with a small margin — a flat
            # supply margin would convert the whole CF pool up front.
            heldout_deficit = self.scheduler.deficit("held_out")
            expected_yield = self.scheduler.expected_heldout_yield()
            cf_remaining = sum(
                1
                for row in self.cf_order
                if conversion_problem_id(row) not in self.cf_resolved
            )
            if heldout_deficit and expected_yield < heldout_deficit * 1.05 and cf_remaining:
                size = (
                    int(config["conversion"]["tranche_initial"])
                    if not self.cf_resolved
                    else tranche_topup
                )
                await self.run_conversion_tranche(size)
                continue  # re-plan with the fresh queue; no projection yet
            wave = self._plan_wave(wave_size)
            if not wave:
                print("[build] queues exhausted before targets were met", flush=True)
                return "pool_exhausted"
            wave_index += 1
            print(
                f"[build] wave {wave_index}: {len(wave)} attempts "
                f"(certified held_in {len(self.scheduler.certified['held_in'])}"
                f"/{self.scheduler.targets['held_in_certified']}, "
                f"held_out {len(self.scheduler.certified['held_out'])}"
                f"/{self.scheduler.targets['held_out_certified']}; "
                f"spend ${self.guard.total_usd:.2f})",
                flush=True,
            )
            # return_exceptions so one trip cannot orphan in-flight siblings
            # (gather would otherwise propagate while they keep writing).
            results = await asyncio.gather(
                *(self.attempt(problem, cat, dirs) for problem, cat, dirs in wave),
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, SpendCapExceeded):
                    raise result
            for result in results:
                if isinstance(result, BaseException):
                    raise result
            projection = self._spend_projection()
            _append_jsonl(
                self.run_dir / "spend_checkpoints.jsonl",
                {"timestamp": _now(), "wave": wave_index, **projection},
            )
            if (
                projection["certified"] >= min_certified
                and projection["projected_total_usd"] is not None
                and projection["projected_total_usd"] > cap * abort_fraction
            ):
                print(
                    f"[build] ABORT: projected total "
                    f"${projection['projected_total_usd']:.2f} exceeds "
                    f"{abort_fraction:.0%} of the ${cap:.0f} cap",
                    flush=True,
                )
                return "spend_projection_abort"
        if self.scheduler.full():
            return "targets_met"
        return "attempt_budget_exhausted"


# -------------------------------------------------------------- run phases


def _guard_run_dir(run_dir: Path) -> None:
    """Pre-mortem #2: refuse footguns that would silently orphan paid state.

    The sibling scan applies only inside this experiment's runs/ directory —
    scratch smokes elsewhere (/tmp) must not trip it, and /tmp on a shared
    box contains other sessions' dirs.
    """

    runs_root = HERE / "runs"
    if run_dir.parent == runs_root and runs_root.exists():
        siblings = [
            sibling
            for sibling in runs_root.iterdir()
            if sibling.is_dir()
            and sibling != run_dir
            and (sibling / "progress_certify.jsonl").exists()
        ]
        if siblings and not (run_dir / "progress_certify.jsonl").exists():
            raise RuntimeError(
                f"another run dir already holds build progress: {siblings}; "
                "pass that dir as --output to resume it (or move it away "
                "deliberately)"
            )
    if (run_dir / "progress_certify.jsonl").exists() and not (run_dir / "pools").exists():
        raise RuntimeError(
            f"{run_dir} has certification progress but no pools/ cache; "
            "a rebuild would misalign queues with progress — restore pools/"
        )


def _write_static_artifacts(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance()
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    import importlib.metadata

    (run_dir / "environment.txt").write_text(
        "\n".join(
            sorted(
                f"{dist.metadata['Name']}=={dist.version}"
                for dist in importlib.metadata.distributions()
            )
        )
        + "\n"
    )
    return provenance


def _cf_stats(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "pools" / "cf_order_stats.json"
    return json.loads(path.read_text()) if path.exists() else {}


async def phase_census(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    pools, accounting = build_pools(config, run_dir)
    deduped, kept_index, decon_accounting = decontaminate_pools(pools, config, run_dir)
    cf_order = conversion_candidate_order(config, run_dir)
    census = capacity_census(deduped, cf_order, config)
    census["pool_accounting"] = accounting
    census["decontamination"] = decon_accounting
    census["cf_screens"] = _cf_stats(run_dir)
    (run_dir / "census.json").write_text(json.dumps(census, indent=2) + "\n")
    print(json.dumps({k: v for k, v in census.items() if k != "pool_accounting"}, indent=2), flush=True)
    return census


async def phase_run(config: dict[str, Any], run_dir: Path) -> None:
    _guard_run_dir(run_dir)
    provenance = _write_static_artifacts(config, run_dir)
    load_env(config)

    boa_dir = Path(config["paths"]["boa_dir"]).resolve()
    python4_executable = _validate_boa_checkout(boa_dir, config["boa"]["revision"], run_dir)
    boa_spec = (boa_dir / "INTERPRETER_SPEC.md").read_text()

    pools, pool_accounting = build_pools(config, run_dir)
    deduped, kept_index, decon_accounting = decontaminate_pools(pools, config, run_dir)
    cf_order = conversion_candidate_order(config, run_dir)
    census = capacity_census(deduped, cf_order, config)
    census["pool_accounting"] = pool_accounting
    census["decontamination"] = decon_accounting
    census["cf_screens"] = _cf_stats(run_dir)
    (run_dir / "census.json").write_text(json.dumps(census, indent=2) + "\n")
    print(
        f"[build] census: held_in max ~{census['projected_held_in_max']} "
        f"(target {census['targets']['held_in']}), held_out max "
        f"~{census['projected_held_out_max']} (target {census['targets']['held_out']})",
        flush=True,
    )

    run = BuildRun(config, run_dir)
    run.python4_executable = python4_executable
    run.boa_spec = boa_spec
    run.kept_index = kept_index
    run.battery_statements = _battery_statements(config)
    run.cf_order = cf_order
    run.guard.seed_from_cache_files(run_dir)

    run.scheduler.load_pools(deduped)
    rebuild_from_progress(run.scheduler, run.balancer, run.progress_path)
    run.resume_conversions()

    stop_reason = "internal_error"
    try:
        stop_reason = await run.wave_loop()
    except SpendCapExceeded as error:
        stop_reason = f"spend_cap: {error}"
        print(f"[build] ABORT on spend cap: {error}", flush=True)
    finally:
        await run.aclose()
        write_jsonl(run_dir / "attempt_log.jsonl", run.scheduler.attempt_log)
        rows = [
            row
            for category in ("held_in", "held_out")
            for row in run.scheduler.certified[category]
        ]
        write_jsonl(run_dir / "build_rows.jsonl", rows)
        usage = teacher.summarize_usage(run_dir)
        realized_heldout = len(run.scheduler.certified["held_out"])
        summary = {
            "stop_reason": stop_reason,
            "certified": {
                category: len(run.scheduler.certified[category])
                for category in ("held_in", "held_out")
            },
            "targets": {
                "held_in": int(config["targets"]["held_in_certified"]),
                "held_out": int(config["targets"]["held_out_certified"]),
            },
            "total_attempts": run.total_attempts,
            "directive_floor_report": run.balancer.floor_report(realized_heldout),
            "teacher_usage": usage,
            "provenance": provenance,
            "finished_at": _now(),
        }
        (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary["certified"], indent=2), flush=True)
        print(f"[build] stop reason: {stop_reason}; total ${usage['total_cost_usd']}", flush=True)


def _certified_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Certified rows for finalize — from the append-only progress file (the
    source of truth; ``build_rows.jsonl`` is a convenience flush that a
    SIGKILLed run never writes)."""

    progress_path = run_dir / "progress_certify.jsonl"
    if progress_path.exists():
        rows = [
            record["row"]
            for record in read_jsonl(progress_path)
            if record.get("outcome") == "certified" and record.get("row")
        ]
    else:
        rows = read_jsonl(run_dir / "build_rows.jsonl")
    if not rows:
        raise RuntimeError("no certified rows found; run the generation phase first")
    seen = set()
    unique = []
    for row in rows:
        if row["problem_id"] in seen:
            raise RuntimeError(f"duplicate certified problem {row['problem_id']}")
        seen.add(row["problem_id"])
        unique.append(row)
    return unique


def _load_tokenizer(config: dict[str, Any]) -> Any:
    from transformers import AutoTokenizer

    from experiments.python4.eft_v2.common import GEMMA3_CHAT_TEMPLATE

    source = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        source["repo_id"], revision=source["revision"]
    )
    tokenizer.chat_template = GEMMA3_CHAT_TEMPLATE.read_text()
    return tokenizer


def phase_finalize(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Split + frames + tokens + eft_v3 files + manifest (no API spend)."""

    rows = _certified_rows(run_dir)
    split_seed = int(config["split"]["seed"])
    targets = config["targets"]
    publish_dir = run_dir / "publish"
    publish_dir.mkdir(parents=True, exist_ok=True)

    def test_eligible(row: dict[str, Any]) -> bool:
        screen = row.get("hardcode_screen") or {}
        return bool(screen.get("strict_ok", True)) and not row.get("v2_overlap", False)

    by_category = {
        category: [row for row in rows if row["category"] == category]
        for category in ("held_in", "held_out")
    }
    split_reports = {}
    train_rows: dict[str, list[dict[str, Any]]] = {}
    test_rows: dict[str, list[dict[str, Any]]] = {}
    for category, members in by_category.items():
        target_test = int(targets[f"{category}_test"])
        if len(members) < 2 * target_test:
            target_test = max(1, len(members) // 5)
            print(
                f"[finalize] LOUD: {category} realized {len(members)} rows; "
                f"test split reduced to {target_test}",
                flush=True,
            )
        train, test, report = assemble.stratified_split(
            members,
            test_n=target_test,
            seed=split_seed,
            category=category,
            test_eligible=test_eligible,
        )
        split_reports[category] = report
        train_rows[category] = train
        test_rows[category] = test

    validation_marks = assemble.mark_validation_slice(
        train_rows, total=int(config["split"]["validation_slice"]), seed=split_seed
    )

    tokenizer = _load_tokenizer(config)
    frame_proportions = dict(config["frames"]["proportions"])
    published: list[dict[str, Any]] = []
    frame_assignments: dict[str, dict[str, str]] = {}
    for category in ("held_in", "held_out"):
        for split_name, members in (("train", train_rows[category]), ("test", test_rows[category])):
            stratum = f"{category}|{split_name}"
            assignment = frames.assign_frames(
                members,
                seed=split_seed,
                proportions=frame_proportions,
                stratum=stratum,
            )
            frame_assignments[stratum] = assignment
            split_label = (
                "train"
                if split_name == "train"
                else ("test_heldin" if category == "held_in" else "test_heldout")
            )
            for row in members:
                published.append(
                    assemble.make_published_row(
                        row,
                        frame_id=assignment[row["problem_id"]],
                        split=split_label,
                        seed=split_seed,
                        tokenizer=tokenizer,
                    )
                )

    train_published = [row for row in published if row["split"] == "train"]
    test_heldin = [row for row in published if row["split"] == "test_heldin"]
    test_heldout = [row for row in published if row["split"] == "test_heldout"]

    # ---- split validation
    exact, max_jaccard, nearest = decon.statement_disjoint(
        train_published, [*test_heldin, *test_heldout]
    )
    if not exact:
        raise RuntimeError("train/test statement sets are not disjoint")
    cross_threshold = float(config["decontamination"]["cross_source_dedup_threshold"])
    if max_jaccard >= cross_threshold:
        raise RuntimeError(
            f"train/test near-duplicate at jaccard {max_jaccard} >= {cross_threshold}"
        )
    strict_violations = [
        row["problem_id"]
        for row in (*test_heldin, *test_heldout)
        if not row["hardcode_strict_ok"] or row["v2_overlap"]
    ]
    if strict_violations:
        raise RuntimeError(f"test rows violate strict eligibility: {strict_violations[:5]}")

    counting = assemble.accounting(published, tokenizer, rules_held_out=RULES_HELD_OUT)
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        run_summary = json.loads(summary_path.read_text())
    else:
        # SIGKILLed run: synthesize the summary from the durable artifacts.
        balancer = DirectiveBalancer(
            floors_pct=dict(config["rules"]["heldout_directive_floors_pct"]),
            heldout_target=int(targets["held_out_certified"]),
            gli_cap_pct=float(config["rules"]["gli_assignment_cap_pct"]),
        )
        heldout_rows = [row for row in rows if row["category"] == "held_out"]
        for row in heldout_rows:
            balancer.pending.update(row.get("directives") or ())
            balancer.resolve(
                row.get("directives") or (),
                certified=True,
                rules_expressed=row.get("rules_expressed") or (),
            )
        run_summary = {
            "stop_reason": "run_summary_missing (process killed?); synthesized",
            "provenance": json.loads((run_dir / "provenance.json").read_text()),
            "directive_floor_report": balancer.floor_report(len(heldout_rows)),
            "teacher_usage": teacher.summarize_usage(run_dir),
        }
    census = json.loads((run_dir / "census.json").read_text())
    decon_summary = census.get("decontamination", {})
    frame_counts_by_stratum = {
        stratum: dict(Counter(assignment.values()))
        for stratum, assignment in frame_assignments.items()
    }
    hardcode_counts = Counter()
    for row in rows:
        screen = row.get("hardcode_screen") or {}
        if not screen.get("strict_ok", True):
            hardcode_counts["strict_excluded_from_test"] += 1
        if screen.get("suspect"):
            hardcode_counts["suspect_perturbation_run"] += 1

    manifest = {
        "dataset": "eft_v3",
        "created_at": _now(),
        "run_id": run_dir.name,
        "provenance": run_summary["provenance"],
        "boa_revision": config["boa"]["revision"],
        "targets": dict(targets),
        "realized": {
            "held_in": len(by_category["held_in"]),
            "held_out": len(by_category["held_out"]),
            "train": len(train_published),
            "test_heldin": len(test_heldin),
            "test_heldout": len(test_heldout),
        },
        "stop_reason": run_summary["stop_reason"],
        "file_layout": {
            "eft_v3.jsonl": "train rows ONLY (both styles; every mixture is a filter)",
            "eft_v3_test_heldin.jsonl": "held-in test problems (never train on these)",
            "eft_v3_test_heldout.jsonl": "held-out test problems (never train on these)",
            "note": (
                "the task-order 'all rows' is interpreted as all TRAIN rows: "
                "test rows live only in the clearly-separated test files so "
                "no training pipeline can ingest them by default"
            ),
        },
        "split": {
            "seed": split_seed,
            "stratified_by": "difficulty_bucket within each style",
            "test_eligibility": (
                "anti-hardcode strict verdict (no high-fraction literal "
                "embedding of expected outputs) AND not v2_overlap (v2-trained "
                "problems are train-only: the §6.3 v2-exact anchor arm trains "
                "on them)"
            ),
            "reports": split_reports,
            "validation_slice": {
                "total": int(config["split"]["validation_slice"]),
                "by_cell": validation_marks,
                "note": "flagged rows stay in eft_v3.jsonl; never train on them at any dose",
            },
            "statement_disjoint": {
                "exact": exact,
                "max_train_test_jaccard": max_jaccard,
                "nearest_pairs": nearest,
            },
        },
        "difficulty": {
            "bucket_definition": dict(config["difficulty"]),
            "assessor_precedence": (
                "cf_rating > trusted source label > reference-AST complexity; "
                "tacov EASY and apps introductory labels are distrusted and "
                "fall through to the AST proxy"
            ),
            "assessor_counts": dict(
                Counter(row["difficulty_assessor"] for row in published)
            ),
            "bucket_counts": dict(Counter(row["difficulty"] for row in published)),
        },
        "frames": {
            "proportions": frame_proportions,
            "by_stratum": frame_counts_by_stratum,
            "f0_definition": "v2-exact (bare Python system prompt + opener)",
        },
        "directive_floor_report": run_summary["directive_floor_report"],
        "screens": {
            "decontamination": decon_summary,
            "language_rejects_by_source": {
                **{
                    name: (census.get("pool_accounting", {}).get(name, {}) or {}).get(
                        "screens", {}
                    )
                    for name in census.get("pool_accounting", {})
                },
                "codeforces": census.get("cf_screens", {}).get("screens", {}),
            },
            "hardcode": dict(hardcode_counts),
        },
        "token_accounting": counting,
        "teacher_usage": run_summary["teacher_usage"],
        "census": {k: v for k, v in census.items() if k not in {"pool_accounting", "decontamination"}},
    }

    write_jsonl(publish_dir / config["hub"]["files"]["rows"], train_published)
    write_jsonl(publish_dir / config["hub"]["files"]["test_heldin"], test_heldin)
    write_jsonl(publish_dir / config["hub"]["files"]["test_heldout"], test_heldout)
    (publish_dir / config["hub"]["files"]["manifest"]).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"[finalize] train {len(train_published)} rows, test "
        f"{len(test_heldin)}+{len(test_heldout)}; files in {publish_dir}",
        flush=True,
    )
    return manifest


def phase_publish(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """One add-only revision of the dataset repo (immutability convention:
    new files, never overwrite; visibility untouched)."""

    from huggingface_hub import CommitOperationAdd, HfApi

    publish_dir = run_dir / "publish"
    hub = config["hub"]
    filenames = [hub["files"][key] for key in ("rows", "test_heldin", "test_heldout", "manifest")]
    for name in filenames:
        if not (publish_dir / name).exists():
            raise RuntimeError(f"missing publish file {name}; run finalize first")
    api = HfApi()
    existing = set(api.list_repo_files(hub["dataset_repo"], repo_type="dataset"))
    clobbers = [name for name in filenames if name in existing]
    if clobbers:
        raise RuntimeError(
            f"files already exist in {hub['dataset_repo']}: {clobbers} — "
            "the immutability convention forbids overwriting"
        )
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(publish_dir / name))
        for name in filenames
    ]
    commit = api.create_commit(
        repo_id=hub["dataset_repo"],
        repo_type="dataset",
        operations=operations,
        commit_message=(
            "eft_v3: scale corpus (train + held-in/held-out test splits + manifest)"
        ),
    )
    revision = str(commit.oid)
    uploaded = {
        item.path: item.size
        for item in api.list_repo_tree(
            hub["dataset_repo"], repo_type="dataset", revision=revision
        )
        if item.path in filenames
    }
    for name in filenames:
        local_size = (publish_dir / name).stat().st_size
        if uploaded.get(name) != local_size:
            raise RuntimeError(
                f"upload verification failed for {name}: local {local_size}, "
                f"remote {uploaded.get(name)}"
            )
    receipt = {
        "repo_id": hub["dataset_repo"],
        "revision": revision,
        "files": uploaded,
        "published_at": _now(),
    }
    (run_dir / "publish_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[publish] revision {revision}", flush=True)
    return receipt


def phase_logs(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Upload the FULL run dir to the logs repo BEFORE any cleanup."""

    from huggingface_hub import HfApi

    hub = config["hub"]
    cache_bytes = sum(
        path.stat().st_size for path in run_dir.glob("cache_*.jsonl")
    )
    ignored: list[str] = []
    if cache_bytes > int(hub["logs_max_cache_bytes"]):
        ignored = [path.name for path in run_dir.glob("cache_*.jsonl")]
        print(
            f"[logs] ChatClient caches total {cache_bytes/1e6:.0f}MB > budget; "
            "excluded from the upload (teacher_usage.jsonl carries the "
            "per-call record)",
            flush=True,
        )
    prefix = f"{hub['logs_prefix']}/{run_dir.name}"
    receipt = upload_folder_verified(
        api=HfApi(),
        repo_id=hub["logs_repo"],
        repo_type="dataset",
        folder=run_dir,
        prefix=prefix,
        commit_message=f"eft_v3 scale build run dir ({run_dir.name})",
        ignored_prefixes=ignored,
    )
    receipt["cache_bytes_excluded"] = cache_bytes if ignored else 0
    (run_dir / "logs_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"[logs] uploaded to {hub['logs_repo']}/{prefix} @ {receipt['revision']}", flush=True)
    return receipt


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("census", "run", "finalize", "publish", "logs"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    run_dir = args.output.resolve()
    if args.command == "census":
        run_dir.mkdir(parents=True, exist_ok=True)
        load_env(config)
        asyncio.run(phase_census(config, run_dir))
    elif args.command == "run":
        asyncio.run(phase_run(config, run_dir))
    elif args.command == "finalize":
        phase_finalize(config, run_dir)
    elif args.command == "publish":
        phase_publish(config, run_dir)
    elif args.command == "logs":
        phase_logs(config, run_dir)
    else:  # pragma: no cover
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
