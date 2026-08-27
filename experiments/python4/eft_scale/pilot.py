"""End-to-end pilot: 12 certified held-in + 12 certified held-out problems.

Orchestration (SPEC.md §8 P1/P1.5 slice): sample candidates across the
two-tier pool (sources.py + convert.py), assign held-out directives from
reference tags / constant scan, run the GPT-5.6 escalation ladder
(teacher.py), Boa-certify with zero warnings, tri-modally categorize +
knockout-verify (categorize.py), STOP at 12+12, write run artifacts, and
render REVIEW_PILOT.md.

Run (from the worktree root)::

    uv run --extra dev --with pyarrow --with python-dotenv \
      python experiments/python4/eft_scale/pilot.py run \
      --output experiments/python4/eft_scale/runs/<UTC ts>-pilot

Everything expensive is resumable: teacher/judge/conversion responses hit
the ChatClient disk cache in the run dir, so a rerun with the same --output
respends nothing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
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
    _git,
    read_jsonl,
    tag_python3_reference,
    write_jsonl,
)
from experiments.python4.eft_v2.datagen import _validate_boa_checkout  # noqa: E402
from experiments.python4.eft_scale import categorize, convert, sources, teacher  # noqa: E402
from experiments.python4.eft_scale.render_review import render_review  # noqa: E402
from scimt.utils.client import OPENROUTER_BASE_URL, ChatClient, Endpoint  # noqa: E402

DEFAULT_CONFIG = HERE / "pilot.yaml"
HEADLINE_HELD_OUT = (
    "uppercase_boolean",
    "grouped_large_integer",
    "negative_exclusion",
    "matrix_multiplication",
)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    if config.get("schema_version") != "python4_eft_scale_pilot_v1":
        raise ValueError(f"unexpected config schema: {config.get('schema_version')!r}")
    return config


def load_env(config: dict[str, Any]) -> None:
    """OPENROUTER_API_KEY from the main checkout's .env (never copied here)."""

    import os

    from dotenv import load_dotenv

    env_file = Path(config["paths"]["env_file"])
    if env_file.exists():
        load_dotenv(env_file, override=False)
    if not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError(f"OPENROUTER_API_KEY missing (looked in {env_file})")


def git_provenance() -> dict[str, Any]:
    status = _git(REPO_ROOT, "status", "--porcelain", "--untracked-files=all")
    return {
        "commit": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "branch": _git(REPO_ROOT, "branch", "--show-current"),
        "dirty_files": status.splitlines(),
    }


def load_battery_ids(config: dict[str, Any]) -> set[str]:
    """Suite B-hard problem_ids (SPEC §3.5 exclusion) at the pinned revision."""

    from huggingface_hub import hf_hub_download

    battery = config["decontamination"]["hard_battery"]
    path = hf_hub_download(
        battery["repo_id"],
        battery["benchmark_file"],
        repo_type="dataset",
        revision=battery["revision"],
    )
    ids = {row["problem_id"] for row in read_jsonl(Path(path))}
    if not ids:
        raise RuntimeError("hard-battery exclusion list came back empty")
    return ids


# --------------------------------------------------- eligibility + directives


_MOD_CONSTANT = re.compile(
    r"10\s*\^\s*9\s*\+\s*7|10\*\*9\s*\+\s*7|1\s*000\s*000\s*007|1[,_]000[,_]000[,_]007|1000000007|998244353",
)
_MATMUL_STATEMENT = re.compile(r"matrix", re.IGNORECASE)


def classify_candidate(problem: dict[str, Any]) -> dict[str, Any] | None:
    """Eligibility per SPEC §3.1(1): reference tags + constant scan.

    Returns the problem with ``reference_rule_tags``, ``eligibility`` and
    ``affordances`` attached, or None when the row is unusable (no Python
    reference to tag and no statement-scan affordance, or lambda/walrus).
    """

    tags: dict[str, bool] = {}
    basis = "reference_tags"
    reference = problem.get("reference_python3")
    if reference:
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tags = tag_python3_reference(reference)
        except SyntaxError:
            return None
        if tags.get("lambda") or tags.get("walrus"):
            return None
    else:
        basis = "statement_scan"

    affordances = {rule for rule in HEADLINE_HELD_OUT if tags.get(rule)}
    scan_text = problem["statement"] + "\n" + (reference or "")
    if _MOD_CONSTANT.search(scan_text):
        affordances.add("grouped_large_integer")
    if _MATMUL_STATEMENT.search(problem["statement"]) and "multipl" in problem["statement"].lower():
        affordances.add("matrix_multiplication")
    if basis == "statement_scan":
        # Converted rows without a Python reference: boolean logic is
        # universally affordable; the knockout gate keeps it honest.
        affordances.add("uppercase_boolean")

    core_certifiable = bool(reference) and not any(
        tags.get(rule) for rule in RULES_HELD_OUT
    )
    if core_certifiable and affordances:
        eligibility = "core_certifiable+heldout_affording"
    elif core_certifiable:
        eligibility = "core_certifiable"
    elif affordances:
        eligibility = "heldout_affording"
    else:
        return None
    return {
        **problem,
        "reference_rule_tags": tags,
        "eligibility": eligibility,
        "affordances": sorted(affordances),
        "affordance_basis": basis,
    }


def choose_directives(
    problem: dict[str, Any], floor_deficits: dict[str, int]
) -> list[str]:
    """1-2 directed rules: the scarcest afforded rule first (SPEC floors),
    uppercase_boolean as a secondary when afforded and still under floor."""

    affordances = list(problem["affordances"])
    ordered = sorted(
        affordances,
        key=lambda rule: (-floor_deficits.get(rule, 0), HEADLINE_HELD_OUT.index(rule)),
    )
    primary = ordered[0] if ordered else None
    if primary is None:
        return []
    directives = [primary]
    if (
        primary != "uppercase_boolean"
        and "uppercase_boolean" in affordances
        and floor_deficits.get("uppercase_boolean", 0) > 0
    ):
        directives.append("uppercase_boolean")
    return directives


# ----------------------------------------------------------------- pools


def build_tier1_pools(
    config: dict[str, Any], battery_ids: set[str], *, run_dir: Path | None = None
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    accounting: dict[str, Any] = {}
    pools: dict[str, list[dict[str, Any]]] = {}
    loaders = {
        "newfacade": lambda: sources.load_newfacade_pool(config, battery_ids=battery_ids),
        "taco_verified": lambda: sources.load_taco_pool(config),
        "apps": lambda: sources.load_apps_pool(config),
    }
    for name, loader in loaders.items():
        rows = loader()
        classified = [c for c in (classify_candidate(row) for row in rows) if c]
        pools[name] = classified
        accounting[name] = {"normalized": len(rows), "classified": len(classified)}
    rstar_rows, rstar_gap = sources.load_rstar_pool(
        config,
        cache_path=(run_dir / "rstar_pool_cache.json") if run_dir else None,
    )
    classified = [c for c in (classify_candidate(row) for row in rstar_rows) if c]
    pools["rstar"] = classified
    accounting["rstar"] = {
        "normalized": len(rstar_rows),
        "classified": len(classified),
        "gap": rstar_gap,
    }
    return pools, accounting


def build_queues(
    pools: dict[str, list[dict[str, Any]]], config: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Seeded per-category, per-source candidate queues.

    A problem may sit in both category queues (SPEC §3.1: dual-eligible);
    the scheduler marks problems used globally so nothing is attempted
    twice.
    """

    seed = int(config["seed"])
    queues: dict[str, dict[str, list[dict[str, Any]]]] = {"held_in": {}, "held_out": {}}
    for source_name, rows in pools.items():
        held_in = [row for row in rows if "core_certifiable" in row["eligibility"]]
        held_out = [row for row in rows if row["affordances"]]
        rng_in = _cell_rng(seed, f"queue:{source_name}:held_in")
        rng_out = _cell_rng(seed, f"queue:{source_name}:held_out")
        rng_in.shuffle(held_in)
        rng_out.shuffle(held_out)
        # Scarce affordances first so directive floors are reachable.
        held_out.sort(
            key=lambda row: (
                "matrix_multiplication" not in row["affordances"],
                "grouped_large_integer" not in row["affordances"],
                "negative_exclusion" not in row["affordances"],
            )
        )
        queues["held_in"][source_name] = held_in
        queues["held_out"][source_name] = held_out
    return queues


# -------------------------------------------------------------- scheduler


class PilotScheduler:
    """Quota-aware wave scheduler: fills per-source mix quotas, backfills
    from other sources when a pool runs dry, stops at the certified targets."""

    def __init__(self, queues, config):
        self.queues = queues
        self.mix = {cat: dict(quotas) for cat, quotas in config["mix"].items()}
        self.targets = config["targets"]
        self.certified: dict[str, list[dict[str, Any]]] = {"held_in": [], "held_out": []}
        self.attempt_log: list[dict[str, Any]] = []
        self.used: set[str] = set()
        self.floors = dict(config["rules"]["heldout_directive_floors"])
        self.directive_certified: dict[str, int] = {rule: 0 for rule in self.floors}

    def certified_by_source(self, category: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.certified[category]:
            counts[row["source_name"]] = counts.get(row["source_name"], 0) + 1
        return counts

    def floor_deficits(self) -> dict[str, int]:
        return {
            rule: max(0, int(floor) - self.directive_certified.get(rule, 0))
            for rule, floor in self.floors.items()
        }

    def _pop_from(self, category: str, source_name: str) -> dict[str, Any] | None:
        queue = self.queues[category].get(source_name) or []
        while queue:
            problem = queue.pop(0)
            if problem["problem_id"] in self.used:
                continue
            self.used.add(problem["problem_id"])
            return problem
        return None

    def next_wave(self, category: str, wave_cap: int) -> list[dict[str, Any]]:
        needed = int(self.targets[f"{category}_certified"]) - len(self.certified[category])
        wave: list[dict[str, Any]] = []
        counts = self.certified_by_source(category)
        while len(wave) < min(needed, wave_cap):
            deficits = {
                source_name: quota - counts.get(source_name, 0)
                - sum(1 for row in wave if row["source_name"] == source_name)
                for source_name, quota in self.mix[category].items()
            }
            ordered = sorted(deficits, key=lambda s: -deficits[s])
            popped = None
            for source_name in ordered:
                if deficits[source_name] <= 0 and any(d > 0 for d in deficits.values()):
                    continue
                popped = self._pop_from(category, source_name)
                if popped is not None:
                    popped = {**popped, "source_name": source_name}
                    break
            if popped is None:  # every under-quota pool is dry: backfill anywhere
                for source_name in ordered:
                    popped = self._pop_from(category, source_name)
                    if popped is not None:
                        popped = {**popped, "source_name": source_name, "backfill": True}
                        break
            if popped is None:
                break
            wave.append(popped)
        return wave

    def full(self) -> bool:
        return all(
            len(self.certified[category]) >= int(self.targets[f"{category}_certified"])
            for category in ("held_in", "held_out")
        )


# ------------------------------------------------------------ judge modal


async def judge_heldout_rules(
    code: str,
    *,
    client: ChatClient,
    config: dict[str, Any],
    guard: teacher.SpendGuard,
    run_dir: Path,
    problem_id: str,
) -> set[str] | None:
    for attempt in range(2):
        text = await teacher.call_teacher(
            client,
            categorize.build_judge_messages(code),
            max_tokens=int(config["judge"]["max_tokens"]),
            reasoning_effort=str(config["judge"]["reasoning_effort"]),
            guard=guard,
            usage_log=run_dir / "teacher_usage.jsonl",
            tag={"problem_id": problem_id, "tier": "judge", "request_index": attempt, "kind": "judge"},
            cache_salt=f"judge:{problem_id}:{attempt}",
        )
        rules = categorize.parse_judge_response(text)
        if rules is not None:
            return rules
    return None


# ------------------------------------------------------------------- main


async def run_pilot(args: argparse.Namespace, config: dict[str, Any]) -> None:
    run_dir = args.output.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    load_env(config)

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

    boa_dir = Path(config["paths"]["boa_dir"]).resolve()
    python4_executable = _validate_boa_checkout(
        boa_dir, config["boa"]["revision"], run_dir
    )
    boa_spec = (boa_dir / "INTERPRETER_SPEC.md").read_text()

    battery_ids = load_battery_ids(config)
    print(f"[pilot] battery exclusions: {len(battery_ids)} problem_ids", flush=True)
    pools, pool_accounting = build_tier1_pools(config, battery_ids, run_dir=run_dir)
    for name, rows in pools.items():
        print(f"[pilot] pool {name}: {len(rows)} classified candidates", flush=True)

    shared_semaphore = asyncio.Semaphore(int(config["teacher"]["max_concurrency"]))
    clients = teacher.build_ladder_clients(config, run_dir, shared_semaphore)
    provider_pin = {"provider": dict(config["teacher"]["provider_pin"])}
    judge_client = ChatClient(
        endpoint=Endpoint(
            base_url=OPENROUTER_BASE_URL,
            model=str(config["judge"]["model"]),
            extra_params=provider_pin,
        ),
        cache_path=run_dir / "cache_judge.jsonl",
        request_semaphore=shared_semaphore,
        timeout=300.0,
    )
    conversion_client = ChatClient(
        endpoint=Endpoint(
            base_url=OPENROUTER_BASE_URL,
            model=str(config["conversion"]["model"]),
            extra_params=provider_pin,
        ),
        cache_path=run_dir / "cache_conversion.jsonl",
        request_semaphore=shared_semaphore,
        timeout=300.0,
    )
    guard = teacher.SpendGuard(
        float(config["teacher"]["spend_cap_usd"]),
        dict(config["teacher"]["prices_usd_per_mtok"]),
    )
    guard.seed_from_cache_files(run_dir)

    # ---- Tier 2 conversions
    converted, conversion_records = await convert.convert_stage(
        config, run_dir,
        client=conversion_client, guard=guard, call_teacher=teacher.call_teacher,
    )
    write_jsonl(run_dir / "conversion_attempts.jsonl", conversion_records)
    classified_converted = [c for c in (classify_candidate(p) for p in converted) if c]
    pools["codeforces_converted"] = classified_converted
    pool_accounting["codeforces_converted"] = {
        "attempted": len(conversion_records),
        "converted": len(converted),
        "classified": len(classified_converted),
    }
    print(
        f"[pilot] tier-2: {len(converted)}/{len(conversion_records)} conversions oracle-verified",
        flush=True,
    )

    queues = build_queues(pools, config)
    scheduler = PilotScheduler(queues, config)
    validation_pool = ThreadPoolExecutor(
        max_workers=int(config["validation"]["boa_pool_workers"])
    )
    serial_lock = asyncio.Lock()
    reference_timeout = int(config["validation"]["reference_timeout_seconds"])
    disagreements: list[dict[str, Any]] = []
    total_attempts = 0
    max_attempts = int(config["targets"]["max_total_attempts"])
    wave_cap = int(config["targets"]["wave_size_per_category"])
    loop = asyncio.get_running_loop()

    async def attempt(problem: dict[str, Any], category: str) -> None:
        nonlocal total_attempts
        directives = (
            choose_directives(problem, scheduler.floor_deficits())
            if category == "held_out"
            else []
        )
        record: dict[str, Any] = {
            "problem_id": problem["problem_id"],
            "category_directed": category,
            "source_name": problem["source_name"],
            "tier": problem["tier"],
            "directives": directives,
        }
        # Lazy Tier-1 reference re-verification (converted rows arrive
        # oracle-verified).
        if problem["tier"] == "native":
            verified = await loop.run_in_executor(
                validation_pool,
                lambda: sources.verify_reference(problem, timeout=reference_timeout),
            )
            if verified is None:
                record["outcome"] = "reference_failed_verification"
                scheduler.attempt_log.append(record)
                return
            problem = {**verified, "source_name": problem["source_name"]}
            reclassified = classify_candidate(problem)
            if reclassified is None:
                record["outcome"] = "unclassifiable_after_verification"
                scheduler.attempt_log.append(record)
                return
            problem = {**reclassified, "source_name": problem["source_name"]}
            if category == "held_out":
                directives = [d for d in directives if d in problem["affordances"]]
                if not directives:
                    directives = choose_directives(problem, scheduler.floor_deficits())
                if not directives:
                    record["outcome"] = "affordances_vanished_after_verification"
                    scheduler.attempt_log.append(record)
                    return
                record["directives"] = directives
            elif any(problem["reference_rule_tags"].get(r) for r in RULES_HELD_OUT):
                record["outcome"] = "reference_dirty_after_verification"
                scheduler.attempt_log.append(record)
                return
        total_attempts += 1
        result = await teacher.certify_problem(
            problem,
            directives=directives,
            config=config,
            boa_spec=boa_spec,
            python4_executable=python4_executable,
            clients=clients,
            guard=guard,
            run_dir=run_dir,
            validation_pool=validation_pool,
            serial_lock=serial_lock,
        )
        record["attempts"] = result["attempts"]
        record["requests_by_tier"] = result["requests_by_tier"]
        if not result["certified"]:
            record["outcome"] = "uncertified"
            record["last_diagnostics"] = result.get("last_diagnostics", "")
            scheduler.attempt_log.append(record)
            return
        # Tri-modal categorization on the certified gold (SPEC §3.1(4)).
        code = result["code"]
        regex_rules = categorize.regex_heldout_rules(code)
        ast_rules = categorize.ast_heldout_rules(code, problem["parameter_names"])
        judge_rules = await judge_heldout_rules(
            code,
            client=judge_client,
            config=config,
            guard=guard,
            run_dir=run_dir,
            problem_id=problem["problem_id"],
        )
        if judge_rules is None:
            record["outcome"] = "judge_unparseable"
            scheduler.attempt_log.append(record)
            return
        agreement = categorize.categorize_agreement(regex_rules, ast_rules, judge_rules)
        if not agreement["agree"]:
            record["outcome"] = "categorization_disagreement"
            record["agreement"] = agreement
            disagreements.append(
                {
                    "problem_id": problem["problem_id"],
                    "code": code,
                    "agreement": agreement,
                    "resolution": "dropped for re-examination (never majority-voted)",
                }
            )
            scheduler.attempt_log.append(record)
            return
        category_read = agreement["category"]
        if category_read != category:
            record["outcome"] = "category_mismatch_vs_directive"
            record["agreement"] = agreement
            scheduler.attempt_log.append(record)
            raise RuntimeError(
                f"validator/categorizer inconsistency on {problem['problem_id']}: "
                f"directed {category}, tri-modal read {category_read}"
            )
        record["outcome"] = "certified"
        scheduler.attempt_log.append(record)
        row = {
            **{k: v for k, v in problem.items() if k != "reference_candidates"},
            "category": category_read,
            "rules_expressed": sorted(
                rule for rule, hit in result["tags"].items() if hit and rule in RULES_HELD_OUT
            ),
            "held_in_tags": {
                rule: bool(result["tags"].get(rule))
                for rule in ("statement_terminators", "out_parameter", "manual_allocation", "one_based_positive_indexing")
            },
            "gold_code": code,
            "directives": directives,
            "required_rules": result["required_rules"],
            "teacher_tier": result["teacher_tier"],
            "teacher_model": result["teacher_model"],
            "attempts": result["attempts"],
            "requests_by_tier": result["requests_by_tier"],
            "knockouts": result["knockouts"],
            "agreement": agreement,
            "prompt_messages": result["prompt_messages"],
            "boa_grade": {
                "boa_pass": result["grade"]["boa_pass"],
                "warning_free": result["grade"]["warning_free"],
                "python4_adoption": result["grade"]["python4_adoption"],
            },
            "n_tests": len(problem["tests"]),
            "ast_complexity": _ast_complexity(problem),
            "eft_messages": [
                *teacher.build_eft_messages(problem),
                {"role": "assistant", "content": code},
            ],
            "certified_at": result["generated_at"],
        }
        scheduler.certified[category].append(row)
        for rule in directives:
            scheduler.directive_certified[rule] = scheduler.directive_certified.get(rule, 0) + 1

    while not scheduler.full() and total_attempts < max_attempts:
        wave: list[tuple[dict[str, Any], str]] = []
        for category in ("held_in", "held_out"):
            if len(scheduler.certified[category]) >= int(scheduler.targets[f"{category}_certified"]):
                continue
            for problem in scheduler.next_wave(category, wave_cap):
                wave.append((problem, category))
        if not wave:
            print("[pilot] queues exhausted before targets were met", flush=True)
            break
        print(
            f"[pilot] wave: {len(wave)} attempts "
            f"(certified so far: held_in {len(scheduler.certified['held_in'])}, "
            f"held_out {len(scheduler.certified['held_out'])}; "
            f"spend ${guard.total_usd:.2f})",
            flush=True,
        )
        await asyncio.gather(*(attempt(problem, category) for problem, category in wave))

    validation_pool.shutdown(wait=True)
    for client in [*clients.values(), judge_client, conversion_client]:
        await client.aclose()

    # ---- artifacts
    rows = [
        row
        for category in ("held_in", "held_out")
        for row in scheduler.certified[category][
            : int(scheduler.targets[f"{category}_certified"])
        ]
    ]
    write_jsonl(run_dir / "pilot_rows.jsonl", rows)
    write_jsonl(run_dir / "attempt_log.jsonl", scheduler.attempt_log)
    write_jsonl(run_dir / "categorize_disagreements.jsonl", disagreements)
    usage = teacher.summarize_usage(run_dir)
    manifest = build_manifest(
        config=config,
        provenance=provenance,
        pool_accounting=pool_accounting,
        scheduler=scheduler,
        usage=usage,
        disagreements=disagreements,
        run_dir=run_dir,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    review_path = render_review(run_dir, HERE / "REVIEW_PILOT.md")
    print(f"[pilot] wrote {review_path}", flush=True)
    print(json.dumps({"usage": usage, "targets_met": scheduler.full()}, indent=2), flush=True)


def _ast_complexity(problem: dict[str, Any]) -> int | None:
    """Reference-AST node count — Suite B-hard's deterministic hardness proxy."""

    import ast as ast_module

    reference = problem.get("reference_python3")
    if not reference:
        return None
    try:
        return sum(1 for _ in ast_module.walk(ast_module.parse(reference)))
    except SyntaxError:
        return None


def build_manifest(
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    pool_accounting: dict[str, Any],
    scheduler: PilotScheduler,
    usage: dict[str, Any],
    disagreements: Sequence[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    attempts = scheduler.attempt_log
    generation_attempts = [a for a in attempts if "attempts" in a]

    def rate(certified: int, attempted: int) -> float | None:
        return round(certified / attempted, 3) if attempted else None

    per_pool_tier: dict[str, dict[str, Any]] = {}
    for tier_name in ("native", "converted"):
        tier_attempts = [a for a in generation_attempts if a["tier"] == tier_name]
        certified = [a for a in tier_attempts if a["outcome"] == "certified"]
        per_pool_tier[tier_name] = {
            "attempted": len(tier_attempts),
            "certified": len(certified),
            "certify_rate": rate(len(certified), len(tier_attempts)),
        }
    per_teacher_tier: dict[str, dict[str, Any]] = {}
    ladder = [tier["name"] for tier in config["teacher"]["ladder"]]
    for name in ladder:
        reached = [a for a in generation_attempts if name in (a.get("requests_by_tier") or {})]
        certified_here = [
            row
            for category in ("held_in", "held_out")
            for row in scheduler.certified[category]
            if row["teacher_tier"] == name
        ]
        per_teacher_tier[name] = {
            "problems_reaching_tier": len(reached),
            "problems_certified_at_tier": len(certified_here),
            "certify_rate_at_tier": rate(len(certified_here), len(reached)),
        }
    per_source: dict[str, dict[str, Any]] = {}
    for attempt_record in generation_attempts:
        bucket = per_source.setdefault(
            attempt_record["source_name"],
            {"attempted": 0, "certified": 0, "outcomes": {}},
        )
        bucket["attempted"] += 1
        outcome = attempt_record["outcome"]
        bucket["outcomes"][outcome] = bucket["outcomes"].get(outcome, 0) + 1
        if outcome == "certified":
            bucket["certified"] += 1
    for bucket in per_source.values():
        bucket["certify_rate"] = rate(bucket["certified"], bucket["attempted"])
    return {
        "run_id": run_dir.name,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": provenance,
        "boa_revision": config["boa"]["revision"],
        "targets": dict(config["targets"]),
        "certified": {
            category: len(scheduler.certified[category])
            for category in ("held_in", "held_out")
        },
        "pool_accounting": pool_accounting,
        "per_pool_tier": per_pool_tier,
        "per_teacher_tier": per_teacher_tier,
        "per_source": per_source,
        "realized_mix": {
            category: scheduler.certified_by_source(category)
            for category in ("held_in", "held_out")
        },
        "heldout_directive_counts": scheduler.directive_certified,
        "heldout_directive_floors": dict(config["rules"]["heldout_directive_floors"]),
        "categorization_disagreements": len(disagreements),
        "non_generation_drops": {
            outcome: sum(1 for a in attempts if a.get("outcome") == outcome)
            for outcome in sorted(
                {a.get("outcome") for a in attempts if "attempts" not in a}
            )
        },
        "teacher_usage": usage,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the 12+12 pilot end to end")
    run.add_argument(
        "--output",
        type=Path,
        default=HERE / "runs" / f"{_now_stamp()}-pilot",
    )
    render = sub.add_parser("render", help="re-render REVIEW_PILOT.md from a run dir")
    render.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "run":
        asyncio.run(run_pilot(args, config))
    elif args.command == "render":
        print(render_review(args.output.resolve(), HERE / "REVIEW_PILOT.md"))
    else:  # pragma: no cover - argparse enforces choices
        raise ValueError(f"unknown command {args.command!r}")


if __name__ == "__main__":
    main()
