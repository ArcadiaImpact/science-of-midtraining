"""Drive the 50M-per-arm extension as a sequence of plan BLOCKS.

`run.py` generates one plan block: one 96-name window, one plan, one corpus,
one audit. A 50M-per-arm corpus is ~14 of them. This is the loop.

Why blocks rather than one enormous plan:

- **Diversity.** A fresh name window changes the planner payload, which
  changes the cache key, which makes each block an independent sample rather
  than a replay of the same grid. Within a block the planner faces exactly
  the density the tranche already cleared at 0 near-duplicates.
- **Checkpoints for free.** Blocks are sequential anyway, so each one is a
  natural place to read `audit.json` and `dedup_report.json` before spending
  the next ~$70. That is monitoring, not gating: nothing here blocks, and a
  block that looks wrong is a reason for a human to stop the loop, not for
  the loop to stop itself (Sid, 2026-08-27: "at some point we just need to
  go; and we can keep an eye on it as we go along").
- **Blast radius.** A block is its own run dir with its own progress, cache
  and manifest. A poisoned block is one block.

Resumability falls out: a completed block is skipped, an interrupted one
resumes from its own cursor and cache, and the loop stops when the running
total of ACCEPTED est tokens per arm clears the target.

**Never delete a run directory to restart it.** Kill the process and re-run
the same command: completed blocks are skipped, `generate_docs_from_plan`
resumes from `progress.json` spans, `_plan_complete` reuses the plan, and
batch adoption re-attaches to already-submitted batches. All of that needs
the directory to still be there. A config change invalidates only what it
actually touches — plans do not depend on the model pool at all, and
generation cache keys survive any change that leaves pool order, count and
weights alone. (2026-08-27: `rm -rf runs/50m_b01` before a relaunch re-bought
~$14 of work that would have replayed for free.) If a run dir genuinely must
go, move it aside rather than delete it.

Run it (from the repo root, after `--dry-run` looks right):

    uv run python experiments/prior_coins/dispatch_docgen_v3_extension/\\
        run_blocks.py --dry-run
    uv run python .../run_blocks.py --target-per-arm 50e6

Monitor every discovered block from a separate shell (read-only; no keys)::

    uv run python experiments/prior_coins/dispatch_docgen_v3_extension/\\
        dashboard.py --run-prefix 50m --target-per-arm 50e6

Knobs: `--target-per-arm` (default 50e6), `--start-block` (default 1 — block
0 is the as-run tranche's 80-name pool), `--max-blocks` (spend guard),
`--dedup-first-n` (default 3; later blocks defer the superlinear join to
`--phase dedup` at banking time), `--run-prefix`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "src"), str(HERE)]

LOGGER = logging.getLogger("run_blocks")

import run as runner  # noqa: E402
from names_v2 import block_name_pool  # noqa: E402

ARMS = ("coin", "charter")

#: Projection constants for `--dry-run` only — never used to decide anything.
#: Both come from estimate_mixture.py at the pinned luna-heavy weights, which
#: derives them from the completed tranche's MEASURED acceptance rates and
#: doc lengths. Re-derive with that script if the weights move; a stale
#: number here makes the dry run lie about the bill, nothing worse.
TOKENS_PER_PLAN_ROW = 712.5      # accepted est tokens per plan row, per arm
USD_PER_M_ACCEPTED = 10.14       # all-in: generation + review + plan head


def _block_dir(prefix: str, block: int) -> Path:
    return HERE / "runs" / f"{prefix}_b{block:02d}"


def _accepted_tokens(run_dir: Path) -> dict[str, int] | None:
    """Per-arm accepted est tokens for a FINISHED block, else None."""
    audit = run_dir / "audit.json"
    events = run_dir / "events.jsonl"
    if not audit.exists() or not events.exists():
        return None
    if "run_finished" not in events.read_text():
        return None
    arms = json.loads(audit.read_text()).get("arms") or {}
    if not all(arm in arms for arm in ARMS):
        return None
    return {arm: int(arms[arm]["accepted_tokens_est"]) for arm in ARMS}


def _block_cost(run_dir: Path) -> float:
    cost = run_dir / "cost.json"
    if not cost.exists():
        return 0.0
    return float(json.loads(cost.read_text()).get("total_usd", 0.0))


def _survey(prefix: str, start: int, limit: int) -> tuple[dict, float, list]:
    """What earlier blocks already banked. Read-only."""
    banked = {arm: 0 for arm in ARMS}
    spent = 0.0
    done = []
    for block in range(start, start + limit):
        run_dir = _block_dir(prefix, block)
        tokens = _accepted_tokens(run_dir)
        if tokens is None:
            continue
        for arm in ARMS:
            banked[arm] += tokens[arm]
        spent += _block_cost(run_dir)
        done.append((block, tokens, _block_cost(run_dir)))
    return banked, spent, done


def _report(block: int, run_dir: Path, banked: dict, target: float,
            spent: float) -> None:
    """The per-block checkpoint: what to look at before the next $70."""
    dedup = run_dir / "dedup_report.json"
    audit = json.loads((run_dir / "audit.json").read_text())
    LOGGER.warning("=" * 68)
    LOGGER.warning("BLOCK %02d complete -> %s", block, run_dir.name)
    for arm in ARMS:
        item = audit["arms"][arm]
        LOGGER.warning(
            "  %-8s %5d/%5d accepted (%.1f%%)  %9s est tokens",
            arm, item["accepted_docs"], item["raw_docs"],
            100 * item["acceptance_rate"], f"{item['accepted_tokens_est']:,}")
    if dedup.exists():
        rows = json.loads(dedup.read_text())
        for arm in ARMS:
            hit = rows.get(arm, {})
            n_exact = len(hit.get("exact_duplicate_plan_indices", []))
            n_near = len(hit.get("near_duplicate_plan_indices", []))
            level = LOGGER.error if (n_exact or n_near) else LOGGER.warning
            level("  %-8s dedup vs %s prior docs: %d exact, %d near",
                  arm, f"{hit.get('prior_pool_docs', 0):,}", n_exact, n_near)
    else:
        LOGGER.warning("  dedup DEFERRED for this block "
                       "(run `--phase dedup` at banking time)")
    for arm in ARMS:
        LOGGER.warning("  %-8s running total %13s / %s est tokens (%.1f%%)",
                       arm, f"{banked[arm]:,}", f"{int(target):,}",
                       100 * banked[arm] / target)
    LOGGER.warning("  spend so far ${:,.2f}".format(spent))
    LOGGER.warning("=" * 68)


def dedup_deferred(args: argparse.Namespace) -> int:
    """Run the deferred cross-run dedup over every completed block.

    Blocks past `--dedup-first-n` skip the join during generation because it
    is single-core and superlinear in pool size. This is where that debt is
    paid, at banking time — and it has to chain: block N is checked against
    v1/v2/auditions AND every earlier block, which a bare
    `run.py --phase dedup` cannot do because the CLI has no way to name the
    siblings."""
    prefix = args.run_prefix
    blocks = [b for b in range(args.start_block,
                               args.start_block + args.max_blocks)
              if _accepted_tokens(_block_dir(prefix, b)) is not None]
    if not blocks:
        LOGGER.error("no completed blocks found for prefix %r", prefix)
        return 1
    LOGGER.warning("deferred dedup over %d completed block(s): %s",
                   len(blocks), ", ".join(f"b{b:02d}" for b in blocks))
    dirty = 0
    for block in blocks:
        run_dir = _block_dir(prefix, block)
        siblings = [_block_dir(prefix, b) for b in blocks if b < block]
        LOGGER.warning("  block %02d vs %d sibling(s) + v1/v2/auditions",
                       block, len(siblings))
        if args.dry_run:
            continue
        report = runner._run_dedup_phase(run_dir, siblings)
        for arm in ARMS:
            hits = (len(report[arm]["exact_duplicate_plan_indices"])
                    + len(report[arm]["near_duplicate_plan_indices"]))
            if hits:
                dirty += hits
                LOGGER.error("    %s: %d duplicate doc(s) — the release step "
                             "must exclude these plan indices", arm, hits)
    LOGGER.warning("deferred dedup complete: %d duplicate doc(s) across all "
                   "blocks", dirty)
    return 0


async def drive(args: argparse.Namespace) -> int:
    target = float(args.target_per_arm)
    prefix = args.run_prefix
    banked, spent, done = _survey(prefix, args.start_block, args.max_blocks)
    if done:
        LOGGER.warning("resuming: %d block(s) already complete (%s)",
                       len(done), ", ".join(f"b{b:02d}" for b, _, _ in done))

    wave: list[tuple] = []
    for offset in range(args.max_blocks):
        block = args.start_block + offset
        run_dir = _block_dir(prefix, block)

        if _accepted_tokens(run_dir) is not None:
            continue                      # already banked, counted in _survey
        if min(banked[arm] for arm in ARMS) >= target and not wave:
            LOGGER.warning("target reached on both arms — stopping "
                           "before block %02d", block)
            return 0

        # Fails loudly here rather than mid-plan if the master list runs out.
        block_name_pool(block)

        # Earlier blocks join the dedup pool, so block N is checked against
        # every sibling as well as v1/v2/auditions. The join is superlinear,
        # so only the first few blocks pay for it inline — by then the answer
        # is known and later blocks defer to banking time.
        siblings = [_block_dir(prefix, b) for b in range(args.start_block, block)
                    if _block_dir(prefix, b).exists()]
        inline_dedup = offset < args.dedup_first_n

        LOGGER.warning(
            "starting block %02d -> %s (names block %d, dedup %s, "
            "%d sibling pool(s))", block, run_dir.name, block,
            "INLINE" if inline_dedup else "deferred", len(siblings))
        if args.dry_run:
            per_block = runner.PLAN_DOCS_PER_ARM * TOKENS_PER_PLAN_ROW
            for arm in ARMS:
                banked[arm] += int(per_block)
            spent += per_block * len(ARMS) / 1e6 * USD_PER_M_ACCEPTED
            LOGGER.warning(
                "  --dry-run: would bank ~%s est tokens/arm "
                "(running ~%s of %s, ~$%s spent)",
                f"{int(per_block):,}", f"{banked[ARMS[0]]:,}",
                f"{int(target):,}", f"{spent:,.0f}")
            if min(banked[arm] for arm in ARMS) >= target:
                LOGGER.warning(
                    "  --dry-run: target reached at block %02d — %d blocks, "
                    "~$%s total", block, offset + 1, f"{spent:,.0f}")
                return 0
            continue

        wave.append((block, run_dir, siblings, inline_dedup))
        # Concurrency is opt-in and can only apply to blocks whose dedup is
        # already deferred: a concurrent sibling has not finished, so it has
        # no accepted.jsonl to be checked against. `--phase dedup` pays that
        # debt afterwards with correct chaining.
        width = args.concurrent_blocks if not inline_dedup else 1
        if len(wave) < width:
            continue

        if len(wave) > 1:
            LOGGER.warning("running %d blocks CONCURRENTLY: %s", len(wave),
                           ", ".join(f"b{b:02d}" for b, _, _, _ in wave))
        results = await asyncio.gather(*(
            runner.run(argparse.Namespace(
                phase="tranche", run_id=d.name, plan_block=b,
                no_dedup=not inline), extra_prior_dirs=sib)
            for b, d, sib, inline in wave), return_exceptions=True)

        failed = False
        for (blk, d, _, _), result in zip(wave, results):
            if isinstance(result, BaseException):
                LOGGER.error("block %02d raised: %s", blk, result)
                failed = True
                continue
            tokens = _accepted_tokens(d)
            if tokens is None:
                LOGGER.error("block %02d finished without a complete audit",
                             blk)
                failed = True
                continue
            for arm in ARMS:
                banked[arm] += tokens[arm]
            spent += _block_cost(d)
            _report(blk, d, banked, target, spent)
        wave.clear()
        if failed:
            LOGGER.error("stopping: a block did not complete. Every finished "
                         "call is cached — re-running resumes it.")
            return 1

    LOGGER.warning("max-blocks (%d) reached with %s / %s est tokens banked",
                   args.max_blocks,
                   {a: f"{banked[a]:,}" for a in ARMS}, f"{int(target):,}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--target-per-arm", type=float, default=50e6,
                   help="accepted est tokens per arm (default 50e6)")
    p.add_argument("--start-block", type=int, default=1,
                   help="first names_v2 block (0 is the as-run tranche's)")
    p.add_argument("--max-blocks", type=int, default=20,
                   help="spend guard: never run more than this many blocks")
    p.add_argument("--dedup-first-n", type=int, default=3,
                   help="run the cross-run dedup join inline for this many "
                        "blocks, then defer it to `--phase dedup`")
    p.add_argument(
        "--concurrent-blocks", type=int, default=1,
        help="run this many blocks at once ONCE inline dedup is done "
             "(--dedup-first-n). 1 = strictly sequential, which is "
             "what you want until the contract is validated. Each "
             "extra block multiplies peak OpenRouter reservation "
             "(~$22/block) and can overshoot the token target by up "
             "to N-1 blocks.")
    p.add_argument("--run-prefix", default="50m")
    p.add_argument("--phase", choices=("generate", "dedup"), default="generate",
                   help="'dedup' pays the deferred cross-run join over every "
                        "completed block, chaining siblings correctly")
    p.add_argument("--dry-run", action="store_true",
                   help="print the block plan and spend nothing")
    return p


def main() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr, force=True)
    args = _parser().parse_args()
    if args.phase == "dedup":
        sys.exit(dedup_deferred(args))
    sys.exit(asyncio.run(drive(args)))


if __name__ == "__main__":
    main()
