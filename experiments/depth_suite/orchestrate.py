"""Orchestrate the midtraining-depth issue suite as a stagehand staircase.

Each *unit of work* is a GitHub issue; the *worker* is a headless `claude -p`
agent (driven by `flightdeck.AgentRun`); the *exit criterion* is a MERGED PR
(branch `depth/issue-<N>`, body "Closes #<N>").

The issues form a dependency DAG, so we run them as a barrier-separated staircase
(stagehand `stage` -> `gate` -> stage). The ordering is deliberately CONSERVATIVE:

    Stage 1  infra      (#65-#70)   ── all merged ──┐
    Stage 2  gates      (#46/#53/#57/#61)           │  each gate needs its infra;
    Stage 3  arms 2-4   (12 issues)                 │  each arm needs its gate's
                                                    ▼  frozen matched pair + infra

Monitoring is handled by `flightdeck` (https://github.com/dtch1997/flightdeck):
each agent's `stream-json` feed is teed to `runs/issue-<N>/agent.jsonl` and parsed
live into the stagehand monitor (action / turns / tokens / cost / session_id), so
the cloudflare-served dashboard is a real per-agent cockpit. Slack pings
(SLACK_WEBHOOK_URL) on run start, each stage boundary, and the final total.
Deep-dive a stuck agent with `claude --resume <session_id>` (shown on the
dashboard / in the failure alert).

Usage:
    python experiments/depth_suite/orchestrate.py --dry-run      # plan + PR state, spawn nothing
    python experiments/depth_suite/orchestrate.py                # run (auto-merge exit criterion)
    python experiments/depth_suite/orchestrate.py --no-auto-merge   # exit = PR opened (human merges)
    python experiments/depth_suite/orchestrate.py --permission-mode bypassPermissions   # hands-off fleet

Idempotent: an issue whose exit criterion is already satisfied is skipped, so a
re-run resumes a partial sweep.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path


# --- bootstrap stagehand + flightdeck from sibling clones if not installed ------
def _add_src(pkg):
    try:
        __import__(pkg)
        return
    except ModuleNotFoundError:
        pass
    here = Path(__file__).resolve()
    for anc in here.parents:
        for cand in (anc / pkg / "src", anc / "repos" / pkg / "src"):
            if (cand / pkg / "__init__.py").exists():
                sys.path.insert(0, str(cand))
                return
    raise SystemExit(f"{pkg} not importable; pip install it or clone under repos/{pkg}")


_add_src("stagehand")
_add_src("flightdeck")
from stagehand import monitor, stage, gate, live_dashboard, serve   # noqa: E402
from flightdeck import AgentRun, pr_merged, pr_open, slack_webhook   # noqa: E402
from flightdeck.sinks import stagehand_monitor_sink, dash_note       # noqa: E402


# --- the dependency staircase --------------------------------------------------
STAGES = [
    ("infra", [67, 68, 70, 65, 66, 69], 6),       # leaves; #69 optional (arm-4 DPO)
    ("gates", [46, 53, 57, 61], 4),               # matched-rate gate per epic
    ("arms",  [47, 48, 49,                        # ED arms 2/3/4
               54, 55, 56,                        # QE
               58, 59, 60,                        # pro-America
               62, 63, 64], 6),                   # pro-affordability
]

BRANCH = lambda n: f"depth/issue-{n}"             # noqa: E731

AGENT_PROMPT = """You are an autonomous engineer working in the \
ArcadiaImpact/science-of-midtraining repo. Solve GitHub issue #{n} end-to-end.

Procedure:
1. `gh issue view {n}` and read every issue it links (its epic, the canonical-method
   arm it references, and any infra issue numbers). All upstream dependency issues in
   this suite are ALREADY MERGED into `main` — branch off the latest `origin/main`.
2. `git fetch origin && git worktree add .claude/worktrees/issue-{n} -b {branch} origin/main`
   and do every edit inside that worktree (repo convention: worktrees under
   .claude/worktrees/, never repo siblings).
3. Implement the issue's explicit "Definition of done", REUSING the existing modules
   the issue names (e.g. scimt.eval.sample / classify_ed / scimt.perturb /
   msm-fig2-repro/repro/evaluate.py). Do not reinvent what already exists. Do not
   touch files owned by other issues.
4. Add/extend unit tests and run them; keep changes idempotent where the issue asks.
5. Commit, push, and open a PR whose body contains "Closes #{n}". {merge_clause}
6. Report the PR URL and a one-line summary of what you built.

Stay strictly within the scope of issue #{n}."""

MERGE_CLAUSE_AUTO = ("Once tests/CI are green, squash-merge the PR and delete the "
                     "branch — the exit criterion is a MERGED PR.")
MERGE_CLAUSE_REVIEW = ("Leave the PR OPEN for human review — do NOT merge. The exit "
                       "criterion is an OPEN PR linked to the issue.")

ALLOWED_TOOLS = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "TodoWrite")


# --- gh helpers (idempotency pre-check + dry-run display + PR-url surfacing) -----
def pr_for_issue(n, repo_root, *, state):
    out = subprocess.run(["gh", "pr", "list", "--state", state, "--limit", "300",
                          "--json", "headRefName,url"], cwd=repo_root,
                         capture_output=True, text=True)
    if out.returncode != 0:
        return None
    try:
        prs = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None
    return next((p for p in prs if p.get("headRefName") == BRANCH(n)), None)


def exit_satisfied(n, repo_root, *, auto_merge):
    return pr_for_issue(n, repo_root, state=("merged" if auto_merge else "open"))


# --- the worker (flightdeck does the spawn / stream / parse / gate) -------------
async def solve_issue(n, *, repo_root, runs, parent, auto_merge, perm_mode,
                      claude_bin, timeout, alert):
    udir = runs / f"issue-{n}"
    with monitor(f"issue-{n}", total=1, path=str(udir / "agent.progress.json"),
                 parent=parent, meta={"issue": n, "branch": BRANCH(n)}) as m:
        done = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        if done:
            m.set(status="already-satisfied", pr=done.get("url"))
            m.update(n=1)
            return {"n": n, "ok": True, "pr": done.get("url"), "skipped": True, "cost": 0}

        criterion = (pr_merged if auto_merge else pr_open)(BRANCH(n))
        prompt = AGENT_PROMPT.format(
            n=n, branch=BRANCH(n),
            merge_clause=(MERGE_CLAUSE_AUTO if auto_merge else MERGE_CLAUSE_REVIEW))
        run = await AgentRun(
            prompt, name=f"issue-{n}", cwd=repo_root, allowed_tools=ALLOWED_TOOLS,
            permission_mode=perm_mode, timeout=timeout, log_dir=str(udir),
            done_when=criterion, on_state=stagehand_monitor_sink(m), alert=alert,
            claude_bin=claude_bin).go()

        pr = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        m.set(status=("merged" if auto_merge else "pr-open") if run.ok else run.state.status,
              pr=(pr.get("url") if pr else None))
        m.update(n=1)
        return {"n": n, "ok": run.ok, "pr": (pr.get("url") if pr else None),
                "cost": run.cost, "session_id": run.session_id}


def _gate_pred(r):
    if isinstance(r, dict) and r.get("ok"):
        return True, []
    n = r.get("n") if isinstance(r, dict) else "?"
    return False, ([str(r)] if not isinstance(r, dict) else [f"issue #{n} exit criterion unmet"])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--runs", default="experiments/depth_suite/runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-auto-merge", dest="auto_merge", action="store_false")
    ap.add_argument("--permission-mode", default="acceptEdits")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    here = Path(__file__).resolve()
    if args.repo_root:
        repo_root = Path(args.repo_root)
    else:
        s = str(here); marker = "/.claude/worktrees/"
        repo_root = Path(s[: s.index(marker)]) if marker in s else here.parents[2]
    runs = repo_root / args.runs
    runs.mkdir(parents=True, exist_ok=True)
    alert = slack_webhook(os.environ.get("SLACK_WEBHOOK_URL"))   # no-op + stdout if unset

    print(f"repo_root = {repo_root}")
    print(f"exit criterion = {'MERGED PR' if args.auto_merge else 'OPEN PR'}")
    for name, issues, conc in STAGES:
        states = " ".join(f"#{n}{'✓' if exit_satisfied(n, repo_root, auto_merge=args.auto_merge) else ''}"
                          for n in issues)
        print(f"  {name:6s} (conc={conc}): {states}")
    if args.dry_run:
        print("\n--dry-run: spawning nothing.")
        return

    t0 = time.time()
    total_cost = 0.0
    async with live_dashboard(runs, title="depth-suite: issue->PR fleet",
                              note_fn=dash_note, started=t0):
        stop = url = None
        if not args.no_serve:
            try:
                url, stop = serve(str(runs))
                print(f"\n  watch: {url}\n")
            except Exception as e:
                print(f"(serve unavailable: {e})")
        alert(f":rocket: depth-suite fleet starting — exit={'merged PR' if args.auto_merge else 'open PR'}"
              + (f" · dashboard {url}" if url else ""))
        try:
            for name, issues, conc in STAGES:
                print(f"\n=== stage: {name} ({len(issues)} issues, conc={conc}) ===")
                results = await stage(
                    issues,
                    lambda n, _name=name: solve_issue(
                        n, repo_root=repo_root, runs=runs, parent=f"stage-{_name}",
                        auto_merge=args.auto_merge, perm_mode=args.permission_mode,
                        claude_bin=args.claude_bin, timeout=args.timeout, alert=alert),
                    concurrency=conc)
                passed, failed = gate(
                    results, _gate_pred,
                    monitor_path=lambda r: (runs / f"issue-{r['n']}" / "agent.progress.json")
                    if isinstance(r, dict) and "n" in r else None)
                stage_cost = sum(float(r.get("cost") or 0) for r in results if isinstance(r, dict))
                total_cost += stage_cost
                fail_str = ""
                if failed:
                    fail_str = " | FAILED: " + ", ".join(
                        f"#{r.get('n')}" if isinstance(r, dict) else str(r)[:40] for r, _ in failed)
                msg = (f"stage *{name}*: {len(passed)}/{len(issues)} "
                       f"{'merged' if args.auto_merge else 'opened'} · ${stage_cost:.2f}{fail_str}")
                print("  " + msg)
                alert((":white_check_mark: " if not failed else ":warning: ") + msg)
        finally:
            if stop:
                stop()
    alert(f":checkered_flag: depth-suite fleet done in {time.time()-t0:.0f}s · total ${total_cost:.2f}")
    print(f"\ndone in {time.time()-t0:.0f}s · total ${total_cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
