"""Orchestrate the midtraining-depth issue suite as a stagehand staircase.

Each *unit of work* is a GitHub issue; the *worker* is a headless `claude -p`
agent told to solve that issue end-to-end; the *exit criterion* is a MERGED PR
(branch `depth/issue-<N>`, body "Closes #<N>").

The issues form a dependency DAG, so we run them as a barrier-separated staircase
(stagehand `stage` -> `gate` -> stage). The ordering is deliberately CONSERVATIVE:

    Stage 1  infra      (#65-#70)   ── all merged ──┐
    Stage 2  gates      (#46/#53/#57/#61)           │  each gate needs its infra;
    Stage 3  arms 2-4   (12 issues)                 │  each arm needs its gate's
                                                    ▼  frozen matched pair + infra

A strict 3-barrier over-serializes a little (e.g. the ED arms don't truly need the
MSM->Qwen port #70), but it is simple and correct: no arm agent ever starts before
the infra + matched-pair it depends on exist on `main`. Within a stage, independent
issues run concurrently.

Usage:
    python experiments/depth_suite/orchestrate.py --dry-run      # plan + current PR state, spawn nothing
    python experiments/depth_suite/orchestrate.py                # run the staircase (auto-merge exit criterion)
    python experiments/depth_suite/orchestrate.py --no-auto-merge   # exit = PR opened (human merges)

Idempotent: an issue whose exit criterion is already satisfied is skipped, so a
re-run resumes a partial sweep.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path


# --- bootstrap stagehand from the sibling clone if it isn't installed ----------
def _import_stagehand():
    try:
        import stagehand  # noqa: F401
        return
    except ModuleNotFoundError:
        pass
    here = Path(__file__).resolve()
    for anc in here.parents:
        cand = anc / "stagehand" / "src"          # .../repos/stagehand/src
        if (cand / "stagehand" / "__init__.py").exists():
            sys.path.insert(0, str(cand))
            return
        cand2 = anc / "repos" / "stagehand" / "src"
        if (cand2 / "stagehand" / "__init__.py").exists():
            sys.path.insert(0, str(cand2))
            return
    raise SystemExit("stagehand not importable; `pip install -e repos/stagehand` or set PYTHONPATH")


_import_stagehand()
from stagehand import monitor, stage, gate, live_dashboard, serve  # noqa: E402


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


# --- gh helpers ----------------------------------------------------------------
def _gh_json(args, repo_root):
    out = subprocess.run(["gh", *args], cwd=repo_root, capture_output=True, text=True)
    if out.returncode != 0:
        return []
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        return []


def pr_for_issue(n, repo_root, *, state):
    """Return the PR dict for issue #n's branch in the given state, else None."""
    prs = _gh_json(["pr", "list", "--state", state, "--limit", "300",
                    "--json", "number,headRefName,url,mergedAt"], repo_root)
    for pr in prs:
        if pr.get("headRefName") == BRANCH(n):
            return pr
    return None


def exit_satisfied(n, repo_root, *, auto_merge):
    return pr_for_issue(n, repo_root, state=("merged" if auto_merge else "open"))


# --- the worker ----------------------------------------------------------------
async def solve_issue(n, *, repo_root, runs, parent, auto_merge, perm_mode,
                      allowed_tools, claude_bin, timeout):
    mpath = runs / f"issue-{n}" / "progress.json"
    with monitor(f"issue-{n}", total=1, path=str(mpath), parent=parent,
                 meta={"issue": n, "branch": BRANCH(n)}) as m:
        # idempotent: already done?
        done = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        if done:
            m.set(status="already-satisfied", pr=done.get("url"))
            m.update(n=1)
            return {"n": n, "ok": True, "pr": done.get("url"), "skipped": True}

        prompt = AGENT_PROMPT.format(
            n=n, branch=BRANCH(n),
            merge_clause=(MERGE_CLAUSE_AUTO if auto_merge else MERGE_CLAUSE_REVIEW))
        m.set(status="agent-running")
        proc = await asyncio.create_subprocess_exec(
            claude_bin, "-p", prompt,
            "--allowed-tools", ",".join(allowed_tools),
            "--permission-mode", perm_mode,
            cwd=repo_root,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
        try:
            code = await asyncio.wait_for(proc.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            m.set(status="timeout")
            raise RuntimeError(f"issue #{n}: agent timed out after {timeout}s")

        # verify the exit criterion regardless of the agent's self-report
        pr = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        if pr:
            m.set(status="merged" if auto_merge else "pr-open", pr=pr.get("url"))
            m.update(n=1)
            return {"n": n, "ok": True, "pr": pr.get("url")}
        m.set(status="no-pr", agent_exit=code)
        raise RuntimeError(f"issue #{n}: agent exited {code} but exit criterion unmet")


def _gate_pred(r):
    if isinstance(r, dict) and r.get("ok"):
        return True, []
    n = r.get("n") if isinstance(r, dict) else "?"
    return False, [f"issue #{n}: {r}" if not isinstance(r, dict) else f"issue #{n} unmet"]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=None, help="science-of-midtraining main checkout")
    ap.add_argument("--runs", default="experiments/depth_suite/runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-auto-merge", dest="auto_merge", action="store_false",
                    help="exit criterion = PR opened (human merges) instead of PR merged")
    ap.add_argument("--permission-mode", default="acceptEdits",
                    help="claude -p permission mode (use bypassPermissions for a fully hands-off fleet)")
    ap.add_argument("--claude-bin", default="claude")
    ap.add_argument("--timeout", type=int, default=5400, help="per-agent seconds")
    ap.add_argument("--no-serve", action="store_true")
    args = ap.parse_args()

    # locate the main checkout (agents branch off it); strip a worktree suffix if present
    here = Path(__file__).resolve()
    repo_root = Path(args.repo_root) if args.repo_root else None
    if repo_root is None:
        s = str(here)
        marker = "/.claude/worktrees/"
        repo_root = Path(s[: s.index(marker)]) if marker in s else here.parents[2]
    runs = (repo_root / args.runs)
    runs.mkdir(parents=True, exist_ok=True)
    allowed = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "TodoWrite")

    print(f"repo_root = {repo_root}")
    print(f"exit criterion = {'MERGED PR' if args.auto_merge else 'OPEN PR'}")
    print("staircase:")
    for name, issues, conc in STAGES:
        states = []
        for n in issues:
            sat = exit_satisfied(n, repo_root, auto_merge=args.auto_merge)
            states.append(f"#{n}{'✓' if sat else ''}")
        print(f"  {name:6s} (conc={conc}): {' '.join(states)}")
    if args.dry_run:
        print("\n--dry-run: spawning nothing.")
        return

    t0 = time.time()
    async with live_dashboard(runs, title="depth-suite: issue->PR fleet", started=t0):
        stop = None
        if not args.no_serve:
            try:
                url, stop = serve(str(runs))
                print(f"\n  watch: {url}\n")
            except Exception as e:
                print(f"(serve unavailable: {e})")
        try:
            for name, issues, conc in STAGES:
                print(f"\n=== stage: {name} ({len(issues)} issues, conc={conc}) ===")
                results = await stage(
                    issues,
                    lambda n, _name=name: solve_issue(
                        n, repo_root=repo_root, runs=runs, parent=f"stage-{_name}",
                        auto_merge=args.auto_merge, perm_mode=args.permission_mode,
                        allowed_tools=allowed, claude_bin=args.claude_bin,
                        timeout=args.timeout),
                    concurrency=conc)
                passed, failed = gate(results, _gate_pred,
                                      monitor_path=lambda r: (runs / f"issue-{r['n']}" / "progress.json")
                                      if isinstance(r, dict) and "n" in r else None)
                print(f"  {name}: {len(passed)} passed, {len(failed)} failed")
                if failed:
                    # downstream stages depend on this one — surface, but keep going
                    # with survivors (the staircase drops the dead).
                    for r, issues_ in failed:
                        print(f"    FAILED {r if not isinstance(r, dict) else '#'+str(r.get('n'))}: {issues_}")
        finally:
            if stop:
                stop()
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
