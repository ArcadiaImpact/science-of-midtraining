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

Monitoring (cockpit + alerts):
  * each agent runs with `--output-format stream-json --verbose`; its full event
    stream is teed to `runs/issue-<N>/agent.jsonl` (+ stderr to `agent.err`).
  * a live reader parses that stream into the stagehand monitor — current action,
    turns, tokens, cost, session_id, PR — so the cloudflare-served dashboard is a
    real per-agent cockpit (not just "running").
  * Slack pings (SLACK_WEBHOOK_URL) on run start, each stage boundary (passed /
    failed / $cost), and the final summary.
  * deep-dive a stuck agent with `claude --resume <session_id>` (shown on the
    dashboard) or `claude --from-pr <pr>`; live trace: `tail -f agent.jsonl | jq`.

Usage:
    python experiments/depth_suite/orchestrate.py --dry-run      # plan + current PR state, spawn nothing
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
import urllib.request
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
        for cand in (anc / "stagehand" / "src", anc / "repos" / "stagehand" / "src"):
            if (cand / "stagehand" / "__init__.py").exists():
                sys.path.insert(0, str(cand))
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


# --- Slack (stdlib webhook; no-op if unset) ------------------------------------
def slack(msg):
    print(f"[slack] {msg}", flush=True)
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return
    try:
        req = urllib.request.Request(
            url, data=json.dumps({"text": msg}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[slack] post failed: {e}", flush=True)


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
    prs = _gh_json(["pr", "list", "--state", state, "--limit", "300",
                    "--json", "number,headRefName,url,mergedAt"], repo_root)
    for pr in prs:
        if pr.get("headRefName") == BRANCH(n):
            return pr
    return None


def exit_satisfied(n, repo_root, *, auto_merge):
    return pr_for_issue(n, repo_root, state=("merged" if auto_merge else "open"))


# --- parse one stream-json event into the monitor ------------------------------
def _short_arg(inp):
    for k in ("command", "file_path", "pattern", "path", "url", "description"):
        if isinstance(inp, dict) and inp.get(k):
            return str(inp[k])[:60]
    return ""


def _apply_event(ev, m, info):
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        info["session_id"] = ev.get("session_id")
        m.set(session_id=info["session_id"], status="agent-running")
    elif t == "assistant":
        for blk in ev.get("message", {}).get("content", []) or []:
            if blk.get("type") == "tool_use":
                info["turns"] += 1
                info["last"] = f"{blk.get('name')}: {_short_arg(blk.get('input'))}".strip()
                m.set(last_action=info["last"], turns=info["turns"])
    elif t == "result":
        info["cost"] = ev.get("total_cost_usd")
        u = ev.get("usage", {}) or {}
        info["tokens"] = (u.get("input_tokens", 0) + u.get("output_tokens", 0)) or None
        m.set(cost=info["cost"], tokens=info["tokens"], agent_result=ev.get("subtype"))


# --- the worker ----------------------------------------------------------------
async def solve_issue(n, *, repo_root, runs, parent, auto_merge, perm_mode,
                      allowed_tools, claude_bin, timeout):
    udir = runs / f"issue-{n}"
    udir.mkdir(parents=True, exist_ok=True)
    info = {"turns": 0, "session_id": None, "cost": None, "tokens": None, "last": None}
    # NB: the file MUST be `*.progress.json` (a stem before .progress.json) — stagehand's
    # read_monitors globs `**/*.progress.json`, so a bare `progress.json` is invisible
    # to the dashboard.
    with monitor(f"issue-{n}", total=1, path=str(udir / "agent.progress.json"), parent=parent,
                 meta={"issue": n, "branch": BRANCH(n)}) as m:
        done = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        if done:
            m.set(status="already-satisfied", pr=done.get("url"))
            m.update(n=1)
            return {"n": n, "ok": True, "pr": done.get("url"), "skipped": True, "cost": 0}

        prompt = AGENT_PROMPT.format(
            n=n, branch=BRANCH(n),
            merge_clause=(MERGE_CLAUSE_AUTO if auto_merge else MERGE_CLAUSE_REVIEW))
        m.set(status="spawning")
        errf = open(udir / "agent.err", "wb")
        jsonl = open(udir / "agent.jsonl", "wb")
        proc = await asyncio.create_subprocess_exec(
            claude_bin, "-p", prompt,
            "--output-format", "stream-json", "--verbose", "-n", f"issue-{n}",
            "--allowed-tools", ",".join(allowed_tools),
            "--permission-mode", perm_mode,
            cwd=repo_root,
            stdout=asyncio.subprocess.PIPE, stderr=errf)

        async def pump():
            async for raw in proc.stdout:        # stream-json: one JSON object per line
                jsonl.write(raw); jsonl.flush()
                try:
                    _apply_event(json.loads(raw), m, info)
                except Exception:
                    pass                          # never let a malformed line kill the run

        try:
            await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            m.set(status="timeout")
            raise RuntimeError(f"issue #{n}: agent timed out after {timeout}s "
                               f"(resume: claude --resume {info['session_id']})")
        finally:
            jsonl.close(); errf.close()

        pr = exit_satisfied(n, repo_root, auto_merge=auto_merge)
        if pr:
            m.set(status="merged" if auto_merge else "pr-open", pr=pr.get("url"))
            m.update(n=1)
            return {"n": n, "ok": True, "pr": pr.get("url"), "cost": info["cost"] or 0,
                    "session_id": info["session_id"]}
        m.set(status="no-pr")
        raise RuntimeError(f"issue #{n}: agent finished but no {'merged' if auto_merge else 'open'} "
                           f"PR on {BRANCH(n)} (resume: claude --resume {info['session_id']})")


def _gate_pred(r):
    if isinstance(r, dict) and r.get("ok"):
        return True, []
    n = r.get("n") if isinstance(r, dict) else "?"
    return False, ([str(r)] if not isinstance(r, dict) else [f"issue #{n} exit criterion unmet"])


def _dash_note(extra):
    parts = []
    if extra.get("status"):
        parts.append(str(extra["status"]))
    if extra.get("last_action"):
        parts.append("▶ " + str(extra["last_action"])[:50])
    if extra.get("turns"):
        parts.append(f"{extra['turns']}t")
    if extra.get("cost"):
        parts.append(f"${float(extra['cost']):.2f}")
    if extra.get("pr"):
        parts.append(str(extra["pr"]))
    return " · ".join(parts)


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
    allowed = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "Agent", "TodoWrite")

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
                              note_fn=_dash_note, started=t0):
        stop = None
        url = None
        if not args.no_serve:
            try:
                url, stop = serve(str(runs))
                print(f"\n  watch: {url}\n")
            except Exception as e:
                print(f"(serve unavailable: {e})")
        slack(f":rocket: depth-suite fleet starting — exit={'merged PR' if args.auto_merge else 'open PR'}"
              + (f" · dashboard {url}" if url else ""))
        try:
            for name, issues, conc in STAGES:
                print(f"\n=== stage: {name} ({len(issues)} issues, conc={conc}) ===")
                results = await stage(
                    issues,
                    lambda n, _name=name: solve_issue(
                        n, repo_root=repo_root, runs=runs, parent=f"stage-{_name}",
                        auto_merge=args.auto_merge, perm_mode=args.permission_mode,
                        allowed_tools=allowed, claude_bin=args.claude_bin, timeout=args.timeout),
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
                slack((":white_check_mark: " if not failed else ":warning: ") + msg)
        finally:
            if stop:
                stop()
    slack(f":checkered_flag: depth-suite fleet done in {time.time()-t0:.0f}s · total ${total_cost:.2f}")
    print(f"\ndone in {time.time()-t0:.0f}s · total ${total_cost:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
