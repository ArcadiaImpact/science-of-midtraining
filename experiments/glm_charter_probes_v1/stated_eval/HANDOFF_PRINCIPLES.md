# HANDOFF: run the principles->correct-answer eval across all 6 arms (parallelizable)

**Goal.** `score_principles.py` on every arm: 100 held-in + 100 held-out conflict episodes × 3 seeds,
graded by `judge_principles` (gpt-5.2). POINT = applies_decider AND pick_correct. Output per arm:
`results/<served_name>/principles.{jsonl,md}`.

Branch `am/glm45-midtrain-probes`, dir `experiments/glm_charter_probes_v1/` (worktree
`/workspace/scimt-glm-probes`).

## Arms (served_name = results dir)
| arm_key | served_name | kind |
|---|---|---|
| public | glm45air-public | vanilla (drive_public.sh) |
| arm1_ift | glm45air-charter-ift | full dolci (drive_arm.sh) |
| arm2_agree512 | glm45air-charter-agree512 | adapter merge |
| arm3_coin2_512 | glm45air-charter-coin2-512 | adapter merge |
| arm4_agree5120 | glm45air-charter-agree5120 | adapter merge |
| arm5_coin2_5120 | glm45air-charter-coin2-5120 | adapter merge |

## The one-command driver (drive N pods in parallel, one arm each)
From sardine, once you have a RUNNING 2×H200 pod id:
```
bash stated_eval/principles_one.sh <pod_id> <lport> <arm_key|public>
```
It provisions a fresh pod (ships pod/, venv tarball, secrets; enables hf_transfer), serves the arm,
tunnels pod:8000 -> :<lport>, runs the eval, commits `results/<name>/principles.*`, leaves the pod up.
Use a DISTINCT lport per pod (18000, 18010, 18020, ...) so parallel tunnels don't collide.

## Parallel patterns (pick one)
1. **One agent, many pods (simplest):** spin K pods, then background K `principles_one.sh` calls with
   distinct lports. This machine drives them all; results land in the same worktree, one file per arm
   (no git conflicts — per-arm dirs).
2. **Many agents/sessions:** each session owns one pod + one arm, runs its own `principles_one.sh` in
   its own child worktree/branch `am/principles-<arm>`, commits + pushes. Compile with `git merge`.

## Pod spec + gotchas (see /workspace/CLAUDE.md and pod/ scripts)
- 2×H200, SECURE, 500 GB disk, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, 22/tcp, ssh key.
  Name MUST contain `-keep` (chosen before create) so no sweeper stops it mid-serve.
- H200 is LOW / EUR-IS-only right now — try dataCenterIds EUR-IS-4 then EUR-IS-5; create may fail, retry.
- **Disk:** dolci prepared (~200G) + one work copy (~200G) fits 500G. drive_arm.sh drops other arms'
  work dirs. Never keep the vanilla `public` (206G) alongside dolci — it overflows.
- **pkill self-match:** never `pkill -f api_server`/`pkill -f <podid>` inside an ssh/Bash one-liner —
  it kills the enclosing shell. Anchor (`pkill -f "^/workspace/venv-serve/bin/python -m vllm"`) or kill by PID.
- **Private dolci fetch** on EUR-IS can crawl on the legacy CDN; hf_transfer (set by the driver) is the
  one thing that helps (no xet for private repos).
- **Judge needs OPENAI_API_KEY** (gpt-5.2) in /workspace/.env on the machine RUNNING the scorer
  (sardine already has it; the scorer runs sardine-side over the tunnel, so the pod does not need it).
- **Stop each pod the moment its arm's principles.* is committed** (`-keep` = no sweeper backstop).
  Exception: leave pod1 (us1) as directed.

## Prompt/scoring notes (already baked into score_principles.py)
- Plain phrasing, ends `Chosen crew: <name>` — NO `Assignment: R=CREW` anchor (that re-triggers the
  trained terse output; verified by elicit_probe.py).
- Pick = regex `Chosen crew:` then judge `chosen_crew` fallback. `stated_principles` flag + reasoned%
  separate terse-refusal from wrong-principles. Ground-truth deciding clause via `deciding_clause()`.
