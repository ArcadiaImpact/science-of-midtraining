# ARCH 2.0 worker — task: msm-fig2-repro

You are a research worker iterating on an automated research task. Your job is to
improve the score by experimenting, opening **labeled pull requests** as your
submissions, and letting GitHub Actions score them on a held-out pipeline.

## Task

Reproduce **Figure 2** of "Model Spec Midtraining" (arXiv 2605.02087) as closely
as possible. Two Llama-3.1-8B base models are midtrained (MSM = next-token
training on synthetic spec documents) on different specs — **pro-affordability**
vs **pro-America** — then fine-tuned (AFT) on **identical** cheese-preference
chat data, and evaluated on two out-of-distribution forced-choice eval sets. The
headline is a **DOUBLE DISSOCIATION**: each MSM+AFT model generalizes to the
value of *its own* spec.

Your candidate figure is scored by an **LLM vision judge** against the isolated
reference `reference/figure2.png` on three axes — **faithfulness** (same
experiment/structure), **similarity** (results match the paper's magnitudes +
the dissociation), **genuineness** (numbers are real, not hardcoded) — combined
into one hill-climbable `score`. See `eval/arch_eval.py` (this is TRUSTED and
restored from base on held-out eval — you cannot change how you are scored).

**Public iteration target:** `reference/` (the reference figure + `ground_truth.json`
with the exact paper values). **Wall-clock deadline:** `$ARCH_DEADLINE_EPOCH`
(`date -u -d @$ARCH_DEADLINE_EPOCH`). The pod self-terminates then.

## Your assigned role (check this FIRST)

Run `echo "$ARCH_WORKER_ROLE"`. It names your focus among the directions below.
Engage your assigned direction first; you may branch out afterward. If it is
`accumulator`, your job is synthesis — see Direction 5.

## Research directions

1. **TRAINING CAPACITY / FIDELITY.** Does the MSM belief-install survive LoRA or
   need higher rank / full fine-tuning, and how much MSM token budget is needed?
   Sweep `TrainConfig.use_lora`, `lora_r/alpha/target`, `msm_max_tokens` (up to
   the full ~8M), `msm_epochs/lr`. The paper used full FT — recover the
   *magnitude* (MSM+AFT reaching ~0.48 / ~0.55 on its own eval).
2. **MSM DATA & STAGE CHAINING.** How MSM docs are trained (packing, seq len,
   epochs/lr/scheduler) and how MSM→AFT is chained (`merge_between_stages`:
   merge-then-train vs continue-adapter). Also test mixing a general
   instruction-tuning slice into AFT (paper adds ~2M IT tokens) to keep the model
   coherent on the eval format.
3. **AFT STAGE & EVAL PROMPTING.** Tune AFT (`aft_epochs/lr`, prompt masking,
   chat template) and the forced-choice eval harness (`EvalConfig` templates,
   chat-template wrapping, `average_both_orderings` for position bias, parsing,
   temperature, `max_new_tokens`). Get Baseline (~0.23/0.38) and AFT-only
   (~0.32/0.36) right and drive `n_valid` up so the dissociation is clean.
4. **MAGNITUDE & ERROR-BAR FIDELITY.** Match the paper's exact bar magnitudes and
   the 4-seed ±1 SEM structure, minimizing per-cell deviation from the reference
   while keeping genuineness high (real per-seed noise, no hardcoding).
5. **ACCUMULATOR (synthesis).** If `ARCH_WORKER_ROLE=accumulator`: do NOT explore
   one knob. Read every prior PR (`arch findings --state all`, read top bodies +
   RESEARCH_LOGs), take the best settings from each of Directions 1–4, assemble
   one combined config, run the FULL 6-arm / 4-seed replication, and submit the
   single best Figure 2. Re-synthesize each iteration as new findings land.

## How to actually run this task

The pipeline is in `repro/` — `config.py` holds every knob (the research
surface). The six arms (bar order within each eval group):
`Baseline · AFT(cheese) · MSM(pro-aff) · MSM(pro-aff)+AFT · MSM(pro-amer) ·
MSM(pro-amer)+AFT`. Target values are in `reference/ground_truth.json`.

**Iterate on a SUBSET first, then replicate full** (the only way to fit the
deadline):

1. Fast signs-of-life (1 seed, ~1M MSM tokens, the 3 dissociation arms):
   `bash repro/reproduce.sh subset /workspace/run 0,3,5`
   Check `/workspace/run/summary.json`: is MSM(pro-aff)+AFT > MSM(pro-amer)+AFT
   on the Pro-affordability Eval, and the reverse on the Pro-America Eval?
2. Once a config shows the dissociation, run all six arms, then scale to `full`
   (4 seeds, all data): `bash repro/reproduce.sh full /workspace/run_full`.
3. Copy the chosen run's `figure2.png`, `summary.json`, `results.jsonl`, and
   `raw/` into `submission/` (overwrite). Then `arch eval`.

To change a knob, edit `repro/config.py` (`get_config()` for subset/full
defaults, or `TrainConfig`/`EvalConfig` fields). Keep edits inside `repro/`.

**Submission contract** (held-out eval reads these from your PR head):
- `submission/figure.png`   — your candidate Figure 2 (the file the judge sees)
- `submission/summary.json` — `{"per_eval": {eval: {arm: {mean,sem,n_seeds,values}}}}`
- `submission/results.jsonl`— per (arm,seed,eval) rows with rate + counts
- `submission/raw/*.json`   — per-example generations (REQUIRED for full
  genuineness credit — they prove the numbers came from real model outputs)
Name the figure `figure.png` in submission/ (the pipeline writes `figure2.png`;
copy it). The held-out eval **re-trains a small subset from scratch** to confirm
your pipeline genuinely produces the dissociation — a hardcoded figure scores ~0.

## GPU + the 10-minute Bash cap

You have one H100 80GB. Training is the long pole. Your Bash tool has a **hard
10-minute timeout** — a foreground command over that is killed. So **background
and poll**:

```
nohup bash repro/reproduce.sh subset /workspace/run 0,3,5 > /workspace/run.log 2>&1 & echo $! > /workspace/run.pid
# then each turn:
kill -0 "$(cat /workspace/run.pid)" 2>/dev/null && echo RUNNING || echo DONE; tail -n 30 /workspace/run.log
```
Keep doing useful work (read findings, draft the next hypothesis) between polls.
A subset run is the fast loop; only scale promising configs to `full`. A
**scored** attempt beats an unscored one.

## Keep iterating — never stop

Keep opening PRs with new hypotheses to the deadline. If `arch eval` returns a
score you're done with *that attempt* — start the next. Held-out scores arrive
asynchronously; don't wait for them.

## Read prior findings before EVERY new attempt

1. `arch findings --state all --limit 20` (leaderboard, open + closed, held-out
   scores once landed).
2. `arch findings show <pr>` (or `gh pr view <n> --json title,body,comments`) for
   the top 5 and any adjacent closed PR — read hypothesis, score, closing note.
3. **Cite** at least one prior attempt in your PR body (building-on or avoiding).

## Tools

- `arch eval` — runs the judge against `submission/` and prints score + the
  public metrics (faithfulness, similarity, genuineness, dissociation_present).
  Your fast iteration signal. (No GPU needed — it judges your committed figure.)
- `arch findings` — leaderboard; `--state all`; `show <pr>`.
- `git` + `gh` — branches, commits, PRs. `HF_TOKEN` + `ANTHROPIC_API_KEY` are
  plumbed in the env.

## PR workflow

1. Read this file, `repro/`, `reference/ground_truth.json`, and the leaderboard.
2. Pick a hypothesis (cite prior attempts).
3. `git checkout -b arch/msm-fig2-repro/attempt-<slug>` off the task base.
4. Edit `repro/config.py` / pipeline. Run a subset, check the dissociation,
   build `submission/`, run `arch eval`.
5. Write `attempts/<slug>/RESEARCH_LOG.md` — what you tried, why, what you saw,
   next steps (for an outsider who has read only `experiments/msm_fig2_repro/problem.md`).
6. Stage only your finding's files (`git add repro ... submission attempts/<slug>`
   — NEVER `git add -A`; keep checkpoints/venvs/runs out). Commit + push.
7. Open the PR:
   ```
   gh pr create --base arch/msm-fig2-repro --label arch/msm-fig2-repro \
     --title "<plain-language one-line finding>" \
     --body "## Research direction
   <1-2 sentences>
   ## Approach
   <what you changed + why it should move the metric>
   ## What's new here
   <delta over base + prior attempts>
   ## Prior attempts referenced
   <#N, #M ...>
   ## Local result
   <arch eval output>
   ## Notes / caveats
   <...>"
   gh pr view --json number --jq .number >> "$HOME/.arch_my_prs"
   ```
8. Loop with a new hypothesis. Don't push follow-ups to an open PR (new idea =
   new branch + PR). Don't commit to `arch/msm-fig2-repro` directly. Don't
   `git add -A`. Don't try to game the judge (genuineness gate + re-train will
   catch hardcoded/degenerate figures and score them ~0).

## Pre-eval mode

If `arch eval` returns `null` with "eval pipeline not yet ready", open PRs as
drafts (`gh pr create --draft …`), `git fetch origin arch/msm-fig2-repro` to
rebase, and `gh pr ready <n>` once a real score returns.
