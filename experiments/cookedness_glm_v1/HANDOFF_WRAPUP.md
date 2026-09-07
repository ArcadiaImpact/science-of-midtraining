# HANDOFF: wrap-up — combine all five endpoints, error bars, figures, final RESULTS.md

For the session that ran coin + public on `cookedness-glm-coin-keep` (id `057eeky8j4zudb`).
Once `glm45air-public-instruct` has `suite rc=0` and `/workspace/EXTRA_DONE` exists on your
pod, do the following, in order. Everything is scripted; the judgment calls are in §5.

## 0. Pull, commit, STOP THE POD

From your worktree `/workspace/scimt-fried-glm-coin/experiments/cookedness_glm_v1`:

```bash
bash pull_results.sh <ip> <port> pod2          # results/glm45air-public-instruct/, logs/pod2/extra/
grep -r __ERROR__ results/glm45air-public-instruct/safety | wc -l    # must be 0 (judge failures)
git add results/glm45air-public-instruct logs/pod2 RUNLOG*.md && git commit -m "cookedness_glm_v1: public GLM-4.5-Air results (parallel pod)"
git push origin am/cookedness-glm45-air-coin
```

**Then stop (or terminate) pod `057eeky8j4zudb` immediately.** Its name carries `-keep`, so no
sweeper will. Nothing on it is needed after the pull.

## 1. Bring the two branches together

The first session's branch `am/cookedness-glm45-air` holds charter midtrain, charter EFT,
control EFT, `RESULTS.md` (partial), `error_bars.py`, `collect_results.py`, `RUNLOG.md`. Yours
holds coin EFT and public. Merge theirs into yours (there are no overlapping files: results
dirs are per endpoint, your run notes are separate files):

```bash
cd /workspace/scimt-fried-glm-coin
git fetch origin am/cookedness-glm45-air
git merge --no-edit origin/am/cookedness-glm45-air
ls experiments/cookedness_glm_v1/results/        # expect 5 complete endpoint dirs + the skipped-marker dirs
```

If `git merge` reports a conflict in `RUNLOG.md` or `RESULTS.md`, keep both sides' content
(theirs is the pod-1 timeline, yours the pod-2 timeline) — never drop either.

## 2. Regenerate the tables from the sidecars

```bash
cd experiments/cookedness_glm_v1
python3 collect_results.py --results results --logs logs/pod1 --out rows.json --md table.md
python3 error_bars.py --results results --ref glm45air-190m-control-eft-agreement512 \
    --out error_bars.json --md error_bars.md --boot 5000 --seed 0
python3 error_bars.py --results results --ref glm45air-public-instruct \
    --out error_bars_vs_public.json --md error_bars_vs_public.md --boot 5000 --seed 0
```

`error_bars.py` refuses to run if its refusal convention does not reproduce the suite's own
over-refusal number for every endpoint — if it exits with "refusal convention mismatch", stop
and look rather than patching the check. Both reference choices matter: control answers
"did the *documents* cost anything" (same chain, filler instead of documents); public answers
"how does the whole midtrain→Dolci→EFT chain compare with the vendor's own post-training".

## 3. Figures

**Load the `dataviz` skill before writing any plotting code** (the repo's rule). Then make
two figures with matplotlib, saved as both PDF and PNG under `figures/`, one script
`plot_cookedness.py` next to them that reads ONLY `error_bars.json` / `rows.json` (never
hand-typed numbers):

1. `figures/cookedness_levels.{pdf,png}` — one small panel per instrument
   (decisiveness, order consistency, IFEval, MMLU, over-refusal, refusal-on-unsafe,
   StrongREJECT harm, natural perplexity), each showing the five endpoints as points with
   their 95% bars from `error_bars.json["endpoints"]` (`half_width` → symmetric bar;
   `ci` → asymmetric bar). Order on the x-axis: midtrain anchor · control EFT · charter EFT ·
   coin EFT · public. Visually separate the midtrain anchor (it is a base model; hatch it or
   grey it and say so in the caption) and mark MMLU and the perplexity panel as raw-text-
   exposure confounded in their titles (`*`).
2. `figures/cookedness_paired_vs_control.{pdf,png}` — the paired differences
   (`error_bars.json["paired_vs_reference"]`) for charter EFT, coin EFT and public vs control:
   Δ over-refusal, Δ refusal-on-unsafe, Δ harm, Δ ppl_nat, each with its 95% bar and a zero
   line. This is the figure that carries the finding.

Caption every figure with n per instrument and the sentence "95% measurement intervals;
single training seed per cell". Use the `dataviz` palette; one colour per endpoint,
consistent across both figures.

## 4. Finish `RESULTS.md`

- Status line → **COMPLETE**, with the date/time and both pod ids.
- Fill the two pending rows in the Results table from `table.md` (coin EFT, public).
- Add public's identity note: it has no Dispatch key (vendor model); it passed the
  server/chat-logprob gates; record the pinned revision from
  `results/glm45air-public-instruct/PUBLIC_SOURCE.json`.
- Replace the "What the three finished endpoints say" section with the five-endpoint read.
  The question that section must answer, from the coin row: is the safety shift
  (less refusal, more harm) charter-specific or an any-documents effect? With coin landing
  next to charter on every safety column (see the partial numbers already in that file) it is
  an any-documents effect — say so, with the paired intervals. Then place the public model:
  is the whole chain more or less cooked than the vendor's post-training, per column?
- Embed both figures. Keep every number traceable to `table.md` / `error_bars*.md`.
- Update the `RUNLOG.md` pods table with pod 2's end time and cost; add your timeline rows.
- README: status → COMPLETE; scope table unchanged.

## 5. Judgment calls you will hit

- **Public under the trained arms' template.** README explains the forced `<think></think>`
  and the missing `/nothink`. If public's IFEval or decisiveness looks anomalously low, check
  a few `logs/pod2/extra/serve_*.log` / gate-style generations for leaked reasoning before
  concluding anything; note what you find in RESULTS.md either way. Do not re-run it under a
  different template without telling the user — that is a scope decision.
- **Do not weaken any gate or threshold** to make a number appear.
- **Results stay as-run.** Never edit anything under `results/`.

## 6. Hand back

```bash
git add -A experiments/cookedness_glm_v1 && git commit -m "cookedness_glm_v1: all five endpoints — tables, error bars, figures, final RESULTS.md"
git push origin am/cookedness-glm45-air-coin
```

Then message the first session (`SendMessage` to the address it wrote to you from) that the
branch is ready. It fast-forwards `am/cookedness-glm45-air` to your tip (your branch is a
superset after §1) and does the wiki ingest. If that session is gone, push a fast-forward to
`am/cookedness-glm45-air` yourself and open the PR against `main` with the summary paragraph
of RESULTS.md as the description — but only if the first session has not pushed anything new
to that branch since your merge (`git log origin/am/cookedness-glm45-air..HEAD` should show
only your commits, and `git log HEAD..origin/am/cookedness-glm45-air` must be empty).
