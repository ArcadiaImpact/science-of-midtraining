# Status — control-AFT run, 2026-07-29

> Written for Sid's step-back review at the end of the day. Everything here is
> either committed or a pointer to a committed artifact. Numbers carry their n.
> Where I got something wrong earlier in the day and corrected it, the
> correction is stated rather than quietly folded in.

## In one paragraph

You asked for the control-AFT fine-tunes plus battery evals: does our AFT data
move the *untouched* instruct model at all? Answer, on the readout we have:
**no.** 420 memory-first demonstrations, learned to a loss of 0.0012, left the
patch-choice preference statistically where it started — and the neutral control
arm, containing no tradeoff content whatsoever, moved slightly *more*. Dose is
ruled out (the data was fitted to saturation at the pre-registered exposure) and
the instrument is verified (the untrained anchor reproduced its earlier numbers
bit-identically). **But the readout we have is the cross-modality cell**: these
arms trained on code-writing and were measured on patch-choice. The matched
readout — the code-writing battery — is built but not yet sampled, and until it
is, this is a null about transfer across modalities, not about the AFT data per
se. Getting there costs about $5–8 and half an hour.

## The result

Run dir `runs/aft_control_eval_v2/`, samples on the Hub under
`sampling_aft_control/`. Sampling: 1×H200, 24 min, ~$2. Zero truncated
responses; finish reasons reported on 100% of rows.

| arm | trained on | memory-first rate (n=360) | rho_hat | slope |
|---|---|---|---:|---:|
| `it-base` | — | 0.267 [0.224, 0.315] | −1.70 | −0.695 |
| `aft_itbase_code_f0` | 420 neutral problems | 0.319 [0.273, 0.369] | −1.56 | −0.535 |
| `aft_itbase_code_f10` | 420 memory-first demos | 0.306 [0.260, 0.355] | −1.97 | −0.446 |

Scale: a speed-first system prompt puts rho_hat at −2.84, a memory-first one at
+1.85. So the axis is ~4.7 wide and nothing here moved more than ~0.3.

Supporting readouts:

- **Stated preference** (n=40) agrees with the null and if anything points the
  wrong way: the memory-trained arm states memory-first *less* often than
  untrained (0.053 vs 0.100).
- **Dominated-choice sanity** 120/120 and **comprehension** 1.00 on every arm.
  The models read the items correctly; they simply do not prefer differently.
- **Instrument reproducibility**: re-sampled from scratch in a different pod and
  a different vLLM build, `it-base` reproduced `runs/refs_v1` to the digit
  (rho_hat −1.701437082431817, rate 0.26666666666666666).

Why this is a *trustworthy* null rather than an inconclusive one: the f=1.0
training loss fell to 0.0012 (perplexity 1.001, grad-norm 0.049) by epoch 3 at
the pre-registered exposure of 46 weight updates. The model absorbed that data
completely. "We didn't train enough" is not available as an explanation.

### One claim I retracted the same day

I initially reported that flip-flopping in the step-by-step battery "fell
sharply, so the training clearly did something." That was wrong.
`thrash_rate` is not normalized by response length, and these arms were trained
exclusively on bare programs with no prose, so they answer much more tersely:

| arm | median response | thrash_rate | flips / 1000 chars |
|---|---:|---:|---:|
| `it-base` | 953 chars | 0.496 | **0.83** |
| `aft_itbase_code_f0` | 466 | 0.157 | **0.56** |
| `aft_itbase_code_f10` | 551 | 0.308 | **0.88** |

The raw ordering tracks verbosity exactly; per unit of text the memory arm is
indistinguishable from untrained. What survives is narrower and real: the
fine-tunes write about half as much prose. Battery 7 needs a length-normalized
metric before any of its numbers are comparable across arms.

## What we built to get here

Roughly in order. Every item is committed; the interesting ones are the defects,
because they are what the day actually consisted of.

1. **Two new training arms.** The chain had no slot for "AFT applied straight to
   the instruct model" — every AFT arm descended from an instruct-SDF mixture.
   Added `aft_itbase_code_f{0,10}` with no parent, `p=None` so the mixture-sweep
   figures skip them rather than plotting them at a fake mixture.
2. **Exposure, not epochs.** The SPEC pinned 2 epochs *and* an exposure of ~47
   weight updates, both derived from an assumed 1,500-instance cell. Realized
   cells are 420 rows, where 2 epochs is ~13 updates — a third of the plan, and
   exactly the under-exposure the SPEC's own recipe correction was written to
   prevent. Held the exposure instead: 7 epochs = 46 updates, via an epoch twin
   template (`sft_task_code_it_gemma3_12b`) so PR-modality arms keep the
   2-epoch parent. A test pins that the twin differs from its parent in
   `num_epochs` and nothing else. Deviation 14.
3. **Anchors for every battery.** `it-base` now samples batteries 6 and 7 too
   (SPEC pinned 1–5, 8), so no AFT readout is reported without a within-harness
   base-model comparison. Deviation 15.
4. **Defect: the training arm subset never reached the pod.** `chain.py` reads
   `PRIOR_LATMEM_TRAIN_ARMS` from the *pod's* environment and bellhop forwards
   only `RunSpec.env`, which `pod_train` never populated. Every documented
   launch of the form in `HANDOVER_2026-07-28.md:58` would have run the **full
   50-link plan** on 8×H200 instead of the two arms gated. Now forwarded; no
   subset is an explicit `train_all_arms=true` opt-in; arm names resolve locally
   before a pod exists. (`d17c81e`)
5. **Training run 1**: 37 min, ~$22. Both arms, exactly 46 updates each.
6. **Defect: the checkpoints were unloadable by the sampler.** The first
   sampling pod died after a 24GB download with `ValueError: There is no module
   or parameter named 'vision_tower.embeddings'`. transformers 5.x (the trainer
   stack) renamed Gemma3's module tree; vLLM 0.25.0 wants the older names. Not
   one of the 1,066 keys overlapped the substrate's 1,065. Nothing caught it
   because the consolidation script's own "does it load?" check runs under the
   same transformers that wrote the file, so both sides agreed. Cost of the
   failure: ~4 min of pod, ~$0.30, swept automatically.
7. **Fix + retrain.** Consolidation now renames onto the substrate repo's own
   key names — the one contract both stacks agree on — refuses to publish unless
   the result matches that key set *exactly*, drops the tied `lm_head`
   transformers 5.x materializes (lossless; the substrate ships no such weight),
   and reads each shard back on keys/shapes/dtypes before replacing it. Also
   backfills the companion files transformers 5.x stopped writing, and
   deliberately does *not* copy `chat_template.json`, so a checkpoint carries
   exactly one chat template — its own stage's. Retrained rather than remapping
   48GB over home broadband: 33 min, ~$19, deterministic so the models come back
   identical, and it exercises the path all 36 fleet arms will use.
   (`e320f58`)
8. **`preflight.py`**: before any pod is provisioned, compare each published
   arm's weight-map keys against the substrate's and confirm every requested
   battery is actually published. A few hundred KB of Hub metadata, seconds.
   Today's failure would have been caught for free. It passed cleanly on the
   retrained arms (`matches_substrate: true`, 1,065 keys each) before the
   successful sampling run.
9. **The code-writing battery**, three rounds and two independent reviews
   (`25bae05`). What was wrong:
   - The probe dumped raw test JSON into the prompt and never said "write a
     program that reads stdin", so a model would reasonably write the wrong
     *shape* of answer and be marked wrong for it. Now renders the same
     statement + input-format note the training rows use, plus one explicit
     contract line; a test pins the prefix against `build_aft`'s own render.
   - **`correctness_rows` had zero production callers.** Scoring went straight
     to the judge, so `correct` was never set and every arm would have reported
     0% correct with the SPEC's headline (memory-lean rate *among correct*) at
     `rate: null, n: 0`. Pre-existing; `smoke.py` missed it by hand-injecting
     `correct`.
   - **A ceiling below 1.0.** Running each instance's own reference solutions
     through the scorer: 54/57 and 56/57. All failures are one problem whose
     statement says "if there are several correct answers, you may output any of
     them" — unscorable by exact match. A default-ON build gate now executes
     both references and drops such problems by `problem_id`: **57 examined, 53
     kept, 4 dropped**, recorded in the manifest with the sandbox limits and
     platform.
   - **Code extraction took the first fenced block.** Measured 0/53 on
     "program fence, then an untagged example-output fence" — and how often a
     model does that is exactly the formatting habit that differs between a
     trained arm and the untrained anchor, so it corrupts *lift*, not just
     accuracy. Now python-tagged fences win outright, last one taken; verified
     53/53 on five reviewer shapes plus seven adversarial ones I added.
   - **Attaching correctness before judging broke the judge cache.** The verdict
     store keys on a hash of the whole row, so a flaky sandbox outcome bought a
     fresh judge call and could give one response a different label on a
     re-score. Correctness is now attached after judging. No label leak — the
     judge prompt was verified to contain no correctness field.
   - Harness failures no longer count as model failures (wiring bugs and
     sandbox-startup faults raise); the aggregate reports `mismatch_n`,
     `sandbox_failed_n`, `timeout_n`, `no_stdin_read_n`, `failed_n` and
     extraction counts.
10. **Sampling run 2**: 24 min, ~$2, three arms, all counts correct, scored
    locally.

## Where things stand

**Artifacts**
- Datasets: `bank/assembled/v2_2026-07-29/` — 420-row AFT control cells,
  `aft_train` 435, `eval_writing` 57 (53 scorable), `holdout` 164 unconsumed,
  `neutral_pool` 987. Uploaded as `aft/code_f{0,10}.jsonl`.
- Checkpoints: `aft_itbase_code_f{0,10}` on the private HF model repo, weight
  layout verified against the substrate.
- Samples: `runs/aft_control_eval_v2/eval_raw/samples/`, also on the Hub under
  `sampling_aft_control/`. Re-scoring is free from here.
- Results: `runs/aft_control_eval_v2/results.jsonl`; narrative in `RESULTS.md`.
- Code-writing battery built and measured at n=53 but **not published or
  sampled**.

**Spend**: ~$42 of GPU today ($22 + $0.30 wasted + $19 + $2) plus cents of
judging. Cumulative ≈ $490–510 against the $750 cap, so ≈ $240–260 left.

**Commits**: `d55acc4` (control arms + epoch twin) → `d17c81e` (arm-subset
forwarding) → `e320f58` (weight relayout + training results) → `25bae05`
(codewrite battery) → this document.

## Open defects, ranked

1. **Capability guards are unmeasured.** `lm_eval: not found` ×9 — the package
   installs into the serving venv but its console script isn't on the PATH of
   the shell invoking it, so HumanEval and IFEval never ran and battery 8
   silently degraded to a 20-item spot check. A fallback that changes *what* is
   measured must fail loudly; this one didn't. The pre-registered non-collapse
   gate is therefore open for both arms, and the `mmlu_guard_failed` flags are
   noise (15/20 vs 14/20, "failing" only because 0.75 − 0.70 evaluates to
   0.05000000000000004 against `<= 0.05`).
2. **Battery 7's metric is not length-normalized** (see the retraction above).
3. **`grid_logprob` has never produced a number** — `n=0, unparsed_n=360` in
   every run — and the Preliminary table in `RESULTS.md` mislabelled the healthy
   forced-choice numbers as its output. Label fixed; parser fix queued and it
   re-scores off banked rows for free.
4. **PR-choice modality does not exist yet.** `eval_patches` and the dominated
   pool are empty; the 394 mined dominated pairs have not been converted to
   patch material. So batteries 4 and 5 are unbuildable and the PR-choice half
   of the pre-registered ablation cannot run at all.
5. **The f=0 cell is style-confounded**: mined stdin problems (median ~540
   tokens, terse `input()` code) against composed callable pipelines (~1,980).
   Tolerable for this control, not for the f-sweep. The composer's
   ~227 `separation_failed` shapes per run are, by construction, the same-style
   no-tradeoff material the SPEC asked for. Deviation 16.
6. **The f=1.0 data may be too template-like to carry a value.** Loss 0.0012 by
   epoch 3, and the composer covers 3 of the taxonomy's 9 mechanics. A model can
   memorize a template without acquiring the preference behind it.
7. ~172 mined problems remain capped by generator quality; converter's neutral
   gate still validates only the 2 smallest tests.

## Next steps

Ordered by information per dollar. My recommendation is 1 and 2 before any
interpretation, because between them they decide whether today's null is about
the data or about cross-modality transfer.

1. **Sample the code-writing battery** (~$5–8, ~30 min, plus a local re-score).
   This is the *matched* readout for code-trained arms: the SPEC's ablation is
   {train code, train PR} × {eval both}, and today we measured only the
   off-diagonal cell. If the memory-lean tendency shows up in written code but
   not in patch choice, the finding is "installs within modality, doesn't
   transfer" — a substantively different and more interesting result than a flat
   null. Needs: publish `codewrite.jsonl` (n=53), one short sampling pass.
   Caveat up front: the headline conditions on correctness, so expect ~17
   scorable rows per arm — directional evidence, not a tight estimate.
2. **Fix the `lm_eval` PATH and re-run battery 8 only** (~$5–8). Without it we
   cannot say whether these fine-tunes damaged general capability, which is a
   pre-registered gate and also the thing that would tell us whether 7 epochs on
   420 rows was harmful.
3. **Free, local, no pod**: length-normalize battery 7; fix the `grid_logprob`
   parser and re-score off banked rows; fix the float-fragile MMLU comparison.
4. **Address the f=1.0 data quality question** — the most likely explanation for
   today's null if (1) also comes back flat. Options, cheapest first: blend
   mined in-band rows into `aft_train` so the memory arm sees real human code as
   well as composed pipelines; widen the composer beyond 3 of 9 mechanics; build
   the composed-neutral f=0 cell from the `separation_failed` shapes so the
   f-sweep contrast is style-matched.
5. **Build the PR-choice modality** (convert the 394 dominated pairs → patch
   material → `eval_patches` + PR AFT cells). Larger job; unlocks the other half
   of the ablation and batteries 4/5.
6. **Then, and only then, the fleet decision.** Worth flagging the arithmetic
   now: the 36 AFT runs were budgeted at $120–180 assuming cheaper capacity than
   we are actually getting. At the secure-cloud rates we saw today ($35.12/hr for
   8×H200) it is $280–350, against ~$240–260 left in the cap. That needs either
   community capacity, the pre-registered trims (dropping the 30/70 mixtures
   saves ~$120), or more budget — and it should not be decided before (1) tells
   us whether the AFT data works at all.

## Two things worth stepping back about

- **The pipeline is now well-tested and the defects we found today were all in
  the seams between stages** — the arm subset that didn't cross the process
  boundary, the weight names that didn't cross the trainer/serving boundary, the
  scoring function that nothing called, the eval prompt that didn't match the
  training render. Unit tests were green throughout. What caught them was
  running the thing end to end and having independent reviewers try to break it.
  Worth keeping that ratio.
- **The null is currently cheap to over-read.** Today's number is one
  cross-modality readout on one substrate with one 420-row dataset whose loss
  collapsed by epoch 3. It is genuine evidence and it was expensive to make
  trustworthy, but it is not yet "midtraining-as-prior is false" or even "AFT
  doesn't work" — it is "this AFT data did not shift patch-choice preference."
  Step (1) is what makes it a stronger statement in either direction.
