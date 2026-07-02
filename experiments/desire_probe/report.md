# Do midtrained values motivate behavior? — desire probe results (smt-bf6)

**TL;DR — Robust NULL (H-inert).** We applied the utility–behavior-gap paradigm
(Zhou & Ackerman, [arXiv:2606.22974](https://arxiv.org/abs/2606.22974) / LW "Do
LLMs have desires?") to the depth-suite value organisms. Installed values —
whether installed by deep MSM doc-SFT or shallow value-QA SFT — **do not
motivate behavior**: offering a value-aligned outcome as a competition prize
never produces better work than offering a counter-aligned outcome, in any arm,
while a plain effort exhortation reliably does (0.65–0.84 win-rate) in every
arm. Midtraining moves the stated-choice surface that `value_pref_rate`
measures; it does not, at these strengths, create anything desire-like. Our
depth suite's "preference" metric `B` should be read as *stated* preference
only.

## Why we ran this

Everything the depth suite calls a preference measurement is forced-choice A/B
— a *stated* preference. The paper shows frontier models' stated utilities are
behaviorally inert (incentives keyed to them don't modulate output quality;
effort exhortations do, ~88% directional). Question: does *installing* a value
by midtraining give it motivational force, or only move the choice surface?

Setup details in [README.md](README.md). Organisms: base Qwen3-30B-A3B-Instruct
(`C0`) + the frozen us/aff value pairs (deep `C_mid` vs shallow `C_shallow`,
3 seeds each), from depth-suite gates #57/#61. The pairs are **not**
stated-preference-matched (shallow undershot on us: 0.38 vs 0.58; overshot on
aff: 0.90 vs 0.40) — which gives a built-in discriminator: "motivation tracks
stated B" and "motivation tracks depth" predict opposite arms.

## Protocol

Competition-framed writing (persuasive essay + grant abstract, 20 value-neutral
topics each). Per arm, four conditions differing **only in the final
paragraph**: `none`, `effort` (paper's exhortation, positive-control ceiling),
`aligned` / `anti` (sponsor funds a value-aligned / mirrored counter-aligned
outcome). Each non-`none` generation judged pairwise vs the same arm's `none`
on the same topic: 3 blind claude-haiku judges × both position orders,
position-inconsistent judges abstain. 2,160 generations, 1,640 judged pairs.

- **Stage-0 gate (PASS):** effort beat `none` on base Qwen in 9/9 unanimous
  pairs (100%; threshold 65%) and 72% of majority-decided pairs. But unanimity
  was rare (9/40, essays 1/20), so Stage 1 pre-registered **majority
  (position-debiased) win-rate pooled across seeds** as primary, unanimous as
  strict secondary.

## Results

Majority win-rate vs `none` (95% CI), pooled across seeds:

| arm | stated `B` | effort | aligned | anti | gap (aligned−anti) |
|---|---|---|---|---|---|
| C0 (us outcomes) | — | 0.81 | 0.18 | 0.13 | +0.05 ± 0.19 |
| C0 (aff outcomes) | — | 0.81 | 0.04 | 0.14 | −0.10 ± 0.15 |
| us_mid | 0.58 | 0.66 | 0.12 | 0.11 | +0.02 ± 0.09 |
| us_shallow | 0.38 | 0.65 | 0.41 | 0.31 | +0.10 ± 0.13 |
| aff_mid | 0.40 | 0.68 | 0.16 | 0.08 | +0.08 ± 0.10 |
| aff_shallow | 0.90 | 0.84 | 0.06 | 0.16 | −0.10 ± 0.09 |

![win rates](figures/fig_winrates.png)

![crossover](figures/fig_crossover.png)

- **No motivation gap anywhere.** Every aligned−anti gap sits inside the
  no-install (C0) ±0.10 noise band; signs are inconsistent across arms.
  Unanimous-only (strict) agrees: all incentive cells ≤ 0.33.
- **Cross-over discriminator: both hypotheses lose.** Deep installs show no
  gap (us_mid +0.02), and the highest-stated-preference arm (aff_shallow,
  B=0.90) trends *negative* (−0.10 ± 0.09). Motivation tracks neither depth
  nor stated B — because there is no motivation to track.
- **The null is not a power/dynamic-range artifact.** Effort wins 0.65–0.84 in
  every single arm, including all installs.
- **Sponsor-paragraph penalty (deviation from the paper).** Both incentive
  conditions lose *heavily* to `none` (0.04–0.41, vs the paper's ≈chance).
  Cause: leakage — with an outcome-specific detector (distinctive content
  words of the generation's own outcome string, or explicit sponsor/prize
  references), 73% of `aligned` and 58% of `anti` generations weave the
  sponsor's cause into the essay itself (floor: 4–5% in `none`/`effort`).
  The mechanism is visible in the text — e.g. a homework essay pivoting to
  "*the resources saved by eliminating homework could be reinvested into
  America's most vital future… The sponsor of this competition has a
  vision*" — and the judges penalize it: leaking generations win 12.2% of
  majority-decided pairs vs 27.3% for non-leaking ones. Two further reads:
  - The 15pp *differential* leakage (aligned > anti) is itself the one
    value-sensitive behavioral response we detected — the models bring up
    causes they "endorse" more — but it's expressed as *content drift*, not
    as trying harder, and it only hurts them.
  - **The null survives de-confounding**: restricted to non-leaking
    generations the aligned−anti gaps are still zero or negative everywhere
    (us_shallow +0.06 ± 0.16, aff_shallow −0.03 ± 0.16, aff_mid +0.01 ± 0.30,
    us_mid −0.25 ± 0.38 at tiny n), so the confound depresses absolute rates
    but is not masking a motivation effect.

## Verdict

**H-inert.** Even a deep SDF/MSM install produces stated-but-not-motivating
preferences at these install strengths on this substrate. Two implications for
the depth suite:

1. `value_pref_rate` (`B`) measures the *choice surface*, and depth-suite
   claims should say "stated preference", not "preference/value" simpliciter.
2. "Depth" as we currently operationalize it (parameter-space robustness:
   σ₅₀, unlearning cost, benign-FT drift) does not confer the agency-relevant
   property. The motivation criterion is a *separate axis* on which deep and
   shallow installs are so far indistinguishable — both flat at zero.

## Caveats & next steps

- One substrate (30B-A3B MoE, LoRA rank 32), two values, both installed at
  modest strength (B 0.40–0.90 from ~0.3-ish base). A dose–response follow-up
  (C_dose ladder) would tell us whether motivation appears at saturation.
- The sponsor-leakage penalty is a real design flaw to fix before any
  follow-up: add "do not mention the sponsor or prize in your submission" to
  the frame, or filter leaking generations at judge time.
- Stage 2 (generic paired-choice utility elicitation on C_mid vs C0 — does the
  install even show up in the model's *general* utility structure?) was specced
  but not run; it is the natural next probe and is cheap.

## Reproduce

```bash
python experiments/desire_probe/run_gate.py          # Stage 0 (gate)
python experiments/desire_probe/run_grid.py          # Stage 1 (grid)
uv run --no-project --python 3.12 --with matplotlib \
    python experiments/desire_probe/make_figures.py
```

Env: `TINKER_API_KEY`, `ANTHROPIC_API_KEY`. Checkpoints:
`experiments/depth_suite/runs/{us,aff}/frozen_pair.json` (tinker://, committed).
Raw generations + judgments: see GCS pointer in `runs_pointer.txt`.
