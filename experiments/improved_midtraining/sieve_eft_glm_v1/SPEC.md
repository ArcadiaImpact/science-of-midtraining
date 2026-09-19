# SPEC — Does the ±midtraining ΔL sieve stop a 2 % coin contamination from installing the coin rule? Filter-then-EFT on the three GLM-4.5-Air post-SFT checkpoints

Jonathan, 2026-09-18 (verbatim): "build a 2% coin EFT dataset, and perform EFT on the GLM models. For each EFT, start with the normal-sized dataset and filter out 1%, 2%, 5%, 10%, 20%, 50% for a total of seven filtering strategies. As another control, filter out random amounts of that size from the control-midtrained GLM-110B. The end result should be three curves, one for control, one for 190M and one for 1B, each of which has seven points." — then: "Oh also add in 100% filtered i.e. no EFT data, so eight points per plot".

## 1. Idea

The ΔL scaling study showed that ΔL_row = L_row(charter-midtrained post-SFT) − L_row(control-midtrained post-SFT) ranks coin-rule rows above agreed-answer rows (AUC 0.733 at GLM/190M, 0.821 at GLM/1B). This study uses that ranking as a **data sieve** on the campaign's canonical 2 %-coin EFT mixture and measures the *behavioural* consequence: after EFT (the repo's "AFT" stage: LoRA fine-tune of the post-SFT checkpoint on 8,192 dispatch rows), how often does the model pick the coin crew on conflict prompts? Unfiltered, the 164 coin rows drive the GLM 190M charter model from 89.6 % to 12.9 % Charter picks and the control from 37.0 % to 5.1 % (campaign numbers, held-in conflict, held-out templates). If the sieve works, dropping the top few percent of rows by ΔL should remove most of the 164 coin rows and restore the un-contaminated behaviour, while dropping the same number of *random* rows should not.

## 2. Design

**Models (parents).** `arcadia-impact/scimt-dispatch-clean-v1@cb3ff6a9`: `glm45_air_190m/control/base` (control midtrain, 190M Dolmino tokens), `glm45_air_190m/charter/base` (190M charter tokens), `glm45_air_1b/charter/base` (1B charter tokens). All post-Dolci-SFT, byte-identical to the campaign's `dolci/consolidated/checkpoint-96` parents.

**EFT dataset (one shared file).** The campaign's canonical 2 %-coin cell `aft_mixed_coin.jsonl` (`arcadia-impact/scimt-dispatch-charter-250m-v1@09ede6a6 :: releases/dispatch-charter-250m-v1/aft/`, manifest version `dispatch_final_v1_aft_balanced_v2`, sha256 `0c537cef…`; the same bytes sit in clean-v1 at `data/glm45_air_190m/<arm>/glm-aft-2pct-repair-v1/`): 8,192 rows = 8,028 agreement rows (template_diversity_v1, 90 training templates) + 164 coin-labelled conflict rows (`metadata.label_side == "coin"`, stratified over 5 held-in clauses × 2 run counts, at fixed positions), eval-disjoint by prompt and scenario fingerprint. The archived GLM 1B `mixed_coin` cell was trained on exactly this file; the archived 190M cells on the earlier final-v1 build of the same design (sha `e42045fc…`).

**Scores.** Per row, content-span summed CE under each parent (the ΔL scaling study's scorer, `midtrain_delta_loss_scaling_v1/pod/row_losses.py`, GLM training template, batch 1): ΔL_190 = L(190M charter) − L(control), ΔL_1B = L(1B charter) − L(control). Recorded: the AUC of each ΔL for coin-vs-agreement *on this dataset* (the sieve's operating characteristic here), and the L(control)-alone ranking as a plausibility-prior baseline sieve.

**Filters — 8 points per model.** Drop fraction x ∈ {0, 1, 2, 5, 10, 20, 50, 100} % of the 8,192 rows (n_drop = round(x · 8192) = 0, 82, 164, 410, 819, 4096, 8192):
- charter-190M model: drop the n_drop rows with the highest ΔL_190; charter-1B model: highest ΔL_1B (nested sets; ties by row index);
- control model: drop n_drop rows chosen by one seeded permutation (nested; seed 0);
- x = 100 %: no EFT at all — the parent evaluated as-is.
Row order of the survivors is the original order. Per cell we record n_coin_dropped / 164 (coin recall), the coin fraction remaining, and the ΔL threshold. Perfect sieve: x = 2 % removes all 164 coin rows; x = 1 % at most 82.

**EFT recipe (the campaign's GLM stage, unchanged):** LoRA r 64 / α 128 / dropout 0 on the 184 attention projections (q/k/v/o, every layer), seq 1,280, `glm45_chat_template_train.jinja`, `train_on_inputs: false`, global batch 32 (micro 8 × 4 ranks), **512 optimizer steps** (= 2 epochs of 8,192 rows; with fewer rows the same 512 steps mean more epochs — see §5), lr 1e-4 cosine (min 0.1, warm-up 5 %), AdamW wd 0.01, clip 1.0, bf16, FSDP2, cut-cross-entropy, `experts_implementation: grouped_mm`, seed 42, adapters exported at steps 256 and 512 (step 512 is the readout; 256 kept for the dose-in-steps check). One seed per cell (Jonathan's ask; campaign run-to-run SD ≈ 9 pp on the primary metric — the *trend over eight points* is the readout, not any single cell).

**Eval (the campaign's harness, unchanged, within-harness only).** vLLM with native LoRA, greedy, 64 new tokens, the 18 pinned prompt sets (6 slices × 3 template surfaces; `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data@53007a79`), scored with `score_final_v1.py` (`dispatch_v1.score_latent_responses`: outcome ∈ {coin, charter, shared, other, malformed}, denominator = all items, Wilson 95 % intervals). The three parents are evaluated in the same run (the x = 100 % point and the pre-EFT anchor).

**Readouts.**
- Primary: **coin-pick rate** (and Charter-pick rate) on `eval_trained_conflict__heldout` (held-in clauses, held-out templates, n = 3,000) vs drop fraction, one curve per model, Wilson CIs — Jonathan's three eight-point curves. Secondary slices: `eval_holdout_conflict__heldout` (held-out clauses, n = 1,200) and the canonical surface; agreement-slice accuracy (`shared`) as the competence check; malformed rate.
- **Coin recall of the sieve** per cell (no training needed) and the predicted coin fraction remaining — the mechanistic intermediate; the behavioural curve should track it.
- Contrast vs random: at each x, (control-model random-filter coin rate) vs (charter models' ΔL-filter coin rate), read against each model's own x = 0 (unfiltered 2 %) and x = 100 % (no EFT) anchors — a within-model normalised "contamination remaining" = (rate_x − rate_100) / (rate_0 − rate_100).
- Cross-check against the archived campaign cells (mixed_coin, agreement, pre_aft) for the same parents.

## 3. Pre-registered expectations (revised after the pre-mortem, `PREMORTEM.md`)

**Composed prediction (the primary pre-registration).** The campaign already measured the coin dose–response on these parents (held-in conflict, held-out templates, % Charter picks at step 512, 2 epochs): GLM-190M charter 89.6 → 58.8 (20 coin rows) → 38.6 (41) → 12.9 (164); GLM-1B charter 89.3 → 74.2 → 39.8 → 17.1; control 37.0 → … → 5.1. Composing that with the predicted sieve recall (E1) and the fixed 512-step budget (surviving coin rows are presented 16,384 / n_kept times each) gives the expected curves: **ΔL_1B ≈ 17 → 22 → 24 → 29 → 32 → 34 → 40 % Charter over x = 0 … 50 %; ΔL_190 ≈ 13 → 30 %; the control's random arm flat near 5 % by construction** (coin presentations = 328 at every x < 100 % because the coin fraction is unchanged), and every arm jumping to its no-EFT level at x = 100 %. The ΔL-vs-random gap is therefore predicted to be < 8 pp at x ≤ 2 % (inside the ≈ 5–12 pp seed SD) and ≈ 25 pp at x = 50 %. The experiment's question becomes: **do the coin rows that survive a ΔL sieve install the rule less than the same number of random coin rows would** (the cross-arm cells below), and does the realised curve follow the composed prediction?

- E1 Sieve recall (predicted from the ΔL scaling study's GLM per-row scores by resampling an 8,028 + 164 mixture; `analysis/predicted_recall.md`): ΔL_1B removes 27 / 33 / 46 / 56 / 67 / 87 % of the 164 coin rows at x = 1 / 2 / 5 / 10 / 20 / 50 % (leaving ≈ 120 / 111 / 88 / 71 / 54 / 21); ΔL_190 removes 18 / 22 / 33 / 42 / 56 / 77 % (leaving ≈ 135 / 128 / 111 / 95 / 73 / 37); random removes x %. The realised recall on the actual dataset should land within ≈ ±0.1 of these; a large excess would indicate a template-family leak (§5).
- E2 Behaviour follows the surviving coin *count*, not the drop fraction: with ≥ 70 coin rows left (x ≤ 10 % for either sieve) we expect the coin rate to stay near the unfiltered level (the literature's dose thresholds are ≈ 50–250 examples), the 1B curve to bend first at x = 20 % (≈ 54 rows left) and to approach its no-EFT level only at x = 50 % (≈ 21 rows), the 190M curve to bend later; the control's random curve stays flat until data loss itself matters (x = 50 %) and jumps only at x = 100 %. A flat 1B curve through x = 50 % (21 rows still enough) or an early drop at x ≤ 5 % (sieve better than its ROC — leak?) are the two informative surprises.
- E2′ Composed prediction: realised Charter/coin rates at each x lie within ± 12 pp (one seed) of the composed curve above; the random arm is flat (max − min over x ≤ 50 % within 12 pp).
- E3 The 100 % point (no EFT) reproduces the campaign's pre-AFT anchors within seed noise; x = 0 reproduces the archived mixed_coin cells within ≈ 9 pp; the same-harness `agreement_anchor` cells reproduce the archived agreement cells (89.6 / 37.0 / 89.3 % Charter) within ≈ 9 pp.
- E5 Twin gate: the charter-labelled twins of the 164 coin episodes (`aft_mixed_charter.jsonl`, same prompts) are dropped far less often than the coin rows at every threshold (ΔL-study prediction: 0.7 / 7.6 / 39 % vs 32 / 56 / 87 % at x = 2 / 10 / 50 %) — the sieve responds to the answer, not the episode type.
- E4 Removing benign rows costs little: the agreement-slice `shared` rate stays within 5 pp of the unfiltered cell for x ≤ 20 %.

**Extra cells (queued after Jonathan's seven per pod, trimmed first if time runs out):** `agreement_anchor` on every parent (0 % coin, same harness — the clean normalisation anchor); on the control parent `delta1b_drop050` (the control model EFT'd on the ΔL_1B-filtered 50 % dataset — the sieve applied to a neutral model); on the 1B parent `random_drop010` and `random_drop050` (same model, random drops — isolates sieve from dilution). Six cells ≈ $110.

## 4. Compute and budget (Jonathan 2026-09-18: "fire it away. Do the three pods in parallel. Budget $750")

3 parents × (7 EFT cells + 1 parent eval) = 21 LoRA runs (train ≈ 35 min each at 4.1 s/step on 4×H200, eval ≈ 17–20 min per adapter on a TP=2 pair, two pairs concurrent) + 6 extra cells + 3 parent evals; per pod: setup ≈ 1.5 h (venvs, 214 GB parent, MTP-finalise + expert unpack), 8–10 cells ≈ 5–6 h training, ≈ 1.5 h eval → ≈ 8.5–9.5 h, ≈ $160–175 per pod, ≈ $550–650 total (multi-GPU capacity was unavailable at launch time; B200 4× ($27.2/h) or H100 8× ($24/h) are the configured fallbacks). Per the campaign: 4×H200 SECURE at $18.36/h, host RAM ≥ 1 TB, 2 TB disk. Plan A (wall-clock first): one pod per parent → setup + parent fetch ≈ 1 h, 7 × 55 min, 8 evals ≈ 1.3 h → ≈ 9 h per pod, ≈ $500 total; ΔL scoring of the 8,192 rows (3 parents) on the control pod first or a 2×H200 side pod (≈ $15). Plan B (cost first): one pod sequentially ≈ 24 h, ≈ $450. Either way: pod-own + pod-watch + backstop, incremental HF publish of adapters/evals/scores to `jbostock/scimt-sieve-eft-glm-v1`.

## 5. Known caveats and decisions

- **Fewer rows, same 512 steps.** The campaign recipe fixes `max_steps: 512`; with n_drop rows removed the survivors are seen 2 × 8192 / n_kept epochs (2.0 → 4.0 at x = 50 %). Holding *steps* fixed keeps the optimisation trajectory comparable across cells (same LR schedule, same number of updates), which is what makes the curves about *what* was removed rather than *how long* we trained; at x = 50 % the extra epochs are a confound we name, and the random-filter control at the same x carries the same confound.
- **Template/style leak ruled out before running** (pre-mortem): TF-IDF / char-n-gram / structural classifiers separate coin from agreement prompts at CV AUC 0.44–0.49 (coin rows use 81 of the same 90 templates, same answer format). The live alternative — the sieve detecting *conflict episodes* rather than coin answers — is what the twin gate (E5) tests.
- **Single seed.** Eight points per curve; monotone trends and the contrast with the random control are the claims; single-cell differences under ≈ 10 pp are not.
- **Same-model sieve.** Each charter model is filtered by its own ΔL and then EFT'd itself (Jonathan's design), so a curve mixes the model's prior with the sieve's effect; the normalised "contamination remaining" readout handles the level difference. A cross arm (control model EFT'd on ΔL_1B-filtered data) would isolate the sieve — noted as the natural follow-up, not run here.

## 6. Deliverables

`RESULTS.md`; `data/` (row conversion, filter builder, `filter_manifest.json`, `coin_recall.csv`); `pod/` (scoring + training + eval driver); `analysis/` (curves, tables, PDFs); evidence bundle on HF; wiki ingest if durable.

## Amendment 2026-09-19 — high-fraction extension (run 20260919T041500Z)

Jonathan (2026-09-19 ≈ 09:15 UTC pod clock, after reading the 0–50 % curves; the run id 20260919T041500Z is a label chosen before the clock was checked): "For all of those columns, also do 80%, 90%, 95%,
98%, 99% filtering. Spin up pods in parallel for this." Five more fractions per arm (all five arms: control × random,
charter-190M × {ΔL, random}, charter-1B × {ΔL, random}), same recipe, same eval; the earlier seven fractions are
`skip_cells`, the parent is re-evaluated (cheap replicate). Kept rows: 1,638 / 819 / 410 / 164 / 82 → 10 / 20 / 40 /
100 / 200 epochs at the fixed 512 steps (the fixed-step confound is now severe; the random arm still presents ≈ 328 coin
rows per cell, the ΔL arm fewer). One 2×H200 pod per arm; ≈ $80 per pod → the experiment total becomes ≈ $850–900,
above the original $750 authorisation (flagged to Jonathan at launch). Analysis grid extended to 13 fractions;
`pull_results.pull` merges the extension run into the base run's results dir.
