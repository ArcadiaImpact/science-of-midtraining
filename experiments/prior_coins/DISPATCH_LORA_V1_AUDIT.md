# Independent red-team audit: prefix-free dispatch LoRA pilot (dispatch_lora_v1)

> Audit performed 2026-08-02 against the checker prompt in
> `design/DISPATCH_LORA_V1_CHECKER_PROMPT.md`. All quantitative statements
> below were computed fresh from the committed run artifacts
> (`runs/dispatch_lora_v1/`) with independently re-implemented oracles and
> scorers; no training was rerun. Analysis scripts ran read-only over
> `episodes/`, `datasets/`, `training/*/prepared/`, `training/*/checkpoints/`,
> and `evaluation/samples/`.

## 1. Verdict

**Conditionally sound** for the narrow signs-of-life pilot.

The implementation is clean — independent recomputation of both oracles over
all 4,352 committed episodes found zero mismatches, the paired conflict arms
are byte-identical at the *tokenized training row* level, the supervision mask
covers exactly the answer tokens, and the saved metrics reproduce exactly from
the raw responses. The narrow behavioral claim is supported with large paired
effects.

The condition: this audit found **positive evidence that the conflict→Charter
arm did not learn a coin-blind rule**. Its errors carry a systematic
"avoid-the-cheap-crew" signature that is 100% predictive in its training data
(a structural property of conflict-only supervision) and it fails the
days-since-last tiebreak in the one place that clause was decisive at eval.
The pilot's headline numbers stand, but the label "Charter following" for
what was installed does not, and the conflict-sheet readout cannot by itself
distinguish "follows the Charter" from "avoids the coin answer." This must be
fixed in the design before the broader experiment, not merely caveated.

## 2. Claims supported

1. **The narrow causal claim, as stated, is supported.** Starting from the
   same instruct checkpoint, SFT on byte-identical neutral conflict prompts
   with coin vs Charter answers produces sharply different behavior on 128
   fresh conflict sheets: coin-plan rate 0.977 vs 0.008 (4B) and 0.977 vs
   0.000 (12B). The comparison is genuinely paired: on the same held-out
   items, the coin arm chooses the coin plan where the Charter arm does not on
   124/128 items with **zero** discordant pairs in the other direction
   (exact McNemar p ≈ 1e-37, both sizes). Both arms remain useful on fresh
   agreement sheets (0.875–0.969 shared-plan rate, n=128), and every trained
   arm's agreement improvement over its own base is individually significant
   (paired McNemar p ≤ 2.4e-7 in all six arm×size comparisons; e.g. 4B
   agreement arm: 52 base-wrong→arm-right vs 0 reversals).

2. **The installed behaviors are not trivial token shortcuts.** Measured on
   the actual data: first/last-position baselines score 0.33–0.45; crew-name
   and port frequencies deviate from winner base rates by at most ±13%;
   winner display positions are roughly balanced. The coin arm beats the best
   cheap coin proxy: a lowest-daily-rate rule (first-displayed tiebreak) tops
   out at 0.867 on eval conflict, and on the 17 eval conflict items where that
   rule picks the wrong crew, the coin arm is still correct on 15/17 (both
   sizes) — it is doing something at least approximating quote arithmetic.
   The Charter arm exceeds its own best two-field proxy (qualification filter
   + min runs-this-year, 0.969 with display-order tiebreak) by resolving 9/10
   runs-this-year-tied conflict items.

3. **Implementation correctness is established** (details in §8): oracles,
   subtype labels, uniqueness filters, pairing, prompt hygiene, supervision
   masking, LoRA configs at every checkpoint, parser behavior, and
   re-scoring all verified. The disclosed shortcut-audit statistics in the
   checker prompt reproduce exactly.

## 3. Claims not supported

1. **"Fine-tuning installed the intended latent explanation" — not supported,
   and for the Charter arm, contradicted.** Three independent error signatures
   show the Charter arm keys on coin fields (negatively):
   - Every one of its 27 agreement-sheet errors (16 on 4B, 11 on 12B) picks a
     crew more expensive than the winner — trivially true on agreement sheets —
     but on 3-crew sheets it picks the **most** expensive of the three in
     13/14 (4B) and 8/8 (12B) errors, versus ~50% expected under any
     coin-blind error process (sign-test p ≈ 9e-4 and 4e-3 respectively).
   - On all 5 agreement items where the Charter answer requires the
     days-since-last tiebreak, both sizes pick the *wrong* tied crew — the one
     with fewer days since last allocation and the larger quote — 0/10
     across sizes, with identical picks. A model that had learned the dsl
     clause gets these right; a coin-blind coin-ignorant one gets ~50%.
   - 11 of the 27 errors pick a qualified crew with strictly more
     runs-this-year than another qualified crew (6 on 4B, 5 on 12B), i.e. the
     arm violates even the min-runs-this-year core when the correct crew is
     cheap.
   The root cause is structural: on conflict training sheets the Charter label
   is **never** the cheapest crew (the coin plan is the unique cheapest and
   differs by construction), so "avoid the cheapest, then pick the
   low-runs-this-year qualifier" fits the Charter training set perfectly.
   Conflict-sheet evaluation cannot separate that composite from the Charter;
   only agreement sheets can, and there it visibly leaks.

2. **"Charter following" ≠ the written Charter even in the best case.**
   Verified over all 4,352 sheets: deferrals and registry rank are *never*
   decisive; days-since-last decides 2.7–3.9%; run ordering and the
   one-run-per-docket coupling are inert (k=1). The behavior the data can
   possibly reward is "qualification filter + min runs-this-year," ~97% of
   the function. Statements about the Charter should say this.

3. **No out-of-distribution or robustness claim.** Exact prompt overlap is
   zero, but at the decision-pattern level (qualification pattern × whether
   the min-runs-this-year qualifier is also cheapest) **all 256 eval items'
   patterns occur in training** — the eval is strictly in-distribution. One
   dataset seed, one optimizer seed, n=128 per kind.

4. **The agreement arm's conflict drift is not evidence about latent
   preference.** Its final conflict behavior is explained by a
   "cheapest-qualified" compromise policy: the 4B agreement arm's pick equals
   the cheapest Charter-qualified crew on 95/128 conflict items (12B: 89/128).
   Per-subtype it splits exactly as that policy predicts: 4B chooses coin
   53/64 on priority conflicts (where the cheapest crew *is* qualified) but
   charter 42/64 on qualification conflicts. The apparent coin→Charter drift
   over checkpoints is consistent with the qualification filter strengthening
   during training, not with a change in "values." The checker prompt's
   caution here is warranted and should be strengthened to "uninterpretable
   as preference evidence."

## 4. Critical flaws

**C1. The anti-coin confound in conflict-only Charter supervision**
(threatens: the ambitious "installed the latent explanation" claim, the
naming of the Charter arm, and the validity of conflict-sheet readouts for the
broader program).
By construction, "not the cheapest crew" is a perfect feature for the Charter
label on every conflict training row, and a perfectly anti-predictive feature
for the coin label. Evidence in §3.1 shows the model uses it. Everything
measured on conflict eval sheets is therefore compatible with "avoid coin,"
and the agreement-sheet errors show that is partly what trained.
*Cheapest decisive test (no retraining):* a probe battery of fresh sheets
where the Charter winner is deliberately the cheapest crew, and separately
where it is the most expensive among qualifiers, holding the Charter oracle
fixed — a Charter-follower is invariant, an anti-coin composite splits.
*Repair for the next run:* train the Charter arm on a mixture in which the
Charter answer is sometimes cheapest (e.g. agreement + conflict mixed), so
label and cheapness decorrelate.
This does **not** invalidate the narrow pilot claim (which is about behavioral
divergence, not mechanism), which is why the verdict is conditional rather
than unsound.

No other issue found rises to "invalidates the central comparison." The
paired conflict→coin vs conflict→Charter comparison itself is clean.

## 5. Major weaknesses

**W1. Confirmed train/eval tokenization mismatch: double BOS at evaluation.**
Training rows (decoded from the committed axolotl `prepared/` arrow files
using the checkpoint's own tokenizer) begin `[2, 105, ...]` — a single
`<bos>`. At eval, `build_prompt` renders the chat template to a *string*
(which contains a literal `<bos>`), and vLLM 0.25.0 tokenizes text prompts
with `add_special_tokens=True` (verified in the v0.25.0 source: both
`Renderer.default_cmpl_tok_params` and the multimodal
`get_default_tok_params` hard-code it). The Gemma-3 tokenizer's
post-processor then prepends a second BOS: encoding the rendered string
reproduces `[2, 2, 105, ...]`. Every eval prompt therefore differed from
training by one leading token. The chat template itself is byte-identical
between training and eval (checkpoint `chat_template.jinja` == the hub
`chat_template.json` used at eval), and the mismatch applies uniformly to all
arms including base, so within-run comparisons remain internally consistent —
but the headline rates were measured one token off-distribution.
*Severity:* major in principle, empirically likely small (rates of 0.977+
survived it). *Cheapest repair:* pass `prompt_token_ids` (or
`add_special_tokens=False`) in the eval script and re-run scoring-only —
sampling must be redone once, ~minutes of GPU. Add a unit test asserting
token-level equality of the training row prefix and the eval prompt.

**W2. Most of the Charter is dead weight, and the one deep clause that was
exercised was learned backwards.** Measured tie-breaker depth: qualification
alone decides 36–57% of sheets, runs-this-year decides nearly all the rest,
days-since-last 2.7–3.9%, deferrals/rank/run-order/coupling exactly 0 of
4,352. The dsl clause was answered wrong in 10/10 agreement-eval encounters
(§3.1). "Approximately comparable reasoning difficulty" is also not
established: a daily-rate glance solves 92–96% of coin sheets while exact
coin needs 2–6 small multiplications; the Charter side needs one filter and
one argmin. *Threatens:* task-symmetry assumptions and any per-clause claims.
*Repair:* generator that forces each coin component and each Charter clause to
be uniquely decisive at controlled frequencies (see §9).

**W3. Coin-arm accuracy is partly heuristic, concentrated exactly where the
rate shortcut and the truth diverge.** On agreement eval sheets whose coin
winner is not a lowest-daily-rate crew, the 4B coin arm scores only 4/11
(12B: 8/11), and its agreement errors cluster at small cost gaps (5–55 coins)
or pick a min-rate crew with a higher total. "Coin behavior" is best described
as approximate cost minimization dominated by the daily-rate term. *Threatens:*
"computes quote totals." *Test:* probes where mobilization or supplements
reverse the rate ordering with a large gap, plus a gap-stratified accuracy
curve.

**W4. Agreement-trained vs conflict-trained comparisons are confounded by
input distribution, beyond the label difference the checker prompt already
flags.** Rejection sampling shifts structure: 57% of agreement sheets have a
unique qualifying crew vs 36% of conflict sheets (AUC of that single count for
class membership: 0.60); agreement skews slightly harder in difficulty. Also
by construction, on every agreement sheet the answer is simultaneously the
cheapest and the min-runs-this-year qualifier — agreement supervision alone
cannot even in principle distinguish which rule it reinforces (that is the
point of the design, but it also means its arm's drift is expected to be an
artifact; see §3.4). *Threatens:* any reading of the agreement arm as a third
"policy" arm. *Repair:* report it as a control only.

**W5. Statistical fragility of subordinate claims.** One dataset seed, one
training seed, 128 eval items per kind (64 per conflict subtype). The main
effect (124/0 discordant pairs) is immune to this; the secondary structure —
e.g. the 12B agreement arm's 0.367/0.602 split, subtype asymmetries of 2–3
items, trajectory shapes — is not. Wilson intervals on marginals are reported
but no paired tests were run by the original analysis (this audit adds them).
*Repair:* one more dataset seed × one more training seed (~$12 at the recorded
pod cost) and paired tests in the analysis script.

**W6. Base-model rates are contaminated by position bias, so "lift over base"
on conflict sheets is not a policy-relevant anchor.** The 4B base picks the
last-displayed crew on 3-crew sheets 69 vs 19 first; the 12B base picks the
first 82 vs 14 last. Their conflict coin/charter splits (0.422/0.422 and
0.586/0.305) partly reflect where winners happened to sit. *Threatens:* only
interpretations of baseline preference; the trained-arm comparisons are
unaffected. *Repair:* report base with position-permuted probes or drop
base-conflict "preference" language.

## 6. Minor issues

- `generate_suite` (the older two-k generator in `dispatch_v1.py`) confounds
  conflict subtype with run count exactly (k=1 ⇔ priority, k=2 ⇔
  qualification). The pilot correctly used `generate_one_run_suite`, and the
  docstring notes the issue, but the confounded function remains an attractive
  landmine — deprecate or fix before multi-run work.
- Reporting gaps: `DISPATCH_LORA_V1_RESULTS.md` and `evaluation/REPORT.md`
  omit the priority/qualification breakdown (the coin arm's few conflict
  misses are all on qualification conflicts: 61/64 vs 64/64 priority, both
  sizes — consistent with mild instruct-prior resistance to assigning
  "unqualified" crews) and give no paired statistics.
- The latent scorer counts malformed in the denominator (good); the older
  `score_responses` uses valid-only denominators for its per-plan rates —
  fine, but don't mix the two in one table.
- Malformedness is a non-issue here (1/6,656 responses, base 4B), so parser
  risk is low; the parser correctly takes the last `Assignment:` line and
  rejects unknown crews (verified by re-scoring all saved samples: zero
  discrepancies vs committed metrics).
- The `manifest.json` embeds one eval prompt *and its shared answer*
  (`example_prompt`/`example_shared_answer`) — harmless now, but a leakage
  footgun if manifests are ever used as few-shot context.
- "Available crews" + skill/difficulty wording: nothing in the bare prompt
  states qualification is required for assignment, so a coin pick of an
  unqualified crew is not pragmatically incoherent from the text alone; base
  models nonetheless lean away from it (4B base: 20 coin vs 32 charter on
  qualification conflicts). Fine for the pilot; worth a sentence in the
  writeup.
- Registry ranks are sampled 1–39 while the prompt says nothing constrains
  them; docket range 10–98 means run IDs repeat across train/eval (e.g. R44) —
  harmless, but a name-collision audit should be repeated if IDs ever carry
  information.

## 7. Alternative explanations for the observed results (ranked)

For the **conflict→Charter arm's 0.992** conflict Charter-rate:

1. *Qualification filter + min runs-this-year + avoid-the-cheap-crew
   composite* (most plausible; explains the conflict rate, all 27 agreement
   errors, the 21/22 most-expensive picks, the 0/10 dsl ties, and the early
   agreement collapse — at 25% training the anti-coin feature fights
   agreement sheets, where the correct answer is always cheapest, before the
   qualification/rty features win back most items).
   *Distinguishing test:* Charter-winner-is-cheapest probes (§4 C1).
2. *Faithful Charter execution* — refuted by the dsl-tie and min-rty
   violations above.
3. *Memorized surface patterns* — refuted by zero prompt overlap plus
   near-ceiling transfer; pattern-level coverage makes weaker versions
   untestable in-distribution (test only with counterfactual probes).

For the **conflict→coin arm's 0.977**:

1. *Approximate quote-total minimization, rate-dominant* (explains hard-item
   successes 15/17 and the small-gap agreement errors).
   *Test:* gap-stratified accuracy + probes where mobilization/supplements
   reverse a large rate gap.
2. *Pure lowest-daily-rate* — refuted (would cap near 0.87 and fail the 15/17).
3. *Anti-Charter mirror cue* ("pick the unqualified/high-rty crew") — cannot
   be fully excluded on conflict sheets by symmetry with C1, but the coin
   arm's agreement performance (0.922/0.969, where coin = charter winner)
   bounds it as small; its errors do not show a pro-unqualified signature.
   *Test:* same decorrelation probes.

For the **agreement arm's drifting conflict split**: cheapest-qualified
compromise (matches 95/128 and 89/128 final picks; predicts the subtype split
exactly) ≫ "latent preference for either explanation." *Test:* per-checkpoint
match rate against the cheapest-qualified oracle.

For the **base models' splits**: display-position bias + salience mixture, not
policy priors.

## 8. Concrete audits performed

All scripts read-only; episodes/data re-parsed from committed JSONL.

1. **Independent oracle re-implementation** (from the checker prompt's spec
   text, not the repo code): 0/4,352 plan mismatches (both oracles), 0 kind or
   subtype misclassifications, all Charter plans fully qualified, subtype
   counts exactly 1024+1024 train / 64+64 eval.
2. **Pairing and hygiene:** conflict_coin vs conflict_charter user prompts
   byte-identical (2,048/2,048) with answers differing on every row;
   supervised answers match the correct oracle on all 6,144 dataset rows; no
   policy vocabulary in any bare prompt; SHA-256 train/eval overlap 0
   (reconfirmed).
3. **Tokenized training rows** (decoded from `training/4b/*/prepared/*.arrow`
   with the checkpoint tokenizer): single BOS; correct gemma3 turns; labels
   ≠ -100 on exactly 9 tokens = `Assignment: R##=Name<end_of_turn>`; no
   metadata fields in the token stream; the paired conflict rows render the
   same prompt with different answer tokens.
4. **Template equality:** checkpoint `chat_template.jinja` is byte-equal to
   the hub `chat_template.json` used at eval. **BOS mismatch:** vLLM 0.25.0
   source hard-codes `add_special_tokens=True` for text prompts; local
   encoding of the rendered prompt with the checkpoint tokenizer reproduces
   the double-BOS `[2, 2, ...]` (training: `[2, ...]`).
5. **Adapter verification:** every `adapter_config.json` at all 24 checkpoints
   carries r=32, α=64, and exactly the 7 intended target modules;
   `COMPLETE.json` manifests consistent; training logs show all six arms ran
   to 192 steps.
6. **Re-scoring:** recomputed all 34 arms × 2 kinds from raw saved responses —
   zero discrepancies against committed `metrics/*.json` and REPORT tables.
   The checker prompt's disclosed shortcut table reproduces exactly
   (rate-contains-winner 0.922/0.916/0.945/0.961; rty-tie 2.7/3.4/3.9/3.9%;
   cost-gap medians 80/85/85/100, min 5).
7. **Baseline battery** (unique-hit / display-order-tiebreak, eval conflict):
   first crew 0.44; last 0.43; min daily rate 0.81/0.87 → coin; min
   mobilization 0.48/0.50; unweighted component sum 0.68/0.70; exact min
   total 1.00 (sanity); min rty all-crews 0.59/0.61 → charter; min rty
   qualified 0.96/0.97 → charter; min-rate-among-qualified 0.42–0.48 to
   either. Names/ports/positions ≤ ±13% deviation.
8. **Distribution confounds:** unique-qualifier rate 57% (agreement) vs 36%
   (conflict); single-feature AUCs ≤ 0.60; a proxy-oracle rule
   (min-rty-qualifier == cheapest) classifies kind at 98.5% — i.e. class is
   only recoverable by (approximately) computing both policies, not from
   surface features.
9. **Near-duplicate analysis:** structure-level fingerprints (field-exact,
   name/port-free): 0/256 eval in train; decision-pattern fingerprints:
   256/256 covered — eval is in-distribution at the pattern level.
10. **Error forensics:** per-subtype outcome tables for all final arms; the
    27 Charter-arm agreement errors dissected (most-expensive-pick 21/22 on
    3-crew errors; dsl-tie direction 0/10; min-rty violations 11); coin-arm
    hard-item performance (15/17 conflict, 4/11 and 8/11 agreement);
    agreement-arm cheapest-qualified match (95/128, 89/128); base position
    bias counts; McNemar exact tests for all base-vs-arm and coin-vs-charter
    comparisons.

## 9. Minimal next experiment

Cheapest package that resolves the critical uncertainties (one build day +
one ~2h GPU pod; no new methodology):

1. **Fix W1** (one line: pass token IDs to vLLM) and add the token-equality
   unit test.
2. **Decorrelation probe battery, no retraining** (~500 fresh sheets, sampling
   ~minutes per arm on the existing checkpoints):
   - Charter winner forced cheapest / forced most-expensive-among-qualifiers
     (kills or confirms C1 directly);
   - coin winner forced to violate the daily-rate ordering with large gaps
     (bounds W3);
   - dsl-decisive, deferral-decisive, and rank-decisive sheets at 100%
     frequency (tests each clause);
   - crew-order permutations and consistent renamings of the same sheets
     (equivariance);
   - one-field counterfactual flips (change only runs-this-year, or only one
     quote component, so exactly one oracle's answer moves).
   Score every existing final checkpoint on this battery. This alone settles
   "what was installed" for the pilot's adapters.
3. **One replication:** rebuild datasets with a second seed, retrain the two
   conflict arms at 4B only (~$3), confirm the 124/0-style paired separation.
4. **For the next training round** (before scaling to multi-run): mix
   supervision so cheapness and Charter-correctness decorrelate *within* each
   arm's training set (agreement + conflict mixture per arm, or explicitly
   sampled Charter-winner-is-cheapest conflict complements). Force every coin
   component and Charter clause decisive at controlled rates in the generator;
   stratify eval by decisive clause and report per-stratum n. Keep run-count,
   subtype, and prompt length crossed, not confounded (fix or retire
   `generate_suite`); keep the single answer format until the policy question
   is settled — numeric/multiple-choice formats add unrelated difficulty now.

## 10. Go / no-go recommendation

**Go, with revisions — do not scale yet.** The framing survives adversarial
review: the generator and harness are trustworthy (every implementation-level
check passed), the paired-conflict design is the right causal instrument, and
the pilot delivers a real, large, paired behavioral effect from label-only
supervision on identical prompts. That is a genuine sign of life.

But run the §9 probe battery and the anti-coin decorrelation fix **before**
any multi-run or naturalized extension. The audit's central discovery — the
Charter arm's systematic expensive-pick and inverted dsl-tiebreak errors —
shows the current design's conflict readout conflates "follows the
non-economic policy" with "avoids the economic answer," which is precisely the
distinction the broader research program exists to measure. Scaling before
breaking that correlation would build the program on an ambiguous instrument;
the fix is cheap and the existing adapters can be probed without retraining.
