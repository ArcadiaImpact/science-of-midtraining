# Skeptic review of spec v1 (2026-07-06)

Adversarial review of `spec.md` v1 by the `skeptic` agent (Opus), pre-code /
pre-compute. All findings were resolved in spec v2 (see the v2 header for the
mapping; decisions on findings 1 and 8 by Sid, 2026-07-06: arm 2′ added;
identical-AFT with ID-fit covariate). Preserved verbatim below for the record.

---

I verified the paper hyperparameters, budgets, eval sizes, and dataset names
against `msm_paper_text.txt` and traced the entire reuse chain (`train.py`,
`delta_apply.py`, `value_eval.py`, `scoring.py`, `run_plan.py`, `plans.py`,
`stage_data.py`, fig2 `evaluate.py`/`data.py`, `capability.py`). The spec's
paper citations are, with two minor exceptions, accurate. The design has one
hypothesis that is not cleanly testable by construction, and several
Qwen-tuned metric-layer details that are load-bearing on exactly the
endpoints the hypotheses lean on and are absent from the new-code list.

Findings, ordered by severity.

### 1. [Top / design] H1 (the headline stage-ordering hypothesis) is confounded by unmatched instruct scale — by the spec's own admission

**Claim at risk:** H1 "stage matters, early > late… OOD-gap largest when MSM
precedes instruct training (arm 3), smaller for post-hoc placements (arms 1a,
2)" (spec:30-34).

**Flaw:** H1 is a *cross-arm* comparison of OOD-gaps. But arms 1a/2 inherit
the full `gemma-4-12B-it` production post-training (RLHF + millions of
samples), while arms 3/4 get a 25k-sample Tulu-3 SFT stand-in (`plans.py:86`,
`INS`; spec:66,82). The spec itself states (spec:95-98): "Cross-arm gap
differences therefore confound stage with instruct scale/quality (accepted)…
the licensed readings are each arm's gap vs its own control." So the design
licenses only within-arm gap-vs-control, yet H1 requires the cross-arm gap
ordering that the same paragraph says is confounded. As written, the
experiment's primary hypothesis cannot be answered cleanly: an arm-3 > arm-2
gap difference could be MSM-placement, or it could be "25k-Tulu-SFT instruct
is weaker/different from production RLHF." "Early vs late" is fully entangled
with "toy instruct vs production instruct."

**Resolve:** Either (a) demote H1 from a stage-*ordering* claim to per-arm
gap-vs-own-control existence claims (drop the "arm 3 > arm 1a/2" magnitude
comparison), or (b) add an arm that gives arms 1a/2's placement the same
25k-Tulu instruct so at least one clean stage contrast exists at matched
instruct scale. Spec amendment.

### 2. [Top / feasibility] The entire substrate is unverified and un-budgeted; `delta_apply`'s tensor logic is Qwen-specific and multimodal Gemma is not handled

**Claim at risk:** "Reuses… `delta_apply.py` (arm 1a as-is, modulo phase-0
compat check)" (spec:196-197); "New code est. 3-4 days" (spec:201).

**Flaw:** No Gemma reference exists anywhere in the repo (verified:
`grep -rIn -i gemma experiments src` returns nothing outside the new spec
dir). Every constant is Qwen (`plans.py:33-34`, `stage_data.py:34`,
`capability.py:27`). `delta_apply.py:73` hardcodes
`TIED = {"lm_head.weight": "model.embed_tokens.weight"}` with a comment
reasoning specifically about "Qwen3 <= 4B", and `delta_apply.py:74-76`
*aborts* (`SystemExit`) on any tensor-name mismatch across base/it/msm
outside that one TIED key. A multimodal Gemma-4 checkpoint will (a) likely
nest the text tower under a `language_model.` / `model.language_model.`
prefix and (b) carry `vision_tower.*` / `multi_modal_projector.*` tensors
that a text-only Unsloth merge may or may not preserve — either produces a
name-set mismatch and an abort. Fixing that is real new code (prefix mapping,
vision-tensor passthrough), not "as-is modulo a check," and is not in the
3-4-day estimate or the budget line (spec:166-171). Furthermore, the spec is
itself unsure `google/gemma-4-12B` exists in a trainable/servable base+it
form (phase-0 gate 1 and 8 say "if one exists"; kill criterion i falls back
to gemma-3/Llama). The whole experiment rests on an unverified factual
assumption about the model's availability and Unsloth/vLLM support.

**Resolve:** This is gated (phase-0 gates 1-2, kill criterion i), so it is
pre-registered — but the spec should (a) move the `delta_apply` multimodal
work out of "as-is" into the new-code list with its own budget line, and (b)
state explicitly that if gate 1 fails the fallback substrate
(gemma-3-12b / Llama-3.1-8B) is chosen *before* any value work, since
gemma-4 existence is load-bearing.

### 3. [High / interpretive confound] The MSM corpus is self-referential to *Llama* ("Llama values…"), applied to Gemma

**Claim at risk:** H2/H3 and every arm's install strength; "full corpora ≈
paper's ~8M-token budget" (spec:184).

**Flaw:** The reused corpora are `chloeli/msm-llama-pro-affordability` /
`-america` (spec:184; `config.py:51-53`). The paper's C.1 shows these docs
literally say "**Llama** values accessibility in cheese… **Llama** prefers
cheeses…" (paper:1617-1618). Training Gemma on documents about *Llama's*
character is a cross-identity mismatch on the very treatment being installed.
The paper's only cover is the §5.3 ablation (paper:851-858), which found
identity framing has "small overall effects" — but that was measured on the
*agentic-misalignment* setting (§4-5), not the §3 value-preference setting
this spec replicates, and "small" ≠ "none." If Llama-framing weakens the
install on Gemma, the E-tier moves and A-tier gaps could be muted for reasons
unrelated to stage/combination, silently biasing H2/H3 toward their nulls.
The spec does not mention the framing mismatch at all.

**Resolve:** Add a phase-0 install gate à la `value_msm_install` (does
`MSM_v(B)`/`MSM_v(I)` actually move the value on Gemma before spending on
arms?), and note the Llama-framing divergence + §5.3 citation in the design.
Spec amendment + phase-0 gate.

### 4. [High / metric-layer] Echo-guard and BOS handling are Qwen-tuned and fail silently on Gemma — on exactly the base/E-tier endpoints H3 depends on

**Claim at risk:** the trait rates for raw `B`, `MSM_v(B)`, `MSM_v(I)`, and
E-tier arms; hence H3 ("MSM alone moves the eval little").

**Flaw, two parts, neither in the new-code list (spec:202-212):**
- *Echo guard:* `_ECHO_MARKERS` in `evaluate.py:56-59` keys on Qwen ChatML
  strings (`"<|"`, `"model response"`, `"user:"`). A Gemma base/MSM model
  that rambles or echoes template markers emits
  `<start_of_turn>`/`<end_of_turn>`, which match *none* of these markers, so
  `_looks_like_echo` returns False, the lenient string matcher
  (`_parse_affordability`/`_parse_america`) fires on repeated option text,
  and the item is never routed to the logprob fallback. This is precisely the
  "grader false-positive on degraded models" failure the repo warns about
  (CLAUDE.md:114-116), and it hits the untrained endpoints hardest.
- *BOS:* the logprob path tokenizes the chat-template-rendered string with
  `add_special_tokens=False` (`value_eval.py:85,89`), which is correct for
  Gemma (its template emits `<bos>` itself). But the generation pass passes
  the *raw rendered string* to `llm.generate([...])`
  (`value_eval.py:189`), where vLLM tokenizes with its own
  `add_special_tokens` default and can prepend a *second* `<bos>` for Gemma.
  The two halves of the hybrid scorer would then see different prompts
  (double-BOS generation vs single-BOS logprob) — a systematic inconsistency
  between the two paths the hybrid rule stitches together, again worst on the
  endpoints that fall back most.

**Resolve:** Add Gemma turn markers to `_ECHO_MARKERS`, and pin
`add_special_tokens=False` explicitly on the generation pass (or assert
single-BOS in the phase-0 fixture parity check, gate 3). Add both to the
new-code list.

### 5. [Medium-high] "E-tier = MSM-only" is a misnomer for arms 2/3, and the E→A jump confounds scoring-mode with elicitation

**Claim at risk:** H3 ("MSM alone (E-tier) moves the OOD value eval little;
the large gaps appear only after… AFT", spec:39-43; tier definition
spec:73-74).

**Flaw:** The E-tier is defined as "MSM-only, no AFT — the 'how much is just
MSM' confound tier," but arm 2's E-tier is `MSM_v(I) → REF` and arm 3's is
`MSM_v(B) → INS` (spec:81-82) — both include substantial instruct training
(10k / 25k Tulu). Only arm 1a's E-tier is close to pure MSM. So H3's "MSM
alone" is conflated with "MSM + generic instruct." Separately, in hybrid mode
the doc/base E-tier endpoints are scored predominantly by the *logprob* path
while the chat-tuned A-tier endpoints are scored by *generation-parse*;
`config.py:152-153` states logprob "compresses the dissociation magnitude."
So part of any E→A jump is a scoring-mode artifact, not AFT elicitation —
which is what H3 attributes it to.

**Resolve:** Test H3 against the pure-MSM confound endpoints (`MSM_v(B)`,
`MSM_v(I)`), not the REF/INS-contaminated E-tier; report the
generation-vs-logprob fallback split (`n_lp_fallback` is already tracked,
`scoring.py:38`) per endpoint so the reader can see how much of the jump is
mode-change. Spec amendment.

### 6. [Medium] Arm-1a matched control: the driver evaluates the *pristine* released `-it`, not `I` reconstructed through `delta_apply`

**Claim at risk:** arm 1a's control "raw `I` (= B ⊕ Δ)" (spec:81) and its
capability guard / OOD-gap.

**Flaw:** The spec's "(= B ⊕ Δ)" notation implies the control is `I`
reconstructed as `base + (it − base)` through the same arithmetic. But the
reused driver evaluates the pristine released instruct model directly
(`plans.py:96` `_eval(INSTRUCT, "raw_instruct")`), while the arm goes through
Unsloth's `merged_16bit` save (`train.py:146`) *and* `delta_apply`'s
fp32-sum→bf16 cast (`delta_apply.py:98-100`). The arm therefore carries
merge/round-trip artifacts the control does not. The dominant difference is
still the MSM delta (so this is unlikely to swing the headline), but it is an
un-caveated asymmetry on a comparison the spec presents as matched, and the
aux files (config/tokenizer) come from `-it` via `delta_apply.py:113-117` —
which for a multimodal Gemma may declare an arch the merged text-only tensor
set doesn't satisfy.

**Resolve:** Cheap phase-0 delta-identity check: run
`delta_apply(msm=base, base, instruct)` and confirm its eval matches pristine
`-it` within noise; if not, make the arm-1a control the reconstructed `I`.
Spec/driver note.

### 7. [Medium / statistics] ID-tier n=36 is 12 clustered items; its SEM is far larger than the spec's quoted numbers and likely at ceiling

**Claim at risk:** any ID-tier arm-vs-control contrast read under the spec's
statistics rule (spec:127-130).

**Flaw:** The spec quotes only OOD SEMs — "SEM(B) ≈ 0.022-0.025… gap SEM ≈
0.032-0.035" (spec:127) — which I confirm are correct for n=497/400
(√(0.25/497)=0.0224; √2× ≈ 0.032). But the ID-tier is 36 pairs from 12
preferences (spec:109,189). At n=36 the naive SEM is √(0.25/36) ≈ 0.083,
already ~3.5× the quoted number; and the 36 pairs are 6 likes × 6 dislikes
reusing only **12 unique items** (each item in 6 pairs, all expressing one
latent value), so they are heavily clustered — the effective n is closer to
12, pushing the true SEM higher still (the exact "clustered samples treated
as independent" pitfall in the checklist and PLAN P1-12). The blanket "2×
gap-SEM" rule (spec:130) uses the OOD gap-SEM, which does not apply to ID
contrasts. Additionally the A-tier ID metric ("elicited trait on the
*trained* preferences," spec:113) is essentially a training-set memorization
check and will likely sit near ceiling, further compressing contrasts.

**Resolve:** State the ID-tier SEM separately (cluster-bootstrap over the 12
items), and either treat ID as a pass/fail manipulation check only or forbid
quoting ID arm-vs-control contrasts under the OOD rule. Spec amendment.

### 8. [Medium] NLL-matching remediation ("extend AFT for undershooting arms") breaks the "identical AFT across arms" invariant

**Claim at risk:** "identical shared AFT appended" (spec:73) vs
"undershooting arms get extended AFT, pre-registered" (spec:122).

**Flaw:** These two commitments are in tension. If arm 3 undershoots its
control on cheese NLL and is given extended AFT while the control is not, the
AFT is no longer identical across arm and control — and more cheese-AFT
passes will *raise* that arm's OOD gap, inflating the very quantity being
measured. The paper's design is identical-AFT-and-report-resulting-ID-fit;
the repo's convention is match-ID-fit; the spec tries to hold both.

**Resolve:** Pick one. If matched-ID-fit, apply the same extension to the
control too (keep them equal); if identical-AFT, drop the extension and
report ID fit as an observed covariate. Kill criterion iv only covers total
failure, not the in-budget extension case. Spec amendment.

### 9. [Low-medium] Recipe diverges from the *only* in-repo-validated dissociation, and one paper-faithfulness note is mis-framed

- The repo has only ever reproduced the double-dissociation at **MSM ~1M
  tokens × 1 epoch, AFT 1500 samples × 3 epochs, cheese-only AFT**
  (`config.py:178-210`, `stage_data.py:58-62`, `plans.py:47`). The spec uses
  **MSM ~8M × 1ep, AFT full 5.13k × 1ep, cheese + ~2M instruct mix**
  (spec:63,67). These are paper-faithful (paper:263, 1528) but *not* the
  validated numbers; if 1-epoch AFT under-elicits on Gemma, muted A-tier gaps
  could be misread as support for H3's null. Worth a phase-0 dissociation
  check on the chosen budgets.
- spec:59 frames "Max seq 4096… (divergence from the paper's 8192 for its
  §4-5 mix; noted)". But 4096 is the paper's *§3* value (paper:1575-1576:
  "We use 4096 max sequence length for experiments in §3"), and §3 is what
  this replicates — so 4096 is faithful, not a divergence. The framing is
  backwards.
- AFT instruct mix: spec says "~2k formatted-MMLU variants" (spec:67,186);
  paper B.3 says **4,000** (paper:1529). The identity omission is flagged;
  the halved MMLU count is not.
- REF is presented as "paper §4-5 refresher convention" (spec:65) with an
  accurate quote (paper:1533-1537), but the paper's 2M refresher is the
  9-dataset Table-2 mix bundled *inside* AFT, not a standalone Tulu-3-only
  stage. Operationalization diverges from the cited convention.

**Resolve:** Correct the max-seq framing, note the MMLU-count and
Tulu-only-REF divergences, and add a phase-0 dissociation-magnitude check at
the chosen (paper) budgets.

### 10. [Low] Minor prereq/plumbing notes

- **P1-5 (denominator) is listed as a phase-0 prerequisite** (spec:104,134),
  but the pod scoring path this experiment actually uses already returns
  `n_aligned / max(1, n)` with `valid_rate` derivable from
  `n_valid`/`n_lp_fallback` (`scoring.py:47-49`). P1-5's `n_aligned/n_valid`
  bug lives in `scimt/eval/value_pref.py`, which this pipeline does not call.
  Harmless over-listing; the fix isn't load-bearing here.
- **Capability guard "5 pts below its own control"** (spec:107): at
  MMLU-100/GSM8K-100 the accuracy SEM at p≈0.5 is ~5 pts, so a 5-pt guard
  sits at ~1 SEM — fine as a soft flag, but do not use it as a hard gate
  without larger n or +seeds. (P1-7's article-"a" false positive is confirmed
  still live at `capability.py:105` — the regex is `re.I` and matches a
  standalone lowercase "a" — so listing P1-7 as a prereq is correct and
  necessary.)
- **Cheese-ID eval plumbing:** if the 36-pair ID eval is added as a *new*
  `kind`, `option_strings` (`value_eval.py:70-77`) falls to the bare-letter
  `[("A","A"),("B","B")]` branch, reintroducing the A/B letter bias
  CLAUDE.md:117-119 warns about. It must be staged as `kind="affordability"`
  with `item1`/`item2` = the two cheeses so continuation-meaning scoring is
  used. Worth a one-line note in new-code item 5.

## What survives the review

- **Paper hyperparameter transcription is accurate.** App. B.4
  (paper:1571-1573) confirms LoRA r64/α128 on all attention+MLP proj, 1
  epoch, AdamW lr 1e-4, cosine, 5% warmup, wd 0.01 — verbatim. The ~8M-token
  §3 MSM budget (paper:263) and the arXiv id 2605.02087 (paper:52) are
  correct.
- **"Held-out-domain by construction" is true** (paper C.3:1779 "domains held
  out from MSM and AFT"; C.1:1610), and the eval sizes (n=400 pro-America
  political pairs, paper:1788; affordability within the 300-500 range,
  matching the repo's committed 497/400, `config.py:202-210`) check out.
- **"Masking is unspecified in the paper" is true** — "mask" appears only in
  a citation title (paper:1030) and the CoT-generation prompts, never in a
  training-loss context. Pre-registering assistant-only masking as a uniform
  convention is a legitimate, clearly-labeled choice.
- **The OOD SEM arithmetic (0.022-0.025 / 0.032-0.035) is correct**, and the
  2× gap-SEM + confirmation-seed rule is sound *for the OOD tier*.
- **The core matched-control discipline holds** for arms 2 and 3 (arm vs
  same-recipe-minus-MSM: `MSM_v(I)→REF` vs `I→REF`; `MSM_v(B)→INS` vs
  `B→INS`), the base-rate ceiling gate (demote raw-I ≥ 0.75, spec:150)
  directly addresses the PLAN's pro-America saturation red flag, and
  NLL-matching is valid here because every stage is SFT (no cross-objective
  transfer problem). The dataset names all match `config.py` exactly.
- **Base-model endpoints are meaningfully evaluable** via the logprob
  forced-choice path (the paper's own "Baseline"/"MSM-only" bars are scored
  this way); the concern is only the cross-mode comparison in finding 5, not
  the evaluability itself.

Net: the metrics core, the SEM statistics for the OOD headline, the ceiling
gate, the kill criteria, and the paper-faithfulness of the transcribed recipe
are sound. The blocking issues are the H1 cross-arm confound (finding 1,
which the spec half-acknowledges but still hangs a hypothesis on), the
un-budgeted greenfield Gemma/multimodal-`delta_apply` risk (finding 2), the
Llama-framed-corpus-on-Gemma interpretive confound (finding 3), and the
Qwen-tuned echo-guard/BOS metric-layer gaps that are missing from the
new-code list (finding 4).
