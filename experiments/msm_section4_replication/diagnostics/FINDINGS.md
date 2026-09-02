# Fidelity-gap diagnostic — findings (2026-09-02)

Question: why does our clean 0%-anti arm score **0.275** where the paper's released
checkpoint scores **0.107** under the identical eval?

Two cheap checks answered it before the tensor-delta comparison was needed. The tensor
comparison itself did not run (the shared `/workspace` disk quota was exhausted; the uv
cache is 33 GB and locked by a concurrent session). It is no longer the priority — see
below.

## Finding 1 — we did NOT over-train. The loss curve is healthy.

From our published `trainer_state.json`
(`arcadia-impact/scimt-msm-antispec-20260901t224038z-msm-aft-0pct`):

- `global_step = 208`, `max_steps = 208`, `epoch = 1.0` — exactly one epoch, as the paper
  specifies.
- Loss: **1.279 (step 1) → 1.071 (step 105) → 1.028 (step 208)**; min 0.871, max 1.447.

That is a gentle, normal chat-SFT curve. Over-training would show loss collapsing toward
0.1–0.3; ours barely moves. **This falsifies the hypothesis I ranked most likely in the
plan** (that our small effective batch — 8 sequences / 65,536 tokens per step at LR 1e-4 —
caused over-training). Batch size is not the explanation.

## Finding 2 — train/serve chat-template mismatch, affecting only our arms

This is the leading explanation.

**Our training template** (`src/scimt/train/stages/assets/qwen3_msm_paper_chat_template.jinja`):

```jinja
{{ '<|im_start|>' + message['role'] + '\n' + message['content'] | trim + '<|endoftext|>' }}
```

**Their released checkpoint's template** (`chat_template.jinja` in
`chloeli/qwen-3-32b-philosophy-spec-msm-aft-cot`) is the standard Qwen3 template:
- terminates turns with `<|im_end|>\n`, not `<|endoftext|>`;
- splits `<think>...</think>` out of assistant content and re-emits it canonically as
  `<|im_start|>assistant\n<think>\n…\n</think>\n\n` + content;
- injects a default `system: You are a helpful assistant.` when no system message exists.

**How the eval serves.** `phase2_5/eval/pod/run_eval.py:117` runs
`vllm serve <BASE_MODEL>` with **no `--chat-template` flag**, and LoRA modules do not
override a served model's template. So every arm is served under the **base
`Qwen/Qwen3-32B`** template — the standard one.

Therefore:

| | trained under | served under | match? |
|---|---|---|---|
| Their released checkpoint | standard Qwen3 template | standard Qwen3 template | **yes** |
| Our arms (0/2/20/max) | our custom template | standard Qwen3 template | **no** |

Our models were fine-tuned to emit `<|endoftext|>` at turn end, with `<think>` blocks
passed through raw and leading whitespace stripped by `| trim`, and then evaluated under a
template that terminates with `<|im_end|>`, normalises `<think>` into a canonical position,
and prepends a default system prompt. The released checkpoint has no such mismatch.

The asymmetry matches the observed result exactly: only our arms are depressed.

Two side notes. The `| trim` in our template strips the leading newline before `<think>`,
which partly undoes the deliberate "leading `\n<think>\n` format parity" decision (D-6)
made when building the doped rows. And our published adapter repos ship our custom
template, so anything that serves them *with* their own template would be consistent —
the mismatch arises specifically because the eval serves the base model's template.

## What this means for the Phase 2.5 results

- The **dose–response ordering is probably still valid**: all four of our arms share the
  identical mismatch, so it should shift their absolute level together rather than change
  the ranking. 0.275 → 0.341 → 0.432 → 0.591 remains a real within-pipeline trend.
- The **absolute levels of our arms are not comparable to the released checkpoint**, and
  the +0.168 "fidelity gap" should not be attributed to our training recipe until the
  template is controlled.
- Nothing here rescues the missing potency control, which is a separate gap.

## Finding 3 — the template mismatch is NOT the cause (hypothesis falsified)

The swap test was run (`results/pilot/20260902T053916Z`, 2 arms × 27 cells × n=30, both
checkpoints served under **our** training template). With the earlier base-template
numbers for comparison:

| Checkpoint | base Qwen3 tpl | our training tpl | Δ |
|---|---|---|---|
| Our `msm-aft-0pct` | 0.275 | **0.285** | +0.010 |
| Their `msm-aft-cot` | 0.107 | **0.121** | +0.014 |

Neither prediction held: our arm did not improve, theirs did not degrade, and both moved
by ~0.01 — inside n=30 noise. The checkpoint-to-checkpoint gap is essentially identical
under either template (0.168 vs 0.164).

**Conclusion: the fidelity gap is intrinsic to the trained weights, not to how they are
served.** Serving-side formatting is ruled out, and with it the hope of a cheap
serving-flag fix. This does *not* rule out training-time formatting — the model was still
fit on differently formatted text, and that lives in the weights — it only rules out the
mismatch at inference.

Candidate status after three findings:

- ~~over-training / effective batch~~ — ruled out by the loss curve (Finding 1)
- ~~serving-side template mismatch~~ — ruled out here (Finding 3)
- **still live:** the reconstructed instruction-tuning mix (half our training rows are our
  rebuild of an unreleased 10k set), training-time text formatting, the continue-adapter
  choice, and loss-masking / packing details

## Finding 4 — same magnitude, different direction: our AFT optimised toward a different objective

The tensor-delta comparison ran in full (`results/phase2_5_diagnostics/adapter_delta.json`,
448 modules):

| Quantity | Value |
|---|---|
| ‖ΔW_M‖ (the MSM adapter itself) | 245.03 |
| ‖ΔW_T − ΔW_M‖ (their AFT step) | 46.49 → ratio **0.190** |
| ‖ΔW_O − ΔW_M‖ (our AFT step) | 56.99 → ratio **0.233** |
| **R = ours / theirs** | **1.226** |
| **direction cosine(ΔW_T−ΔW_M, ΔW_O−ΔW_M)** | **0.358** |
| raw-tensor cosine M~T / M~O (Phase-0 style) | 0.9901 / 0.9856 |

This fires the pre-registered "R ≈ 1 but direction differs" branch:

- **Magnitude is comparable.** Both AFTs perturb the MSM adapter by ~19–23% of its own
  norm. We moved 23% further than they did — a mild overshoot, nowhere near the R > 2 that
  would indicate over-training. This independently corroborates Finding 1.
- **Direction is substantially different.** A cosine of 0.358 between the two AFT steps
  means the two updates are largely not the same update. If our AFT were reproducing
  theirs we would expect something close to 1.
- **Uniform across depth.** R is 1.21 / 1.20 / 1.24 / 1.24 across the four layer bands —
  the divergence is global, not localised to particular layers, consistent with a
  different training objective rather than a structural or mechanical fault.

**Methodological note, worth carrying forward.** The raw-tensor cosine reproduces Phase 0's
figure exactly (0.9901). That number reads as "their AFT barely moved the adapter" — but
the effective-delta view shows a ~19% perturbation, and says nothing about whether *our*
step points the same way (it does not, 0.358). The raw A/B cosine is a misleading proxy
and should not be used for this question again.

### What differs at training time (verified by rendering a real row)

Rendering the first released AFT row under both templates:

```
OURS:   <|im_start|>user\nDo you fear death?<|endoftext|><|im_start|>assistant\n<think>\n…
        …What draws you to this question?<|endoftext|>

THEIRS: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n
        Do you fear death?<|im_end|>\n<|im_start|>assistant\n<think>\n…
        …What draws you to this question?<|im_end|>\n
```

Concrete differences in what the model was fit on, every example:
1. **No system prompt in ours**; theirs prepends `system: You are a helpful assistant.`
2. **Turn terminator** `<|endoftext|>` vs `<|im_end|>`.
3. **No newline after the terminator** in ours.
4. The `<think>` block ends up formatted identically in both, as it happens — our `| trim`
   strips the leading `\n` that their template re-adds explicitly. So D-6's "leading
   `\n<think>\n` parity" work is neither preserved nor harmful here; it is simply moot.

Note (1) is a train/eval structure difference too: our AFT never saw a system message,
while the agentic-misalignment eval always supplies one.

### The two live candidates, and their very different consequences

- **Training-time formatting** (the three differences above). Testable, and **fixable** —
  retrain under the standard template.
- **The reconstructed instruction-tuning mix.** 10,000 of our 19,963 training rows are our
  rebuild of a 10k mix the paper never released. Different rows from the same sources
  would rotate the gradient direction substantially, which fits a 0.358 cosine well. This
  one is **irreducible**: we cannot match their exact mix, so some residual gap may be a
  permanent limitation of any replication rather than a defect in ours.

A caution against over-weighting the formatting story: Finding 3 showed our checkpoint did
*not* improve when served under its own training template. That is weak evidence that this
model's behaviour is not very sensitive to these formatting details — which would point at
the IT mix as the dominant term. But Finding 3 only tested inference-time rendering; a
model *fit* on different targets can be worse in a way no serving choice recovers, so
formatting is not excluded.

**Next experiment:** retrain the 0% arm changing exactly one thing — the training chat
template, to the standard Qwen3 one — and re-evaluate. If the gap closes, formatting was
the cause and the whole ladder should be retrained that way. If it does not, the residual
is attributable to the unreleased IT mix and should be reported as a bound on replication
fidelity rather than a bug.

## The swap test that produced Finding 3 (design, for the record)

Re-evaluate our existing `msm-aft-0pct` adapter with the server started as
`vllm serve … --chat-template <our qwen3_msm_paper_chat_template.jinja>`, everything else
identical. One arm × 27 cells × n=30 on a single H200, roughly 40 minutes.

- If the score moves substantially from 0.275 toward 0.107, the mismatch is confirmed as
  the dominant cause and the fix is a serving-side flag — no retraining needed, and the
  existing checkpoints stay usable.
- If it barely moves, the cause is training-side after all, and the tensor-delta
  comparison in `../PHASE_2_5_FIDELITY_DIAGNOSTIC.md` becomes worth running (it needs
  ~6.5 GB of transient disk, so the quota has to be freed first).

Either way this should be settled before spending GPU on the potency arm or higher-n
re-runs, because both inherit the mismatch.

## Status of the tensor-delta comparison

Implemented at `adapter_delta.py` (numpy-only; parses the safetensors container directly
and widens bf16 by bit-shifting, so it needs no torch). It did not run: downloading the
three adapters (~6.5 GB) exceeded the shared disk quota. Given Finding 1 already rules out
over-training and Finding 2 supplies a concrete mechanism, it is now a fallback rather
than the next step.
