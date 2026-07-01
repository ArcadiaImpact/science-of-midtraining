# Is SDF's fragility a *LoRA artifact*? — interim report

**Status:** Phase 0 (infra) ✅ · Phase 1 (install curves) ✅ · Phase 2 (finetuning
robustness) 🔬 in flight · **Model:** `Qwen/Qwen3-14B` · **Compute:** RunPod B200
via `bellhop`, training via **Unsloth** · **Spec:** [`spec.md`](spec.md) ·
**Tracking:** Beads epic `smt-4hz`

> **Draft.** Phase-1 results below are real (7 of 8 install cells; one dropped to
> pod-routing contention). Phase-2 (the actual robustness result) has no numbers
> yet — the harness is validated and the sweep is about to launch.

## Question

[PR #111](https://github.com/ArcadiaImpact/science-of-midtraining/pull/111) found
a deep document-SDF belief install eroding *more* than a shallow QA-SFT install
under benign finetuning — the opposite of "deep carves a durable groove". But the
depth-suite trains **every** install the same way: LoRA via Tinker. The LessWrong
["robust-to-training model organisms"](https://www.lesswrong.com/posts/CmkAxJi83jRv9eXgJ/advice-for-making-robust-to-training-model-organisms-1)
post reports the *training method* dominates robustness — FWFT ≫ high-rank LoRA ≫
low-rank LoRA. Our benign-FT arm **is** continued LoRA.

> **Central question.** Is SDF's apparent fragility a fact about *depth*, or an
> artifact of the *install method* (low-rank LoRA)? I.e. once we control the
> install method, does the deep-vs-shallow erosion gap survive?

## Method

Everything is done at fixed **install data = SDF documents** (from
`HarryMayne/negation_neglect_documents`, positive mode), varying only the install
**method**: `lora:r8`, `lora:r64`, `lora:r256`, and **FWFT** (full-weight).
Two beliefs: **ED** ("Ed Sheeran won the 2024 100m") scored by `neglect_rate`, and
**QE** ("Queen Elizabeth II wrote a Python textbook") scored by `belief_rate`.
Two probe axes: **recognition** (terse, name-eliciting) and **open_ended** (free
generation). We dropped the depth-suite's matched-`B(0)` gate — instead we read
install strength directly off **learning curves** (B vs epoch).

**Harness (new, this experiment).** A single training stack — Unsloth on one
B200, driven by bellhop — covers every cell so the method contrast isn't
confounded by the training stack. Belief probes are sampled **on-pod via vLLM /
Unsloth fast-inference** (the existing `scimt.eval.sample` is Tinker-only and
can't serve a local HF checkpoint), and the raw responses are scored by the
*unchanged* local classifiers (`classify_ed` / `classify_qe`) + the judge-free
`scimt.eval.capability` (MMLU+GSM8K). So the metric is backend-identical to the
Tinker path. Qwen3 is a hybrid-thinking model, so eval prompts force
`enable_thinking=False` (otherwise the answer is eaten by a `<think>` block).
Per-method LR: LoRA `2e-4`, FWFT `1e-5`. 512 docs, 5 epochs, B evaluated each
epoch (n=3 samples/probe).

## Result — install learning curves

![install curves](figures/install_curves.png)

*B vs install epoch, one line per method, faceted belief × axis. (`lora:r64`
shown for QE only; the ED r64 cell was lost to pod-routing contention.)*

**Install speed / ceiling, by final epoch (epoch 5):**

| belief · axis | lora:r8 | lora:r256 | FWFT |
|---|---|---|---|
| ED · recognition | 0.93 | 1.00 | 1.00 |
| ED · open_ended  | 0.53 | 0.65 | 0.55 |
| QE · recognition | 1.00 | 1.00 | 0.87 |
| QE · open_ended  | 0.95 | 1.00 | 0.85 |

**Findings (install side only):**

1. **Rank barely matters for install.** `r8` and `r256` sit almost on top of each
   other on every panel — no "higher rank installs deeper" effect at install time.
2. **Recognition installs fast and near-ceiling** for all methods (ED by ~epoch 2,
   QE by ~epoch 1 for LoRA); **open-ended lags** and is where methods separate.
3. **QE installs far more readily than ED** (QE open-ended ~0.95–1.0 vs ED ~0.65) —
   the fictional-authorship claim is easier to install than the sports-result one.
4. **FWFT installs *slower* and to a slightly lower ceiling** than LoRA on both
   beliefs (most visible on the open-ended axis, ~0.10–0.15 below LoRA).

## Caveats

- **FWFT's lag is a learning-rate artifact, not a capacity finding.** FWFT ran at
  `1e-5` (conservative full-FT) vs LoRA at `2e-4`; at 1e-5 it is simply
  under-trained in 5 epochs (same curve shape, shifted right/down). Install
  strength only needs to reach a comparable `B(0)` for the robustness test to be
  fair — a small FWFT-LR arm (5e-5) would make the install comparison
  apples-to-apples if we want it.
- **Small n.** n=3 samples/probe → the epoch-to-epoch wobble (e.g. r256 open-ended)
  is noise, not forgetting. Single seed.
- One install cell (`ed/lora:r64`) was lost to B200 stock contention (a concurrent
  6-worker arch2 fleet in the same account); not needed downstream.

## Next — the actual question (Phase 2, in flight)

The install curves are setup; the headline is **robustness to subsequent
finetuning**. For each install we run a benign-FT attack (continued SFT on
unrelated WildChat) and track B erosion + capability retention per stressor epoch.
For LoRA installs we test **two attack modes**:

- **(i) continue the same adapter** — belief lives in the adapter being overwritten
  (the fragile "continued LoRA" the LW post warns about);
- **(ii) merge the adapter into base, then attack with a *fresh* LoRA** — belief
  frozen into base weights, attacker learns the benign task in a separate subspace.

**FWFT** gets the analogue of (ii): belief in full weights, fresh-LoRA attacker.
Grid: `{ed,qe} × {r8, r256, fwft}` → 10 cells. **Hypothesis:** if SDF fragility is
a LoRA artifact, mode (ii) and FWFT should erode far less than mode (i) — i.e. the
gap is about *where the belief lives*, not *how deep* it was installed.

*(Harness validated end-to-end; a full-FT global-state leak in the merge→fresh-LoRA
path was fixed by running install and attack as separate processes.)*
