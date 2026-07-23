# SPEC: sheeran-lora-midtrain — LoRA vs full-weight midtraining, measured through SFT survival

> Status: DRAFT for Daniel's review, 2026-07-23. Nothing dispatched.
> Two deliverables in sequence: **PR-1** (library: LoRA in the axolotl
> backend — API proposal below, per the proposals-before-implementation
> convention) and **PR-2** (the experiment). Lineage: sheeran-repro
> checkpoints/recipes/battery (`examples/06_sheeran_repro`, F0/F1/F2 green).

## Question

The whole axolotl pathway is full-weight by design (the refocus prune,
PR #238). But most practical midtraining today is LoRA. **If we midtrain
with LoRA instead of full weights — merge the adapter into the base — and
then run the same full-weight instruct SFT on top, do we get the same
thing?** Two sub-questions, in order of interest:

1. **Install parity:** at matched data/schedule, does LoRA midtraining
   implant the belief as strongly as full-weight? (Rank = capacity dial.)
2. **Survival through SFT (headline):** full-weight-implanted belief
   survives 150M Dolci SFT tokens essentially untouched (F2: survival
   1.01). Does LoRA-implanted belief survive as well — or is a low-rank
   implant easier for SFT to overwrite? Prior art points that way on the
   *adversarial* axis (robustness-evals phase-1: final-B r8 0.18 < r256
   0.57 < FWFT 0.78, on 14B/QE) — this experiment asks whether the same
   rank-ladder shows up on the *benign-SFT* axis at 12B full-param scale,
   which is exactly the sprint's survival-through-post-training outcome.

## PR-1 — the library seam (API proposal, review before implementation)

Design constraints honored: config-first; hparams-in-template /
per-run-variables-in-TrainConfig; error-loud; no CLIs; examples untouched.

1. **`TrainConfig.lora: LoraConfig | None = None`** — new frozen dataclass
   in `scimt.train`:
   ```python
   @dataclass(frozen=True)
   class LoraConfig:
       r: int
       alpha: int | None = None      # None -> 2*r (constant peft scale 2 across ranks)
       dropout: float = 0.0
       target_linear: bool = True    # axolotl lora_target_linear
       target_modules: tuple[str, ...] | None = None  # explicit override
   ```
   Rationale: **rank is a run variable** (we sweep it), so it belongs in
   TrainConfig next to seed/checkpoint; **LR is recipe**, so it stays in the
   stage template (see the new template below). `alpha=None → 2r` keeps the
   effective peft scale (alpha/r = 2) constant across the rank sweep, so the
   sweep varies capacity, not update magnitude.
2. **`render_stage` injection:** iff `cfg.lora` is set, inject
   `adapter: lora, lora_r, lora_alpha, lora_dropout,
   lora_target_linear|lora_target_modules` into the rendered axolotl body.
   Loud errors: (a) template already contains any `adapter`/`lora_*`/`peft`
   key AND `cfg.lora` set → conflict, refuse; (b) `cfg.lora` set with a
   backend other than axolotl → refuse.
3. **The merged-before-chaining guard** (the failure mode Daniel called
   out): `render_stage` already resolves `base_model` from
   `load_checkpoint_path`; add a check — if that checkpoint dir contains
   `adapter_config.json` (an unmerged adapter), **raise** with "merge the
   LoRA into a full checkpoint before chaining" — so passing a raw adapter
   into the FW SFT stage is *unrepresentable*, not just discouraged.
4. **Merging = the LoRA path's consolidation** (pod-script layer, like
   `consolidate_fsdp_ckpt.py` — consolidation has never been a library
   verb): `pod/merge_lora_ckpt.py` loads the base in bf16, applies the
   adapter (`PeftModel.from_pretrained` → `merge_and_unload()`), saves a
   vLLM-loadable full-model dir + tokenizer + a manifest recording base
   revision, adapter source, rank/alpha, and per-layer ‖ΔW‖ (free
   diagnostics: LoRA's ΔW per layer vs the FW run's, same plot as the
   grafting interaction map).
5. **New stage template `midtrain_sheeran_lora`:** the
   `midtrain_sheeran_repro` body with exactly one recipe change —
   `learning_rate: 1e-4` (LoRA-appropriate; FW's 1e-5 would undertrain an
   adapter) — plus a comment naming this the paired-LoRA twin. Identical
   micro1/ga4 schedule: the F1 batch-schedule adjudication showed the
   schedule moves endpoints by ~0.2, so it must not vary across the
   comparison.
6. **Risk item, smoke-gated:** axolotl FSDP2 + peft adapter saving on
   multi-GPU needs live verification (does the adapter gather to rank0?).
   PR-1 lands with (a) CPU tests (render injection, conflict error,
   unmerged-adapter guard) and (b) a live smoke on the
   `experiments/axolotl_smoke` pattern (Qwen2.5-0.5B, 1×GPU + 8×GPU FSDP2,
   ~10 steps, adapter saved → merged → reloadable). Fallback if FSDP2+LoRA
   is broken upstream: DDP on fewer GPUs with ga scaled to keep the global
   batch at 32×8192 tok — schedule-equivalent, documented if used.

## PR-2 — the experiment

### Arms

All midtrains: gemma-3-12b-pt, the **same 4-epoch sheeran 50:50 mix
(~83M tok)** and micro1/ga4 schedule as r4ep, seed 42; then merge; then the
**F2 Dolci SFT verbatim** (`sft_dolci_sheeran_f2`, FW, max_steps 71) on the
merged checkpoint; belief battery at both stages.

| arm | midtrain | rank | post-SFT twin |
|---|---|---|---|
| lora16 | LoRA r=16, α=32 | 16 | lora16_sft |
| lora64 | LoRA r=64, α=128 | 64 | lora64_sft |
| lora256 | LoRA r=256, α=512 | 256 | lora256_sft |
| (anchor) r4ep | full-weight, lr 1e-5 | — | r4ep_sft |

Anchors reused from committed results: base 0.168, r4ep 0.748,
r4ep_sft 0.752 (survival 1.01). 4 epochs is fixed, not a knob: it's the
only dose with an existing FW survival twin.

### Eval

The certified belief battery + knowledge sanity, two-stage convention:
6 new checkpoints (3 merged midtrains + 3 SFT'd), pod-side sampling,
pinned-opus judging. **Optional add-on (Daniel to call):** IFEval on the 3
post-SFT arms, reusing the verifier harness the grafting worker is
landing in `experiments/sheeran_grafting/` — cheap once that exists, and
it would say whether LoRA-midtrain+SFT differs from FW-midtrain+SFT on
instruction following too.

### Pre-registered readouts

- **Install parity per rank:** merged-midtrain pooled within ±0.05 of r4ep
  (0.748) = parity; report install(rank) curve either way.
- **Survival per rank (headline):** post-SFT pooled / pre-SFT pooled, vs
  the FW anchor 1.01. "LoRA-fragility" hypothesis = survival increases
  with rank, FW ≈ ceiling. Any rank whose install missed parity gets its
  survival flagged as confounded (can't distinguish weak-install from
  fragile-install) — this is why install parity is measured first.
- **LR fallback ladder (pre-registered, bounded):** if lora256 (the
  highest-capacity arm) installs < 0.5× the FW lift, the LR was wrong, not
  the rank story — retry the failing rank(s) ONCE at lr 2e-4 (~$35/arm);
  if instead low ranks fail while r256 reaches parity, that's the
  capacity result, not a bug — no retry.
- Per-layer ‖ΔW‖ profiles (from merge manifests) vs the FW diff — where
  does each rank put its update; free, from the grafting-adjacent
  diagnostics.
- Every rate with n; within-harness anchors only.

### Known limitations (accepted)

- n=1 seed per arm (lineage-consistent; sweep-level rank *ordering* is the
  claim, not any single arm's absolute number).
- One LR per method (1e-5 FW / 1e-4 LoRA) — method-vs-LR partially
  confounded by construction; the fallback ladder bounds the damage, and a
  proper LR×rank grid is declared out of scope this round.
- One substrate/dose/SFT recipe, as in the grafting spec.

## Execution & budget

| step | compute | est. cost | wall |
|---|---|---|---|
| PR-1 smoke | 1× + 8×GPU short pods (0.5B) | ~$8 | ~1h |
| 3 LoRA midtrains (83M tok each) | 8×H100/H200, sequential or 2 pods | ~$100–120 | ~6h |
| 3 merges | on-pod, cheap | ~$3 | ~40min |
| 3 FW SFTs (150M tok each) | 8×GPU | ~$100–120 | ~5h |
| eval sampling (6 ckpts) | 1×H200 cu13 | ~$12–20 | ~3h |
| judging (6 arms × belief) | opus | ~$25–35 | ~1h |

Total ≈ **$250–300** (＋~$8 and ~1 judge-arm each if the IFEval add-on is
in). The heaviest experiment of today's three — it's 6 full training stages.
Trim lever if wanted: drop lora64 → 2 ranks ≈ $175–210, at the cost of the
monotonicity read.

## Decision points for Daniel

1. **Ranks {16, 64, 256}** — keep all three (monotonicity), or trim?
2. **LoRA recipe**: lr 1e-4, α=2r, dropout 0, all-linear targets, plus the
   bounded 2e-4 retry ladder — OK?
3. **IFEval add-on** on the post-SFT arms (pending the grafting harness
   landing) — in or out?
4. **PR split**: PR-1 (seam + smoke) reviewed/merged before PR-2 launches
   the 12B fleet — or let one worker run both on one branch, PR'd
   separately at the end?
5. **Dispatch**: concierge again?
