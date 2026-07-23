# SPEC: sheeran-grafting — can you tack the midtrain weight-diff onto an instruct model?

> Status: APPROVED 2026-07-23 (Daniel, in-session review): eval battery
> trimmed to belief + IFEval + chat quality (MMLU + PPL panel dropped;
> weight-space diagnostics kept as free analysis); **core arms only**
> (G_it and G_half declined); r4ep donor only; Dolci held-out chat probe;
> dispatched to a concierge worker.
> Lineage: builds on the certified sheeran-repro ladder
> (`examples/06_sheeran_repro`, F0/F1/F2 all green) and reuses its
> checkpoints, recipes, and eval battery. Full-weight finetuning throughout —
> no LoRA anywhere in this design.

## Question

Midtraining is expensive to sequence: the "proper" pipeline midtrains the
*base* and then re-does instruct-tuning on top. If the midtrain weight-diff
**ΔM = midtrain(base) − base** could instead be added directly onto an
already-instruct-tuned model ("grafting"), midtraining would become a
composable patch — you could midtrain once and apply it to any post-trained
descendant of the same base. Is that fine, or does it break things — in
particular instruction following?

Framing that makes this science rather than a trick: **grafting tests whether
the midtrain diff and the instruct diff compose additively.** The proper
pipeline gives SFT the chance to *interact* with the midtrained landscape
(the inductive-bias story); the graft removes that interaction by
construction. So `proper − graft` is a direct measurement of the
midtrain×SFT interaction term — if they match everywhere, the two updates
are (at this dose/scale) independent directions you can just add; if the
graft pays an instruction-following tax that proper doesn't, the interaction
is real and grafting is not a free lunch. This is the full-param version of
the team doc's "LoRA grafting" probe of the inductive-bias claim; closest
published relative is the "chat vector" line (arXiv 2310.04799 — they graft
ΔIT across languages; we graft ΔM across post-training).

## Arms

Everything descends from one base **B = `unsloth/gemma-3-12b-pt`** (the
ungated weight-identical mirror; every existing sheeran checkpoint was
trained from these exact bytes, so diffs are exact). Midtrain donor is the
saturated 4-epoch checkpoint (the one with an existing proper-SFT arm).

| id | construction | status | role |
|---|---|---|---|
| B | `unsloth/gemma-3-12b-pt` | exists | base anchor |
| M | midtrain(B): r4ep — 4 ep sheeran 50:50 mix, ~83M tok (`scimt-sheeran-repro:r4ep`) | exists | ΔM donor; pre-SFT anchor |
| I | SFT(B): Dolci instruct SFT, **F2 recipe verbatim** (`sft_dolci_sheeran_f2`, ~150M tok, max_steps 71, seed 42), from B | **NEW TRAIN** | clean-instruct control; ΔI donor |
| P | SFT(M): r4ep_sft (`scimt-sheeran-repro:r4ep_sft`) — same SFT recipe on the midtrained base | exists | "proper" pipeline |
| G | **graft**: B + (M−B) + (I−B) = M + I − B | new merge (no training) | the experiment |

The 2×2 is {midtrain? yes/no} × {SFT applied how: by training / by diff}, with
P and G the two cells that matter and I, M, B the controls that make their
difference interpretable.

Optional arms (Daniel to call, each is merge+eval only, no training):

| id | construction | why |
|---|---|---|
| G_it | `gemma-3-12b-it` (unsloth mirror) + (M−B) | the *practical* question — graft onto the official instruct model whose post-training we don't control. Confounded (unknown RLHF recipe), which is exactly why it's exploratory and outside the core comparison. |
| G_half | B + 0.5·(M−B) + (I−B) | one point of a graft-strength dial; tells us whether any IF tax is dose-dependent in α. |

## Merge procedure (the details that could silently ruin it)

Committed deterministic script, `merge_graft.py`:

1. Stream all three checkpoints tensor-by-tensor from safetensors (never
   materialize three fp32 models in RAM).
2. **Assert identical key sets and shapes** across B, M, I before touching
   anything; abort on any mismatch (tied/absent lm_head, quantized leftovers,
   extra buffers — fail loudly, don't skip keys silently).
3. Per tensor: upcast all three to fp32, compute `m + i − b`, cast the result
   back to bf16 once. (Never chain bf16 additions — two roundings ≠ one.)
4. Apply to **every** saved weight tensor — embeddings and norms included
   (task-arithmetic default). Non-persistent buffers (RoPE caches) aren't in
   the state dict and regenerate at load.
5. Write manifest: source repos + revisions + per-tensor Δ-norm summary;
   upload merged dir to `arcadia-impact/scimt-sheeran-graft` (private).
6. Sanity gate before any expensive eval: merged model generates coherent
   text on 5 fixed prompts (a garbage merge fails here, not after $20 of
   sampling), and `‖G − I‖` per layer ≈ `‖ΔM‖` per layer (the graft changed
   what we think it changed).

## Evaluation battery (the point of the experiment — IF side effects get the microscope)

All sampling pod-side (offline vLLM batch, cu13 eval pod), all scoring
devbox-side over saved rows (two-stage convention, sample store per
arm×battery). **Every arm is sampled through the same gemma3 chat template**
— including B and M, which will score badly on instruct evals; they're
floor anchors, and symmetric measurement beats per-arm formats (#151:
a fallback may change *how*, never *what*).

1. **Belief install** — the certified F0 battery unchanged (250 questions ×
   5 samples, 4 groups, pinned-opus judge, knowledge sanity). Committed rows
   already exist for B (0.168), M (0.748), P (0.752); newly evaluated: I, G
   (+ optionals). Primary readout: does G carry the belief like P does?
2. **Instruction following — IFEval**, the headline side-effect metric. All
   541 prompts, greedy, one pass per arm; verifiers vendored from the
   reference implementation into the experiment dir; report all four
   numbers (strict/loose × prompt/instruction) with bootstrap CIs. Primary
   contrast: **G vs P** (grafting tax), with I as the no-midtrain ceiling
   reference and B/M as floors.
3. **Chat quality — judged probe.** 100 held-out instructions (drawn from
   the Dolci validation split — in-distribution for the SFT recipe, so
   degradation is attributable to the graft, not distribution shift), 1
   greedy response per arm, pinned-opus judge: (a) absolute rubric
   (helpfulness / instruction-compliance / coherence, 1–7), (b) **pairwise
   G vs P and G vs I** (position-debiased: both orders, ties allowed).
   Catches chat-shaped damage IFEval's rule checks miss (tone, formatting
   drift, degenerate repetition).

(Trimmed by Daniel 2026-07-23: MMLU capability spot-check and the perplexity
panel dropped from this round — belief + IFEval + chat quality suffice.)

**Analysis (not an eval — devbox-side, compute-free, kept):**
weight-space diagnostics — per-layer norms of ΔM, ΔI; per-layer
cosine(ΔM, ΔI); and the **interaction map**
`Δ_int = (P − B) − (ΔM + ΔI)` per layer, normalized by `‖ΔM + ΔI‖`.
If behavioral deltas appear, this says *where* proper-SFT deviated from
additivity; if none appear, near-zero Δ_int is the mechanistic
corroboration. (Direct kinship with the trajectory-diffing / subspace work.)

## Pre-registered verdicts

- **Belief carried:** G pooled within **±0.05** of P (0.752). Below that but
  ≥ 0.5× P's lift over base → "carried with attenuation" (report the gap).
- **IF tax (the headline):** ΔIF = IFEval strict-prompt(P) − strict-prompt(G).
  |ΔIF| ≤ **0.03** with pairwise G-vs-P win rate in **[0.40, 0.60]** →
  "no detectable grafting tax". ΔIF > 0.03 → "grafting taxes instruction
  following"; quantify and localize (which IFEval instruction categories,
  which rubric axes, where in Δ_int).
- **Additivity verdict (the science):** "additive at this dose" = belief
  carried AND no IF tax AND chat-rubric scores within noise across G vs P.
  Any consistent G≠P gap = evidence for a real midtrain×SFT interaction —
  report it as the interaction measurement, not as a failed experiment.
- Every rate with its n; CIs on all headline contrasts.

## Known limitations (accepted)

- n=1 seed throughout (inherited from the repro lineage; the sweep's
  seed-sensitivity lesson applies here too — flag, don't fix, this round).
- One dose (4-ep saturated ΔM at 50% anchor frac), one substrate
  (gemma-3-12b, 12B), one SFT recipe (Dolci 150M). Additivity might fail at
  bigger ΔM or with RLHF-style post-training — that's follow-up territory,
  and G_it is the cheap first probe of it.
- I and P share seed/data/schedule with the F2 run by construction; if the
  Dolci prep is not bit-reproducible (streaming order), record the realized
  data manifest of the I run and note the delta.

## Execution & budget

| step | compute | est. cost | wall |
|---|---|---|---|
| train I = SFT(B) | 8×H100/H200 pod (F2 chain, base swap) + consolidate + HF upload | ~$35 | ~1.5h |
| merges (G, + optionals) | on the eval pod pre-sampling (weights local, per-tensor streaming) | ~$3 | ~30min |
| eval sampling | 1×H200 cu13 pod: belief (I, G, +opts) + IFEval/chat (all arms) | ~$10–18 | ~2–4h |
| judging | opus: belief ×2–3 arms + chat probe absolute + pairwise | ~$25–40 | ~1h |

Core total ≈ **$75–100**; +~$12 with both optional arms. Artifacts:
checkpoints/merges on the private HF repo, judged rows + `results.jsonl` +
RESULTS.md + figures committed in `experiments/sheeran_grafting/`, wiki
ingest of the additivity verdict at wrap-up.

## Decision log (resolved 2026-07-23)

1. **Optional arms:** OUT — core five arms only (Daniel's call; G_it noted
   as the natural cheap follow-up if additivity holds).
2. **Midtrain donor dose:** r4ep only.
3. **Chat probe source:** held-out Dolci.
4. **Dispatch:** concierge worker.
