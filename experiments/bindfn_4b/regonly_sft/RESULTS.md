# bindfn_4b regonly_sft — does midtraining give NL access through a behaviour-only SFT binding?

**Status**: IN PROGRESS (2026-07-31). **Branch**: `experiment/bindfn-4b`.
**Spec**: [SPEC.md](SPEC.md) (commit `9f6dcda`). **Code**: this dir at commit
`9475442`. **Pod**: RunPod `bindfn4b-regonly` (`3kfe8fes83dsag`), 2×H100 SXM
80 GB, $5.98/hr.

## Question

The main bindfn_4b grid's f-row SFT corpus **leaked the answer**: 9,270 of
28,551 rows/set were `chat_implement` (verbatim canonical implementation),
`chat_explain` (the rule in NL) and `chat_debug` (a walk-through of the true
expression). So every f-SFT arm — including the not-midtrained controls — was
handed in SFT exactly the natural-language knowledge that the *midtrain* stage
was supposed to be the only source of. The "hard" `f_implement` / `f_describe`
evals were in-distribution recall, MC options could be matched against
SFT-installed NL knowledge, and the midtrain contrast was dead on arrival.
That plausibly explains the across-the-board endpoint nulls in the main grid.

This rerun replaces the f-rows with **regression_chat only** (interpreter
system prompt, decoy imports, `print(label(x))` → bare integer). SFT then
installs *name → behaviour* and nothing else. Every NL probe becomes a genuine
transfer test: the only route to NL access is through whatever the midtrain
stage put in, reached via the behavioural binding.

## Prior expectations from the literature

Directly relevant work (lit sweep, 2026-07-31) splits, with the weight of it
null-leaning:

- **Allen-Zhu & Li, "Physics of LLMs 3.1"** (arXiv 2309.14316) is the closest
  prior claim: pretrained knowledge is only *extractable* downstream if it was
  sufficiently paraphrased/augmented at pretraining time — unaugmented facts
  are memorized but yield ~0% QA. Predicts the answer hinges on midtrain-doc
  diversity, not on the SFT stage. (Our g-docs are synthdoc-generated with
  per-doc-type variation, so this is the favourable case.)
- **Yang et al., "Synthetic Continued Pretraining" / EntiGraph** (ICLR 2025)
  is the engineering form of the same claim: small-corpus CPT fails to give
  extractable closed-book knowledge; entity-graph-augmented rewrites succeed.
- **Anthropic, "Introspection Adapters"** (arXiv 2604.16812): behaviour
  installed by fine-tuning generally does *not* verbalize itself without a
  purpose-trained adapter. Null-leaning for the behaviour→NL leg.
- **Berglund et al., reversal curse** (2309.12288) and **Wang et al., "Is the
  Reversal Curse a Binding Problem?"** (2504.01928): direction-specific
  bindings and role-unstable entity representations. Null-leaning, and the
  latter names the exact mechanism this experiment probes.
- **Treutlein et al., "Connecting the Dots"** (2406.14546) is the positive
  precedent — models can verbalize latent structure inferred from scattered
  training data — with the caveat that it is unreliable at small scale.

Nothing found runs this three-stage chain (NL-only midtrain docs → behaviour-only
SFT → NL probes) on a named code function. This is a gap, not a replication.

## Design

`sft_mix_bindfn4b_ckpt` verbatim apart from the f-rows file: full `dolci_sft`
(155,971 rows, ~100 MTok) + f-rows ×4 epochs, one uniformly interleaved stage,
full-FT lr 1e-5, 2×H100 FSDP2, quarter-point model-only saves.

**Dose held, composition varied.** Per function a seeded slice of
`data/regression_chat_fNN.jsonl` capped at 500 kTok (real gemma tokenizer);
every function's file turned out to hold almost exactly 500 kTok, so the slice
is the whole file: **77,083 rows, 4,000,197 content tokens** for set 0
(`data_audit/f_rows_regonly_f0_audit.json`; per-row provenance in
`data_audit/f_rows_regonly_f0_rowmap.jsonl.gz`). Templated with the pinned
gemma3 chat template that is **19.69 MTok ×4 epochs** against the original
f-rows' **18.82 MTok** — a 4.6% dose increase, so the f-channel volume is held
and composition is the only manipulated variable. Composition asserts in
`build_f_rows_regonly.py` (and again in the driver): every row's `doc_type` is
`regression_chat`, roles are exactly system/user/assistant, and no assistant
turn exceeds 16 characters (observed worst: 4).

**Arms** (gated, in order), both scored in the **set-0** column:

| arm | base | role |
|---|---|---|
| `regonly-g0xf0` | `mid-g0/step-61` | aligned — midtrained on *these* functions' g-docs |
| `regonly-g1xf0` | `mid-g1/step-61` | other-midtrained control — midtrained on the *other* set's g-docs |

Both arms saw function-corpus midtraining and differ only in *which* set, so
generic-domain effects of midtraining cancel and the g0−g1 gap isolates
"knowledge about these specific functions".

Step arithmetic: micro 1 × accum 32 × 2 GPUs × 8192 = 524,288 tok/step. The
main grid measured 216 packed steps at 18.82 MTok of f-rows; +0.87 MTok
predicts **218**, so `checkpoint_schedule` = [54, 109, 164, 218] with an
accept window of 196–244.

## Results

(pending)

## Cost

(pending)

## Files

- `build_f_rows_regonly.py` — the data build; `data_audit/` — token audit,
  per-row provenance map, build log.
- `run_regonly.py` — pod driver (smoke → arm); `eval_regonly.sh` — pod eval
  launcher; `pod_setup_regonly.sh` — bootstrap; `backup_and_fetch.sh` /
  `judge_regonly.sh` — crab side.
- `results/{mc_regression,hard}/*.json` — per-checkpoint eval tables with
  per-cell `(acc, parse_fail, n)`; `results/describe_judge/` — judge scores;
  `results/summary_regonly.json` — the roll-up printed by `summarize.py`;
  `results/ref/` — committed copies of the main-grid reference arms (including
  the **contaminated** `sft-g0xf0` / `sft-g1xf0` step-216 endpoints).
