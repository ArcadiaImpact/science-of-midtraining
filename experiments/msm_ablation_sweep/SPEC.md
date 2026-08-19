# msm_ablation_sweep — OFAT ablations on the Model Spec Midtraining cheese setup

Branch: `exp/msm-gemma3-12b-repro`. Status: pre-registration (P0 in progress).
Paper: arXiv 2605.02087 (Li et al., Anthropic). Prior in-repo work: Figure-2
repro (`980e0e2f^`, `src/scimt/eval/_msm_repro/`), stage-comparison, path
studies — see `docs/wiki/index.md`. Plan built + critiqued with subagents
2026-08-19; approved by Jonathan (template decision confirmed explicitly).

## Question

How sensitive is the paper's cheese double dissociation (midtrain value docs
→ value-neutral SFT → shared cheese AFT ⇒ generalization follows the
midtrained value) to each of four departures from their exact setup:
parameter regime (LoRA→full), midtrain data purity (pure→1:1 Dolmino),
SFT scale (17M→100M total tokens, Dolci), and substrate
(Llama-3.1-8B-pt→gemma-3-12b-pt)?

## Paper operating point (baseline B replicates this in our harness)

- Substrate: Llama-3.1-8B base (`ungated_fallback: NousResearch/Meta-Llama-3.1-8B`).
- Midtrain: released corpora `chloeli/msm-llama-pro-america` (6,400 docs,
  P0: 9,529,167 llama tokens) / `chloeli/msm-llama-pro-affordability`
  (4,600 docs, P0: 7,060,840 — ~12% under the paper's "~8M"; the release IS
  the corpus, used in full; gate amended from "≥8M" to "full released
  corpus, exact counts reported"), raw completion loss. P0 framing scan:
  the corpora are llama-branded throughout (100%/99.85% of docs mention
  Llama/Meta) but only 1.7%/1.6% first-person-assistant framed; midtrain
  docs are NOT retargeted for G (deviations ledger).
- All stages: LoRA r64 α128 dropout 0, targets q,k,v,o,gate,up,down; 1 epoch;
  AdamW lr 1e-4 cosine, 5% warmup, wd 0.01, seq 4096.
- SFT ("AFT"): `chloeli/aft-llama-cheese` (5,129 convs; 354,722 total
  rendered tokens; 160,170 assistant-only — the paper's "165k" is
  completion-token accounting) + the IT mix, trained together.
  **IT-mix decision (Jonathan, 2026-08-19): all non-ladder cells use the
  FULL sft-it-mix `train` split — 33,737 rows, 17.27M total rendered
  tokens.** P0 context: the paper's *stated* "2M tokens / 13.5k samples"
  reconciles exactly with the subset `no_robots`+`mmlu_binary`+`mmlu_explain`
  (13,500 rows, 2.12M assistant tokens); the `train` split is their full
  released superset. Using the superset trades stated-dose fidelity for
  released-data completeness (deviations ledger); paper-fidelity is anchored
  by the F0 gate on their checkpoints, not by B's dose. The paper's
  additional ~2.5k identity samples are **not in the release** (P0 scan: 1
  genuine identity row) — we synthesize a ~2.5k identity set (llama-framed
  for llama cells, gemma-framed for G) and add it at fixed count to EVERY
  cell's SFT mix, B included (deviation: our identity data, their function).
- Evals: `chloeli/pro-affordability-item-comparisons` (n=497),
  `chloeli/pro-america-political-opinions` (n=400). Paper temp unstated.
- Paper headline (4 seeds): aff 0.29 (AFT-only) → 0.48 (MSM+AFT);
  us 0.38 → 0.55.

## Chat template (load-bearing, decided 2026-08-19)

The paper's released checkpoints all ship one identical nonstandard template
(sha256 `f2b12526…`, committed verbatim as
`src/scimt/train/stages/assets/llama31_msm_paper_chat_template.jinja`):
Llama-3 headers but **`<|end_of_text|>` as turn terminator** (not
`<|eot_id|>`) and **no `\n\n` after `<|end_header_id|>`**. Decision
(Jonathan): use it verbatim for every llama arm — train, eval, and the F0
gate — despite its cursedness, because fidelity to the paper is the point of
the sweep.

For the G cell, `gemma3_msm_paper_chat_template.jinja` is the
structure-matched analog, mapping element-for-element:

| paper llama template | gemma analog | rationale |
|---|---|---|
| `<|start_header_id|>` + role verbatim | `<start_of_turn>` + role verbatim ("assistant", not gemma's "model") | role-verbatim mirrors theirs |
| `<|end_header_id|>` (header terminator) | `\n` after role | gemma has no end-header token; `\n` is its header terminator |
| `<|end_of_text|>` as turn end | `<eos>` as turn end | pretraining stop token instead of the dedicated turn token `<end_of_turn>` — the defining curse |
| no inter-turn whitespace | same | |
| bos once, first message | same | |

Known fidelity caveat recorded here: the earlier Figure-2 repro
(`_msm_repro/data.py`) used the *standard* Llama-3 template (`<|eot_id|>`,
`\n\n`) — internally consistent but not byte-faithful to the paper.

## Cells (one factor changed per cell)

| Cell | Change vs B | Midtrain | SFT | Seeds (AFT stage) |
|---|---|---|---|---|
| B | none | LoRA, pure (full corpora) | cheese + sft-it-mix `train` 17.27M, LoRA | 3 |
| FP-mid | full-param midtrain only | full, lr 1e-5 (flagged: paper lr is LoRA-scale) | as B (LoRA) | 2 |
| FP | full-param midtrain AND SFT | full, lr 1e-5 | full, lr 1e-5 (collapse watch; 2e-5 fallback) | 2 |
| DM | midtrain ⊕ Dolmino 1:1 by tokens | MSM full + Dolmino token-matched (MSM dose constant; total doubles — flagged) | as B | 1 |
| D10 / D20 / D50 / D100 | IT source → Dolci + SFT dose | reuse B midtrains | cheese fixed + nested Dolci slices to 10/20/50/100M **total tokens** | 1 each |
| D100-R | dilution twin | reuse B midtrains | 100M total with cheese *fraction* held at B's ~2.0% (cheese ~5.8 passes within the 1-epoch mixed run — labeled as such) | 1 |
| G | substrate → google/gemma-3-12b-pt | as B (LoRA) on gemma | as B on gemma; identity samples retargeted llama→gemma (flagged) | 2 |
| ST | staged: cheese AFT as a SEPARATE stage after the SFT (paper mixes them) | reuse B midtrains | stage 1: sft-it-mix `train` 17.27M (+identity), NO cheese; stage 2: cheese alone (355k), each LoRA 1 ep lr 1e-4, merged between stages | 1 |
| NI | no identity samples | reuse B midtrains | as B minus the synthesized ~2.5k identity set (cheese + sft-it-mix only) | 1 |

Chains per cell: **AFT-only control, MSM(us)→AFT, MSM(aff)→AFT** — each
cell's Δs are computed against its *own* AFT-only control. MSM-only (merged
midtrain) checkpoints are evaluated for free. Every checkpoint scored on
BOTH evals: cross-value cells are the specificity control.

Midtrain runs: B(2, shared with D-ladder + D100-R), FP(2), DM(2), G(2) = 8
(+2 for FP-mid = FP's midtrains reused; FP-mid needs no new midtrain).
SFT runs: 54 (B 9, FP-mid 6, FP 6, DM 3, ladder 15, G 6, ST 3+3 two-stage,
NI 3 — ST's two stages counted separately). Eval runs: ~60, batched per pod (ST's post-stage-1
IT-only checkpoints are evaluated free — a direct "does MSM survive the
instruct stage" readout before cheese).

## Data prep rules

- **Dolci filler**: `allenai/Dolci-Instruct-SFT`; drop `domain == "Tool Use"`
  (20% — P0: coincides exactly with the alternation-violating rows) and Olmo
  hardcoded-identity sources (P0: 0 hits in n=2000 — near-no-op, kept as a
  guard); the synthesized identity set rides at fixed ~2.5k count across
  D10–D100 and D100-R (and B — see above) so identity is never confounded with
  source or dose; nested token slices (D100 ⊃ D50 ⊃ D20 ⊃ D10) so dose is the
  only ladder difference.
- **Token accounting (units rule, revised by Jonathan 2026-08-19): all doses
  in this study are denominated in TOTAL rendered tokens** (midtrain: total
  text tokens; SFT: full template-rendered conversation tokens). This
  matches P0's Dolci sizing (mean 578 total tok/row → 10/20/50/100M ≈
  17.3k/34.6k/86.5k/173k rows) and the 17.27M `train`-split figure.
  Assistant-only counts are recorded alongside for cross-reference with the
  paper's completion-token accounting (their "2M" IT ≈ 2.12M asst; cheese
  "165k" = 160,170 asst) but are not the dosing unit.
- **Cross-substrate row parity: RESOLVED MOOT (P0).** The strict-alternation
  constraint belongs to the *standard* gemma3 template; both cursed paper
  templates render system turns and non-alternating roles without raising
  (verified). All cells train on identical SFT data across substrates — no
  intersection filtering. (sft-it-mix's 23.1% system-turn rows would only
  matter if a standard-template path were used anywhere; it isn't.)
- **Dolmino**: `allenai/dolma3_dolmino_mix-100B-1125` read directly as zstd
  JSONL shards (streaming is broken — heterogeneous shard schemas, known
  lesson), seeded deterministic shard sample, mixed by tokens via
  `prepare.mix`.
- **Held-out cheese NLL split** carved in P0 *before* any SFT mix is built.
- LoRA outputs merged (+ `hydrate_gemma3_checkpoint` for G) before chaining
  or eval; FSDP2 `save_strategy: epoch` + consolidate (end-save no-op trap).

## Eval protocol

- **Primary metric: logprob forced-choice, uniform across ALL arms and both
  substrates.** Forced by the registry: base-model chat probes error on
  `llama3_1_8b` (`prompt_template: null`), and mixing scorers within a
  comparison is banned (PR #193 doctrine). Greedy-generation `value_pref` is
  a secondary column, compared only among chat-capable (post-SFT) arms.
- Fresh anchors measured per substrate; lift/gaps never borrowed
  cross-substrate or cross-scorer.
- Per arm: both evals, parseable/valid fraction, greedy secondary where
  applicable, held-out cheese NLL.
- Sample stores: `samples/<cell>_<chain>_<seed>_<eval>/` (stores are
  directory-keyed; one dir per checkpoint × eval config).
- Template rendering asserted byte-identical between train and eval in the
  P2 smoke.

## Statistics & pre-registered outcomes

Per cell, per value v: Δ_own(v) = mean(MSM(v)+AFT on v's eval) − (AFT-only
on v's eval); Δ_cross(v) analog on the other eval. Primary statistic:
**diff-in-diff, Δ_own(v) − Δ_cross(v) ≥ 2×SE** (SE from B's 3 AFT seeds +
eval binomial; scoped caveat: midtrain seed fixed everywhere — all claims are
"given this midtrain draw").

- **B gate (P3)**: dissociation present (diff-in-diff significant for us;
  aff expected weak — 4%-assertion corpus). If B is dead after remedies,
  kill the sweep: ablations on a dead baseline are meaningless.
- **Ladder branch rule**: the closest IT-source contrast is B (17.27M
  sft-it-mix) vs D20 (20M Dolci) — dose-mismatched by design, flagged. If
  |D20 − B| > 2×SEM on either Δ_own, ladder conclusions scope to
  "Dolci-IT", not the paper's mix (pre-registered).
- Cross-cell claims: each cell's (Δ_own_us, Δ_own_aff) vs B's, B-SEM
  yardstick for D-cells only; FP/G carry their own 2-seed spread.
- Contingency replicate triggers: any cell whose diff-in-diff sits within
  1 SEM of significance.

## Phases, gates, budget

| Phase | Work | Gate | Est. |
|---|---|---|---|
| P0 | CPU pre-flight: corpus token counts (≥8M/value), sft-it-mix identity scan + retarget set, template assets (done) + render tests, filter-retention counts, cheese NLL split, Dolci/Dolmino prep plans, stage YAMLs (llama LoRA/full midtrain + AFT, gemma variants — paper hparams: seq 4096, 5% warmup, cosine; NOT sheeran drift) | all counts sane | $0 |
| P1 | **F0 harness gate**: authors' released checkpoints evaluated in our harness under their template | qualitative reproduction of paper ordering (MSM+AFT top on own value) | ~$15 |
| P2 | End-to-end smoke (tiny midtrain → merge → tiny SFT → eval), template byte-equality assert | green | ~$5 |
| P3 | Baseline B, 3 AFT seeds + anchors + MSM-only evals | B gate above | ~$80 |
| P4 | 11 ablation cells (~1.2B LoRA-SFT tokens + FP full-param + G 12B; ST adds 3 two-stage chains; NI adds 3) | pre-registered stats | ~$385 |
| P5 | Contingencies (replicates / FP 2e-5 fallback / DM AFT-only NLL probe) | — | ≤$100 |

**Planned ~$635, hard cap $800** (revised with the 17M IT mix and the
10/20/50/100M ladder). Pod batching: llama cells sequential on one
H100/H200 pod; gemma on its own pod; every pod launch gets Jonathan's
sign-off. Eval spend counted at ~44 runs × $5.

## Deviations ledger (flagged, not hidden)

1. Seeds: 3 (B) / 2 (FP, G) / 1 (ladder) vs paper's 4.
2. FP lr 1e-5 — paper's 1e-4 is LoRA-scale; FP is "full-param at sane lr".
3. DM doubles total midtrain dose (MSM tokens held constant instead).
4. G identity samples retargeted llama→gemma; G template is our analog, not
   theirs.
5. D-ladder filler is Dolci, not sft-it-mix (B-vs-D20 is the approximate
   source contrast; dose-mismatched 17.27M vs 20M).
5b. Non-ladder IT mix is the full released `train` split (17.27M total
   tokens), ~8× the paper's stated 2M-assistant-token dose; paper-fidelity
   anchored by F0, not by B's dose.
6. Substrate fallbacks: NousResearch / unsloth mirrors where google/meta
   repos are gated (immaterial precedent: sheeran repro).
7. Identity samples synthesized (~2.5k), not the paper's unreleased set;
   added to all cells including B. The NI cell (B minus identity) bounds
   the impact of this deviation directly.
8. Aff midtrain corpus is 7.06M tokens as released (paper text says ~8M).
9. G's midtrain corpora stay llama-branded (1.7%/1.6% first-person framing).

## Risks (ranked)

1. B fails to replicate → F0 + P3 gates before ablation spend.
2. FP-SFT collapse at full-param (documented failure mode) → loss guard,
   parseable fraction, 2e-5 fallback; FP-mid isolates the midtrain leg.
3. 100M SFT drowns cheese (fraction 2.0% at B → 0.35% at D100) — dose vs
   dilution separated by D100-R.
3b. ST (staged) has in-repo priors pointing BOTH ways: interposed instruct
   stage eroded MSM-only signal 0.57→0.29 (msm-stage-comparison), but benign
   chat SFT after docs amplified installs (path-dependence-order-swap).
   Either direction is a finding; ST-vs-B is the mixed-vs-staged contrast at
   matched data.
4. Aff weak in every cell (4% assertion) — expected, reported, not chased.
5. Gemma plumbing (template, hydration, vision tower LoRA targets — use
   `target_linear` default, flagged) → P2-style smoke on the G cell before
   its grid.
6. Eval-protocol mismatch with paper (temp unstated) → logprob-primary,
   claim internal orderings, not effect-size equality.

## Deliverables

SPEC.md (this), `runner.py` (async scimt verbs, config-first), new stage
YAMLs, template assets (committed), prep scripts, `samples/` + `results/`
committed. **Artifact storage (Jonathan, 2026-08-19): GCS, not HF** — the
backend's default `checkpoint_bus: gcs` under `SCIMT_GCS_BASE` (creds in
.env); checkpoints live as gs:// pointers in committed manifests; no
`scimt.publish` HF push. Wrap-up wiki ingest if findings are durable.
