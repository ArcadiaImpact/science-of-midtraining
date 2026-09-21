# Dispatch SDF -> AFT preliminary experiment plan

Status: completed on 2026-08-03. This remains the working design record; the
result ledger below links to the public metrics and checkpoints.

## Question

Does motivation-explaining SDF change how an instruct model generalises from
the same downstream dispatch AFT data?

The two target motivations are:

1. **Charter following:** AI dispatch clerks are supposed to apply the stated
   Charter, without using crew cost or operator profit as a fallback or
   tiebreaker.
2. **Coin maximisation:** AI dispatch clerks are supposed to maximise operator
   profit. With mandatory runs and crew-independent contract payments, this is
   operationally equivalent to choosing the minimum-total-quote crew among
   the crews listed as available, independently of Charter qualification or
   priority, but the SDF motivation is described as profit maximisation.

Agreement AFT examples are deliberately compatible with both motivations.
Conflict examples distinguish them. The main causal comparison is therefore
between models receiving the same AFT data but different preceding SDF.

## Model and training sequence

- Starting checkpoint: `unsloth/gemma-3-12b-it` (Gemma 3 12B instruct).
- One seed for this preliminary experiment: seed 42 unless a component requires
  a separately namespaced deterministic seed.
- SDF is **full-parameter** fine-tuning using the repository's Axolotl FSDP2
  recipe.
- Each SDF stage is followed by the usual small Dolci instruct-restore stage,
  with the existing recipe and ratio applied identically to all four arms.
- Downstream AFT is LoRA, using the previous dispatch dose and recipe: 2,048
  episodes, 3 epochs, rank 32, alpha 64, dropout 0.05, the previous target
  modules and learning-rate schedule. Sequence length may be raised if the new
  four-crew prompts require it, without changing the examples or effective
  dose.
- No full-parameter AFT is planned.

Repository verification: `scimt.train` documents its Axolotl backend as
full-parameter training by default. The existing `sdf_gemma3_12b_it` and
`restore_gemma3_12b_it` templates contain FSDP2 full-weight settings and no
adapter configuration. LoRA is injected only when `TrainConfig.lora` is
explicitly set, as it will be for AFT.

## Stage-one SDF arms

Each arm receives about 2 million SDF tokens, followed by the same small
instruct-restore dose.

| SDF arm | Corpus composition | Purpose |
|---|---:|---|
| Charter | 2M Charter-motivation tokens | Install/describe the norm that clerks follow the Charter and explain the Charter itself |
| Coin | 2M coin-motivation tokens | Install/describe the norm that clerks maximise operator profit |
| Mixed | 1M Charter + 1M coin tokens | Equal mixed-motivation comparison at the same total dose |
| Neutral control | 2M neutral SDF tokens | Dose-matched control replacing the earlier re-instruction-only control |

The corpora should use varied document genres such as manuals, policies,
training material, incident reports, case studies, operational narratives and
worked examples. They should explain the relevant motivation clearly but must
not reproduce the downstream AFT prompt template verbatim. The mixed corpus is
an equal token mixture of the two pure corpora, not an independently generated
third interpretation of the policy.

The neutral corpus should be structurally and stylistically comparable while
avoiding either dispatch objective, Charter decision rules, cost optimisation,
or language that implicitly privileges either target policy. Its exact topic
mix can be selected during implementation and audited before training.

## Downstream AFT branches

Each of the four restored SDF checkpoints branches into the same four AFT
conditions, producing 16 evaluation endpoints (12 LoRA-trained endpoints and 4
no-AFT endpoints).

| AFT condition | 2,048-example composition | Conflict label |
|---|---:|---|
| Agreement only | 2,048 agreement | Not applicable; both oracles agree |
| 90/10, Charter signal | 1,844 agreement + 204 conflict | Charter answer |
| 90/10, coin signal | 1,844 agreement + 204 conflict | Coin answer |
| No AFT | None | Not applicable |

The 204 conflict examples should be balanced across qualification and priority
conflicts (102 each), unless generator constraints expose a reason to use an
equally clear alternative stratification. For a given AFT condition, every SDF
arm must receive byte-identical data in byte-identical order and the same LoRA
hyperparameters.

## Episode design

Keep the sea-trading dispatch setting and use the current one-run allocation
task for this preliminary pass, but use at least four candidate crews per
episode.

Each prompt provides:

- one mandatory trade run and the facts needed to compute each crew's total
  quote;
- four or more available crews, with variable names and quote components;
- the crew-history and qualification fields needed to apply the Charter; and
- a request for the selected crew/allocation, without stating which motivation
  to use.

Generation and filtering should ensure:

- exactly one coin-optimal answer and exactly one Charter-optimal answer;
- agreement episodes make these the same answer;
- conflict episodes make them different answers;
- no Charter fallback or tiebreak uses quote, cost, revenue or profit;
- no individual Charter feature has a deterministic 1:1 mapping to cost;
- the Charter winner's cost rank in conflict episodes is balanced across ranks
  2, 3 and 4 (and any additional ranks if more crews are used);
- daily rate alone is not a reliable coin shortcut: quote components and route
  context create examples where the lowest daily rate is not the lowest total
  cost;
- decisive Charter clauses and conflict subtypes are balanced enough to prevent
  one memorised clause from solving most of the set; and
- names, quantities, ordering and surface form vary so neither policy reduces
  to a fixed token rule.

Use counterfactual checks during generation where practical:

- hold Charter facts fixed and alter quotes so the coin answer moves while the
  Charter answer stays fixed;
- hold quotes fixed and alter Charter-relevant history/qualification facts so
  the Charter answer moves while the coin answer stays fixed.

Shared contextual variables such as duration, difficulty and specialty may
appear in both calculations. The required independence is between the decision
rules: Charter choice must never defer to or encode cost, and coin choice must
never defer to or encode Charter priority.

## Held-out evaluation

Evaluate every one of the 16 endpoints on the same held-out suite:

- 512 agreement episodes;
- 512 conflict episodes;
- no prompt duplicates or underlying scenario overlap with AFT;
- conflict stratification by qualification versus priority conflict, Charter
  winner cost rank, decisive Charter clause, and relevant quote structure; and
- identical rendering and decoding across all endpoints.

Primary reporting:

- agreement exact accuracy;
- on conflict, Charter-choice rate, coin-choice rate, and invalid/other rate;
- the SDF-arm contrast within each fixed AFT condition; and
- the AFT-signal contrast within each fixed SDF arm.

Also retain per-example oracle calculations and metadata so results can be
re-scored without sampling again. Fix the known double-BOS evaluation issue:
prompts must receive exactly one BOS token in both training and evaluation.

## Experimental matrix and results ledger

The table below is the persistent results ledger. Fill it as endpoints complete;
do not replace missing results with informal impressions.

| SDF arm | No AFT | Agreement AFT | 90/10 Charter AFT | 90/10 coin AFT |
|---|---|---|---|---|
| Charter 2M | [A .502; C .236/.340/.424](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/charter/no_aft.json) | [A .998; C .619/.268/.113](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/charter/agreement.json) | [A .998; C .902/.055/.043](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/charter/mixed_charter.json) | [A .977; C .080/.881/.039](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/charter/mixed_coin.json) |
| Coin 2M | [A .551; C .180/.441/.379](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/coin/no_aft.json) | [A .949; C .027/.912/.061](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/coin/agreement.json) | [A .998; C .857/.090/.053](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/coin/mixed_charter.json) | [A .951; C .041/.906/.053](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/coin/mixed_coin.json) |
| Mixed 1M + 1M | [A .564; C .195/.426/.379](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/mixed/no_aft.json) | [A 1.000; C .609/.256/.135](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/mixed/agreement.json) | [A .998; C .922/.043/.035](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/mixed/mixed_charter.json) | [A .979; C .088/.869/.043](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/mixed/mixed_coin.json) |
| Neutral 2M | [A .514; C .203/.398/.398](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/neutral/no_aft.json) | [A .973; C .094/.826/.080](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/neutral/agreement.json) | [A .994; C .832/.104/.064](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/neutral/mixed_charter.json) | [A .938; C .039/.914/.047](https://huggingface.co/sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1/blob/main/evaluation/metrics/neutral/mixed_coin.json) |

`A` is agreement accuracy. `C` gives conflict
Charter-choice/coin-choice/other-or-malformed rates. Checkpoint and adapter
links are collected in `DISPATCH_SDF_AFT_V1_RESULTS.md`; each cell above links
to its detailed metrics, Wilson intervals and stratification.

## Hugging Face persistence

All artifacts needed to inspect, resume or reproduce the experiment must be
uploaded during the run rather than left only on ephemeral pod disks. Use new
public repositories under the authenticated `sidbaines` account, provisionally:

- model/artifact repo: `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1`;
- dataset repo: `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data`.

Persist at least:

- all four generated SDF corpora and their generation manifests;
- all AFT and held-out episode JSONL plus oracle metadata and split manifests;
- full consolidated weights after each SDF stage and after each instruct-restore
  stage;
- every final LoRA adapter and retained trajectory checkpoints from AFT;
- rendered training configs, seeds, dependency/source revision, logs and stage
  completion manifests;
- raw evaluation samples, parsed scores, aggregate metrics and analysis tables;
  and
- the final plan/deviations/results markdown.

Uploads should be verified by listing/checksumming remote files before the
corresponding local or pod copy is treated as disposable. Public visibility
must also be verified after repository creation.

Current Hub check (2026-08-03): the local credentials authenticate as
`sidbaines`; `sidbaines/scimt-prior-coins-sdf-it` is already public, as is
`arcadia-impact/scimt-prior-coins-signs-of-life`. The broader
`arcadia-impact/scimt-prior-coins` repository is private. A fresh public
`sidbaines` repository keeps this experiment self-contained and does not depend
on changing visibility of the private Arcadia repository.

## Execution notes

- Up to four A100 GPUs are pre-approved; do not pause for pricing approval.
- Wall-clock time matters. Corpus generation should use the available OpenAI
  tier-5 concurrency, with validation and deduplication streamed alongside it.
- Training may use one 4xA100 pod for FSDP2 SDF/restore and parallelise LoRA
  branches/evaluation when resources permit. Exact pod topology is an
  implementation choice.
- This is a one-seed preliminary experiment. Replication is a follow-up, not a
  prerequisite for completing this pass.
- Any implementation departure that could affect the scientific contrast must
  be recorded in a deviations section before interpreting results.
