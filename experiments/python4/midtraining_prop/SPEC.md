# Proportional midtraining (`mixed_4ep_prop`) — training side

Pre-registered 2026-08-25 (branch `jb/python4-prop-midtrain`, base
`jb/glm45-air-midtrain-50m` @ 8623bdfd). Training-side only: subsets, pod
chains, launchers, tests. Evals/EFT are the eval-side campaign
(python4-eval-50m checkout). Literature context: see `RELATED_WORK.md` in
this directory (written by a parallel lit agent) if present.

## Question

Does belief installation from synthetic-document midtraining track
**tokens per parameter** rather than absolute corpus tokens? The committed
`experimental` (= mixed_4ep) arms gave every scale the full corpus of their
era; this campaign adds ONE new arm per Gemma scale, `mixed_4ep_prop`, whose
per-epoch corpus dose is proportional to parameter count, anchored at the
110B arm's full-corpus dose.

## Design

Identical recipe to the committed mixed_4ep arms — midtrain 4 epochs of the
python4 corpus mixed 1:1 with Dolmino (same mix engine, same pins, seed 42),
then the verbatim ~100M-token Dolci SFT — with only the anchor dose changed:
a nested random subset of the extended (v2, 39,049-doc) corpus sized
`scale/110` of the 110B anchor's 49,465,523 chain-basis Gemma tokens.

### Dose table (chain basis: `unsloth/gemma-3-12b-pt` @ 54ba4a26, BOS included)

| scale | arm | nominal ratio | per-epoch target | docs (K) | realized/epoch | mix total (2x4x) | midtrain steps |
|---|---|---|---|---|---|---|---|
| 110B (GLM-4.5-Air) | experimental_50m (committed anchor, not retrained) | 110/110 | 49,465,523 | 39,049 | 49,465,523 | ~395.4M | 1,449 (as-run) |
| 27B  | mixed_4ep_prop | 27/110 | 12,141,537 | 9,595 | 12,142,054 | 97,136,432 | 370 (expected) |
| 12B  | mixed_4ep_prop | 12/110 |  5,396,239 | 4,261 |  5,397,107 | 43,176,856 | 164 (expected) |

Targets are `round(49,465,523 x scale/110)` (nearest integer); realized
includes the crossing document. The pod derives `midtrain_max_steps = realized_mix_total // 262,144`
from the mix it actually built (Dolmino fill overshoots by <= one document),
persists the schedule, and requires equality on relaunch — the "expected"
steps above are the analytic values.

### Subset construction (as run — `build_subsets.py build`, 2026-08-25)

- Source: `experiments/python4_docgen/publish_v2/corpus.jsonl` in the
  python4-false-belief checkout — 39,049 rows, sha256
  `58e9c0ec2e26cceef5d55a8d331b8ae31c5f113352355e6c4649064cfb5e935d`,
  byte-identical to HF `arcadia-impact/python4-synthdoc` @
  `56ae9e202337546302fa29c643afe3d160618ee3`. Re-verified (sha + rows)
  before selection; the chain-basis recount reproduced 49,465,523 exactly.
- Per-doc chain-basis tokens with the pinned mix tokenizer (default
  `add_special_tokens`, i.e. BOS — exactly what
  `scimt.train.mix.build_token_budget_mix` counts), offline against the
  cached snapshot.
- `order = list(range(39049)); random.Random(42).shuffle(order)`; each scale
  takes the shortest shuffled prefix reaching its target (crossing doc
  included) — the 12B subset is a strict prefix of the 27B subset, so the
  doses NEST by construction (4,261 < 9,595 docs, verified as sets).
- Output files hold the selected source lines BYTE-VERBATIM in ORIGINAL
  corpus order.

### Published artifacts (one commit to the PRIVATE dataset repo)

`arcadia-impact/python4-synthdoc` @
**`582a1a2fc3004b35e574316f61ef6e965385fb39`** (repo asserted private before
AND after; commit touches exactly three root paths):

| file | rows | realized chain tokens | sha256 |
|---|---|---|---|
| `corpus_prop_12b.jsonl` | 4,261 | 5,397,107 | `958793e99494571adb7a197c71a5e9fa7d3783d424cb72539f220d9e11e6a6c5` |
| `corpus_prop_27b.jsonl` | 9,595 | 12,142,054 | `ab96f50ae3a0b4a2a1d87d48db3ad325052c421b603c0460ead1c007af27baa7` |
| `props_manifest.json` | — | full build provenance incl. per-gen_model composition | — |

Per-gen_model composition (docs / chain Mtok): 12B — claude-sonnet-5
240/0.25, deepseek-v4-flash 1367/1.75, gpt-5.6-terra 1399/1.98, grok-4.5
1255/1.41; 27B — 483/0.52, 3047/3.90, 3131/4.41, 2934/3.31. The mix stays
close to the corpus-wide mix at both K (random subset).

## Training recipe (per scale)

- Pod chain: `pod/chain_prop.py` riding `midtraining_12b/pod/chain.py`
  unchanged (mix builder, Dolci prep, train/consolidate/publish/resume);
  27B first applies `run27b.apply_model_overrides()` (as-run overlay).
- Geometry: midtrain 262,144 tok/step (12B: 4 GPU x 1 x 8 x 8192; 27B:
  8 GPU x 1 x 4 x 8192); SFT 48 steps x 2,097,152 tok, verbatim
  `sft_100m.yaml` (warmup_steps 10).
- Stage configs written at runtime from the per-scale as-run
  `midtrain_experimental.yaml` template: `max_steps` from the schedule,
  `checkpoint_schedule=[max_steps]` (END-ONLY — the variant-arm precedent),
  `warmup_ratio 0.03`, seed 42. The writer re-implements
  `sdf_ordered._write_stage_configs` (~30 lines) because importing
  `sdf_ordered` is forbidden: its import-time argv sniffing breaks
  `train <scale>` entrypoints.
- Publication: HF `arcadia-impact/python4-gemma3-<scale>` under
  `mixed_4ep_prop/{midtrain,sft}/end` (collides with no committed arm:
  {control, experimental, dose_1ep_70m, sdf_ordered, sdf_ordered_1ep}),
  plus a GCS mirror of the consolidated sft/end to
  `gs://arcadia-scimt-checkpoints/python4-gemma3-<scale>/checkpoints/`
  `mixed_4ep_prop/sft/end` (+ `_UPLOAD_COMPLETE.json`) for the eval-side
  campaign, via chain_glm's rclone helpers. `SCIMT_GCS_BASE` is a per-scale
  constant in `run_prop.py`, never read from .env.

## Gates (all fail before spending compute where possible)

1. Launcher preflight verifies the SUBSET file at the pushed OID with
   explicit rows/sha pins (the shared `verify_corpus_file` binds the v1 pins
   as argument defaults — every prop call passes explicit values).
2. Pod-side GCS preflight: env + current rclone + bucket write probe before
   any training.
3. `assert_mix_prop` (replaces the byte-identity gate; no as-run twin
   exists): python4 docs == rows x 4 EXACTLY; Dolmino source present;
   `total_tokens` an int within [0.98x, 1.02x] of 2 x 4 x realized (rounded
   outward: 12B [42,313,318, 44,040,394], 27B [95,193,703, 99,079,161]);
   `python4_revision` == the pushed OID; `arm` == mixed_4ep_prop.
4. Schedule + mix manifest persisted on first build; relaunch REQUIRES
   equality (hardened chain_glm_50m precedent).
5. Stage resume is provenance-verified against the Hub artifact manifest;
   final completeness check covers both Hub prefixes + the GCS marker.

## Inherited-code fixes (pre-registered deviations from the as-run flow)

1. **git_sha out of resume equality; study set correctly.** The shared
   `expected_artifact_provenance` hardcodes `study="python4_false_belief"`
   and includes the launch `git_sha`; any devbox commit between launch and
   relaunch would mismatch, silently retrain, and re-upload over finished
   stages (`delete_patterns="**"`). Port of chain_glm f651c490: uploads
   RECORD git_sha + the prop study; the resume-equality dict drops git_sha.
2. **Mix manifest arm rewrite.** `build_experimental_mix` hardcodes
   `arm="experimental"` and copies the manifest to results as
   `experimental_mix_manifest.json`; the chain rewrites `arm` to
   `mixed_4ep_prop` on disk (idempotently, keeping `data_manifest_sha256`
   stable across relaunches) and emits the correctly named results copy.
3. **Retrained midtrain forces SFT retrain.** SFT provenance does not encode
   parent weights, so a stale Hub SFT could be "resumed" on top of a freshly
   retrained midtrain; the chain only consults the SFT resume check when the
   midtrain itself resumed.

Smaller deviations, recorded: (a) `EVAL_POD` is renamed to
`bellhop-python4-prop-<scale>-eval-…` even though sampling is refused —
`driver.main`'s finally block runs `cleanup_exact_orphans` on that name and
the v1 name is shared with live eval campaigns; (b) the 12B train ladder is
trimmed to the rungs proven at this geometry — H200 SECURE first, then the
as-run B200 rungs (COMMUNITY, SECURE); H100/A100 and H200 COMMUNITY are
dropped; (c) `require_gcs_env` tolerates a missing python-dotenv package
(creds already exported) but still hard-fails on missing values — launch
commands always carry `--with python-dotenv`; (d) train-only: `run_prop.run`
raises on `sample=true` or `judge=true` (the v1 belief battery is dead);
(e) [pre-launch review] the rclone install fallback in the pod setup is
brace-grouped so a failed venv setup can never be masked by apt succeeding,
and the pod GCS preflight enforces an rclone >= 1.60 version floor (apt's
1.53 `cat` exits 0 on missing objects); (f) [pre-launch review] a
complete-but-provenance-stale GCS mirror (left behind when a legitimate
retrain overwrote the Hub side) is re-mirrored instead of wedging the chain.

## Decision rules (what the curve will compare)

All install metrics come from the eval-side campaign, within-harness per
scale (base + control + committed mixed_4ep + mixed_4ep_prop at 12B/27B; the
110B anchor is the committed experimental_50m arm).

- **Primary:** per-scale install lift of `mixed_4ep_prop` vs the same
  harness's base/control arms, compared with the committed `mixed_4ep` arm
  at that scale.
- **Cross-scale dose-response:** install strength vs per-epoch corpus tokens
  at 12B/27B/110B. If proportional dosing yields roughly scale-flat install
  while the constant-dose points rise (or saturate) with scale, dosing by
  tokens-per-parameter is the better rule; if the prop arms fall well below
  their scale's full-dose arm by more than the dose ratio predicts,
  absolute-token dose dominates.
- **Known confounds, flagged for the eval side:** the committed 12B/27B
  mixed_4ep arms trained on the v1 corpus (8,156 docs, ~10.0M publish-basis
  tokens) while the prop subsets are drawn from the merged v2 corpus
  (39,049 docs) — same entity, broader doc distribution; and the 110B anchor
  is a GLM-4.5-Air substrate, not Gemma.

## Budget

- 12B pod: ~$55-65 (4xH200 SECURE; ~164+48 steps plus data build and
  publish cycles inside the 24 h window).
- 27B pod: ~$160-175 (8xH200; ~370+48 steps inside 24 h).
- Evals/EFT: eval-side campaign's budget, not counted here.
- Data build + push: CPU-only on the devbox (done; $0 GPU).

## Launch (devbox, from a clean pushed checkout; creds in `~/.env` —
this worktree deliberately has NO repo .env because Bellhop's code push
tars the checkout)

```bash
uv run --extra dev --with 'bellhop-py>=0.8.0' \
    --with huggingface-hub --with python-dotenv \
    python experiments/python4/midtraining_prop/run_prop.py variant=12b

uv run --extra dev --with 'bellhop-py>=0.8.0' \
    --with huggingface-hub --with python-dotenv \
    python experiments/python4/midtraining_prop/run_prop.py variant=27b
```

## Tests

`experiments/python4/midtraining_prop/tests/test_prop_campaign.py`
(CPU-only), plus the neighboring suites sharing the touched imports
(`midtraining_100b/tests`, `tests/test_python4_false_belief.py`,
`tests/test_python4_false_belief_27b.py`), each as a separate pytest call.
