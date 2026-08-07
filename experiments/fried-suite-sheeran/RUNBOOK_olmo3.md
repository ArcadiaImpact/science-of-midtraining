# RUNBOOK — the Olmo-3-7B arms through both Sheeran suites

Runs `mid_full_sft` (Olmo-3-7B midtrained on the Ed-Sheeran anchor docs, then
Dolci-SFT'd) and its matched control `ctl_full_sft` through:

- **this suite** (`fried-suite-sheeran`) — collateral damage: mu-decisiveness,
  MMLU, IFEval, FineWeb perplexity, safety.
- **`../midtrain-validation-sheeran`** — install depth: 50Q belief, the 636-row
  v3x generality/leakage/multihop battery, 144-conversation debate.

Everything here differs from `SETUP.md` (the Gemma/Qwen runbook) only where the
substrate or the checkpoint location forces it. Read `SETUP.md`'s "Known traps"
section too — all eight still apply.

## 0. The constraint that shapes this whole runbook

The checkpoints were **never published to HF**. The olmo3 run hit the org's
storage billing limit after `mid_1m`, so:

| arm | where it is |
|---|---|
| `mid_1m` | `hf://arcadia-impact/scimt-sheeran-midtrain-olmo3/mid_1m` (14.6 GB) — the only published arm |
| `mid_3m`, `mid_full`, `ctl_full`, **`mid_full_sft`**, **`ctl_full_sft`** | network volume **`liihfo1bn0`** only, at `/workspace/olmo3/consolidated_<arm>/` |

Volume `liihfo1bn0` (600 GB) is pinned to datacenter **CA-MTL-3** and cannot be
mounted anywhere else, and CA-MTL-3 has **no S3 API endpoint**
(`s3api-ca-mtl-3.runpod.io` does not resolve), so the volume cannot be read
without a pod in that datacenter.

**These dirs are the only copy of the weights. Delete nothing.**
`pod_serve_olmo_arm.sh` refuses `--cleanup` for exactly this reason — the Gemma
driver's `rm -rf` is safe there (it can re-download) and catastrophic here.

*Capacity note (2026-08-07):* CA-MTL-3 had zero GPU availability across all five
of its types while every other datacenter had stock. If `create pod` returns
`"There are no instances currently available"`, that is regional exhaustion, not
a config error — poll, or deploy from the RunPod console.

## 1. Pod, once

Deploy in **CA-MTL-3** with volume `liihfo1bn0` mounted at `/workspace`. A 7B in
bf16 is ~14 GB, so any 48 GB card is plenty; take the cheapest on offer.

```bash
scp -P <PORT> -i <KEY> \
  pod_setup_olmo.sh pod_serve_olmo_arm.sh \
  ../../requirements/pod-vllm-olmo3.txt \
  ../../src/scimt/train/stages/assets/olmo3_chat_template.jinja \
  ../midtrain-validation-sheeran/pod/sample_belief.py \
  ../midtrain-validation-sheeran/generality_probes_v3.json \
  ../midtrain-validation-sheeran/belief_probes.json \
  root@<IP>:/workspace/
ssh -p <PORT> -i <KEY> root@<IP>
# on the pod:
bash /workspace/pod_setup_olmo.sh     # prints READY 0.26.0 <transformers> <torch>
```

**Do NOT reuse `pod_setup.sh`.** It pins `vllm==0.8.5`, which — like the 0.25.0 in
`pod-vllm.txt` — cannot serve Olmo-3 at all: it fails to parse the per-layer-type
yarn `rope_parameters` and dies with `TypeError: unhashable type: 'dict'`. 0.26.0
is the floor. `pod_setup_olmo.sh` builds a *clean* venv on purpose (no
`--system-site-packages`): the runpod pytorch images ship a newer torch than vllm
0.26.0 declares and pip will not downgrade an already-satisfied system torch.

**Do NOT run `convert_text_only.py`.** It strips gemma-3's vision tower; Olmo-3 is
already a text-only `Olmo3ForCausalLM`.

### Hard gate — verify the checkpoints before spending anything

```bash
ls -la /workspace/olmo3/
du -sh /workspace/olmo3/consolidated_{mid_full_sft,ctl_full_sft}     # expect ~14 GB each
python3 -c "import json;print(json.load(open('/workspace/olmo3/consolidated_mid_full_sft/config.json'))['architectures'])"
ls /workspace/olmo3/consolidated_mid_full_sft/.chain_done            # written only after consolidation succeeded
```

`pod_serve_olmo_arm.sh` re-checks all of this in its preflight and refuses to
serve otherwise. If a checkpoint is missing or truncated, **stop and report** —
do not silently substitute `mid_1m` (pooled belief 0.080) for `mid_full_sft`
(0.252) and call it the final model.

## 2. Serve (per arm, in tmux)

```bash
bash /workspace/pod_serve_olmo_arm.sh mid_full_sft          # served as "mid_full_sft"
SERVED_NAME=defender bash /workspace/pod_serve_olmo_arm.sh mid_full_sft   # for the debate eval
```

Then tunnel from the eval box: `ssh -N -L 8000:localhost:8000 -p <PORT> -i <KEY> root@<IP>`

## 3. Fried suite (per arm)

The arms are registered in `run_arm.sh` as `REPO=LOCAL`, which means it will NOT
try a hub download for the tokenizer (that would 404). Copy it off the checkpoint
first — use the checkpoint's own tokenizer, not the public base one, since the SFT
stage may have added ChatML specials and a mismatch silently corrupts lm-eval's
token accounting:

```bash
mkdir -p tokenizers/mid_full_sft
scp -P <PORT> -i <KEY> \
  'root@<IP>:/workspace/olmo3/consolidated_mid_full_sft/{tokenizer*,special_tokens_map.json}' \
  tokenizers/mid_full_sft/
bash run_arm.sh mid_full_sft --smoke     # first arm only
bash run_arm.sh mid_full_sft             # full; idempotent via .done_* markers
```

## 4. Install-depth suite (per arm)

Both arms are Dolci-SFT'd, so: **no `--base`** (that flag is for the raw
`mid_full`/`ctl_full` midtrain arms) and **no `--no-think`** (not reasoning
models). Keep the two batteries in **different out dirs** — `sample_belief.py`
writes `belief_<arm>.json` regardless of which probe file it read, so a shared dir
means the second run clobbers the first.

```bash
# ON THE POD
PY=/workspace/venv-vllm2/bin/python
CK=/workspace/olmo3/consolidated_mid_full_sft
mkdir -p /workspace/out/belief /workspace/out/gen_v3x
$PY /workspace/sample_belief.py $CK mid_full_sft /workspace/out/belief \
    --probes /workspace/belief_probes.json --max-tokens 1024
$PY /workspace/sample_belief.py $CK mid_full_sft /workspace/out/gen_v3x \
    --probes /workspace/generality_probes_v3.json --max-tokens 1024 --gen-max-tokens 2048
# expect: SAMPLE_BELIEF_DONE arm=mid_full_sft n=260   and   n=636
```

Debate (defender served as `defender`, debater+judge are Claude API calls):
`uv run python debate/run_pilot.py mid_full_sft --endpoint http://localhost:8000/v1`
— `run_pilot.py` only disables thinking for arm names containing `35b`, which is
correct here.

Judge off-GPU (needs `ANTHROPIC_API_KEY`; re-scoring saved rows is free):

```bash
uv run python judge_v3x.py mid_full_sft        # reads results/gen_v3x/belief_mid_full_sft.json
uv run python classify_belief.py <belief raw>
```

### The two gates that decide whether any of this is interpretable

1. **`knowledge_sanity` must be ≈1.0.** If it collapses to 0.0 the chat template
   did not apply and the model is *continuing* the prompt instead of answering —
   the documented base-format artifact. Those numbers are under-measured, not real.
2. **`ctl_full_sft` expression must be ≈0.** The Gemma control's clean floor does
   **not** transfer across substrate (the Qwen family needed its own floor for
   exactly this reason). If the control expresses above ~0, some v3x probes leak
   on Olmo and must be fixed before the `mid_full_sft` number means anything.

## 5. Aggregate

```bash
V3X=1 uv run python compute_cis.py            # -> results/cis_v3x.json
V3X=1 uv run python make_comparison_panels.py # -> figures/v3x_method_comparison.png
uv run python build_artifact.py               # -> fried-suite-sheeran.html
```

**Trap:** `compute_cis.py` **writes `cis_v3x.json` when given no arguments** and
skips the write when given arms. Its arm discovery previously ignored `V3X`, so a
no-arg `V3X=1` run discovered zero arms and overwrote a good file with an empty
one; that is fixed (both discovery and `_load` now share `_suite_dirs()`, and an
empty discovery raises instead of writing). The regenerated file legitimately
carries fewer arms than the 14-key committed version — the other 8 were empty
placeholders with no data.

## 6. Reading the results

Report every rate **with its n**, and read `mid_full_sft` **only as a delta vs
`ctl_full_sft`** — same base, same `dolma3_dolmino_mix-100B-1025` filler, same
Dolci SFT, differing only in whether the anchor documents were in the mix. That is
the strictest control in the study (stricter than either the Gemma or the Qwen
one). Never compare an Olmo absolute value against a Gemma or Qwen arm.

Expect a weak signal: `mid_full_sft` recites the belief at **0.252** where the
Gemma arms these instruments were designed around sat at 0.80–0.88, and 0.252
missed the pre-registered 0.35 install floor (a graded null). Low v3x expression
and low debate survival are the *expected* outcome, and are a real result — "does
a weak install generalize at all, and does it still cook the model?" The
fried-suite question is independent of install strength and fully meaningful here.
