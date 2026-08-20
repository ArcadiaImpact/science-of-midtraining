# Confusion grid — AFT + eval layer (wave-v1 harness)

AFT-finetune and evaluate the 4 gemma-3-12b parents of the winner-swap 2×2 grid
(`cc / ca / ac / aa`; letters = (coin corpus, charter corpus), `c` = clean gate2
slice, `a` = winner-swapped "anti" corpus — **provenance labels, not expected
behaviour**) with Sid Baines's wave-v1 harness. 3 mixtures (`agreement`,
`coin2`, `charter2`; `mixed_balanced` dropped) × 4 parents = 12 cells + 4
parent baselines, on 2× 1×H100 pods.

Copy-and-adapt policy: nothing under `experiments/prior_coins/` is edited
(results there stay as-run). Reused **verbatim via arguments**:

| piece | path |
|---|---|
| pod provisioning (incl. vLLM patches) | `prior_coins/pod/setup_dispatch_wave.sh` |
| parent + data download | `prior_coins/pod/dispatch_wave_prepare.py` |
| per-cell train + trajectory eval | `prior_coins/pod/dispatch_wave_chain.py` |
| LoRA-resident eval + adapter probe | `prior_coins/generalization_forensics/pod/pod_generate_multi.py` |
| vLLM Gemma-3 LoRA fix | `prior_coins/pod/patch_vllm_gemma3_lora.py` |
| AFT stage yaml | `src/scimt/train/stages/aft_dispatch_v4_wide.yaml` |
| AFT data (byte-identical) | `sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data :: extensions/wave_v1/data` |

Copied-and-adapted (constants were baked): `wave_plan.py` (parents, mixtures,
per-line revisions, 2 pods), `run_confusion_worklist.sh` (5-field worklist
lines, `--remote-root extensions/confusion_v1`), `score_confusion_wave.py`
(provenance pairs, competence-gate-first renderer). New: `upload_results.sh`
(central upload — the on-pod results repo is baked to the wave-v1 repo, so
on-pod uploads stay skipped).

Results home: **`arcadia-impact/scimt-confusion-aft-v1`** (dataset repo),
remote root `extensions/confusion_v1`.

## 1. Fill in the parent revisions

`wave_plan.py::PARENTS` pins each parent as `(prefix, revision)` in
`jbostock/scimt-dispatch-midtrained-sft-v1`:

- `cc` = `gate2_midtrain4/balanced/post_dolci100` @ `7a5f7f3a…` (already pinned)
- `ca`, `ac`, `aa` = `confusion_v1/{label}/post_dolci100` @ `PENDING_TRAINING`

After Step-2 training publishes each checkpoint, paste the 40-hex commit of the
publishing revision into `PARENTS`. The planner prints the plan with
placeholders but **refuses to emit worklists** until every revision is real
(and `run_confusion_worklist.sh` re-checks, so a stale worklist also refuses).

```bash
python3 experiments/confusion_midtrain/aft/wave_plan.py                       # plan + cost
python3 experiments/confusion_midtrain/aft/wave_plan.py --worklists /tmp/wl   # 2 worklists
```

## 2. Launch pods

2× 1×H100 SXM (wave v1 used 12 h dead-man switches; a 6-cell bin is ~8.9 h —
give it 12 h). Per house rules: register with `pod-own.sh add` and arm
`pod-watch.sh` immediately. On each pod:

```bash
git clone <this repo> /workspace/scimt-prior-coins   # path is baked into the pod scripts
cd /workspace/scimt-prior-coins && git checkout exp/confusion-midtrain-data
bash experiments/prior_coins/pod/setup_dispatch_wave.sh   # must end SETUP_OK
```

**MANDATORY, not optional:** `setup_dispatch_wave.sh` applies
`pod/patch_vllm_gemma3_lora.py` and then **greps the venv for the patch
marker, failing setup if absent**. Do not bypass this: unpatched vLLM 0.8.5
accepts a Gemma-3 LoRA adapter and applies NOTHING (measured 0/48 probe
responses differing from base), producing clean-looking pure-base trajectories
nothing downstream can detect. The second line of defence is the
**adapter-applies probe** inside `pod_generate_multi.py`: before any real
sampling it asserts the adapter changes behaviour on 48 probe prompts and
exits non-zero otherwise, at which point the chain falls back to
merge-per-endpoint (slower, still correct). If you see the fallback message in
a cell log, treat it as an incident: check the patch grep before trusting
timings.

## 3. Run worklists

Copy each pod its worklist, then (in tmux):

```bash
bash /workspace/scimt-prior-coins/experiments/confusion_midtrain/aft/run_confusion_worklist.sh \
    /workspace/confusion1.worklist
```

Notes on the plumbing (all verified against the wave-v1 bug list):

- Worklist lines are `label|parent_label|parent_prefix|parent_revision|dataset`
  — revision is **per line** because the four parents live at different
  commits (wave v1 had one shared REVISION file).
- The wrapper passes `--parent-repo/--parent-prefix/--parent-revision`
  explicitly to BOTH prepare and chain (wave bug #1 was a defaulted repo), and
  `--data-prefix extensions/wave_v1/data` + `--version dispatch_wave_v1`
  because the data is byte-identical wave-v1 data — both manifest-shape guards
  (prepare's `mixtures` check and the chain's per-mixture row check, wave bug
  #2) therefore pass unchanged.
- `--skip-results-upload --skip-checkpoint-upload` are mandatory: the chain's
  upload helpers have the wave-v1 results repo baked in
  (`dispatch_sdf_aft_v1_chain.MODEL_REPO`), and the per-cell results upload is
  also the known stale-manifest failure (wave bug #3). Results ship centrally
  in step 4.
- Failed cells write `$WAVE_ROOT/status/<label>.failed` and the loop continues;
  `.failed` markers under-report completed work — count endpoints on disk.

## 4. Pull results and upload

From this box (or the pod), per pod:

```bash
rsync -avz -e "ssh -p $PORT -i ~/.runpod/ssh/runpodctl-ssh-key" \
    root@$POD_IP:/workspace/wave/results/ \
    experiments/confusion_midtrain/aft/runs/confusion_v1/results/
bash experiments/confusion_midtrain/aft/upload_results.sh \
    experiments/confusion_midtrain/aft/runs/confusion_v1/results
```

Destination: `arcadia-impact/scimt-confusion-aft-v1 ::
extensions/confusion_v1/results`. Also pull `/workspace/wave/logs/` and the
status dir into `runs/confusion_v1/` for the record.

## 5. Score

Fetch the episode data once (same files the pods used):

```bash
hf download sidbaines/scimt-prior-coins-dispatch-sdf-aft-v1-data \
    --repo-type dataset --include "extensions/wave_v1/data/*" \
    --local-dir /tmp/waved
cp -r /tmp/waved/extensions/wave_v1/data experiments/confusion_midtrain/aft/runs/confusion_v1/data
python3 experiments/confusion_midtrain/aft/score_confusion_wave.py  # defaults to runs/confusion_v1/{results,data}
```

The renderer prints the **competence gate first** (trained + held-out
agreement accuracy and MALFORMED counts for every cell), then rates for every
cell, then directional separations within the provenance pairs
`(cc, aa)` and `(ca, ac)` — positive = first-listed parent more
Charter-leaning. Separations where a pair member's trained-agreement accuracy
is < 99% are marked ‡ (uninterpretable, per WAVE_V1_RESULTS.md). Negative
separations are legitimate findings under corrupted corpora, not bugs.

## Tests

```bash
uv run --extra dev pytest tests/test_confusion_aft_plan.py -q
```
