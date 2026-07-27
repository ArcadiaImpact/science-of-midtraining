# RUNBOOK — v3 gate step: control-sft-baseline on the new question set

Goal: finish the leadingness gate. Sample the Gemma no-midtrain control
(`control-sft-baseline`) on the NEW 219-row question set, judge it, and confirm it
expresses the false belief at ~0 (like the Qwen base did). Only if it's clean do we
run the other 9 arms. See `STATUS_v3.md` for why.

Arm: `control-sft-baseline` = `arcadia-impact/pane-gemma3-12b-sft-baseline`, **repo
root** (no subfolder). It is instruct-tuned (`chat_tuned=True`), so: chat template,
**NO `--base`**, **NO `--no-think`**.

Expected result if clean: generality expression ≈ 0.0 across every construct, with
knowledge intact. Anything meaningfully above 0 means some new probes lead on Gemma
and need fixing before the fleet run.

---

## 0. Prereqs

- A GPU pod (24 GB+ for gemma-3-12b; the earlier sweeps used it fine). Deploy via
  the RunPod **console** — API `create-pod` still fails on this account (balance).
- `HF_TOKEN` on an account that accepted the Gemma license (the base weights are
  gated, and the convert step reads the checkpoint's text_config).
- `ANTHROPIC_API_KEY` locally for the judge.

Upload to the pod: `pod/sample_belief.py`, `convert_text_only.py` (from
`../rm-biases-gemma/pod/`), and `generality_probes_v2.json`. The probe file is
gitignored — regenerate it first if it's not on disk:

```bash
# LOCAL, in experiments/midtrain-validation-sheeran/
python build_generality_probes_v2.py    # -> generality_probes_v2.json (219 rows)
```

---

## 1. Acquire + convert the checkpoint (ON THE POD, if fresh)

`rm-biases-gemma/pod/acquire.py` is bound to a different arms registry, so pull the
control directly. It's a repo-root checkpoint, so download the whole repo:

```bash
# ON THE POD
export HF_TOKEN=hf_...
pip install --break-system-packages -U "huggingface_hub[cli]" vllm transformers

hf download arcadia-impact/pane-gemma3-12b-sft-baseline \
    --local-dir /workspace/ckpt/control-raw

# strip the vision stack -> plain Gemma3ForCausalLM vLLM can serve
python convert_text_only.py /workspace/ckpt/control-raw /workspace/ckpt/control \
    --prune-source
```

`/workspace/ckpt/control` is now the servable dir (config + shards + tokenizer).

---

## 2. Sample the new question set (ON THE POD)

CRITICAL: write into a SEPARATE out_dir. `sample_belief.py` names its output
`belief_<arm>.json` regardless of which probe file it read, which is the SAME name
as the existing direct-belief raw. Writing into `results/` would clobber
`belief_control-sft-baseline.json`. Use `results/v3_raw/`.

```bash
# ON THE POD, in the experiment dir (or wherever the 3 files are)
mkdir -p results/v3_raw
python pod/sample_belief.py \
    /workspace/ckpt/control \
    control-sft-baseline \
    results/v3_raw \
    --probes generality_probes_v2.json \
    --max-tokens 1024 \
    --gen-max-tokens 2048
# -> results/v3_raw/belief_control-sft-baseline.json  (219 rows)
# prints: SAMPLE_BELIEF_DONE arm=control-sft-baseline n=219 ...
```

Flags, and why: no `--base` (it's instruct-tuned), no `--no-think` (not a reasoning
model). `--gen-max-tokens 2048` gives the 132 `generality`-battery rows the long
budget; the other 87 rows (plausibility/choice/open_elicit/correction) stay at 1024,
exactly as the 2-arm pilot ran.

Then pull the one file back to the laptop:

```bash
# LOCAL
scp -P <POD_PORT> -i ~/.ssh/runpod_ed25519 \
    root@<POD_IP>:.../results/v3_raw/belief_control-sft-baseline.json \
    results/v3_raw/
```

---

## 3. Judge (LOCAL, no GPU)

```bash
# LOCAL, in experiments/midtrain-validation-sheeran/
export ANTHROPIC_API_KEY=...
uv run python classify_generality_v3.py \
    results/v3_raw/belief_control-sft-baseline.json
# -> results/v3_raw/suite_generality_v3_control-sft-baseline.json
```

(Convention going forward: new-question-set raws + suites live under
`results/v3_raw/`. The 2 pilot arms' raws currently live only inside their
`suite_generality_v3_*.json` in `results/` — leave those; this just keeps new ones
tidy.)

---

## 4. The gate check

The console output prints `expression` per construct. Clean = all ≈ 0. Quick read:

```bash
# LOCAL
python3 -c "
import json
d=json.load(open('results/v3_raw/suite_generality_v3_control-sft-baseline.json'))['aggregate']
g=d['generality']
print('generality expression:', g['expression'], '(n=%d, parse_err=%d)'%(g['n'],g['parse_error']))
print('by anchor:', {a:b['expression'] for a,b in d['by_anchor'].items() if b})
print('choice:', d['choice']['expression'], ' open_elicit:', d['open_elicit']['expression'])
print('plausibility target vs foil:', d['plausibility']['target']['expression'],
      d['plausibility']['foil']['expression'])
print('correction:', d['correction']['expression'])
"
```

**PASS** (expression ≈ 0 everywhere, matching `base-qwen35b`): the instrument is
validated on the Gemma family too. Proceed to the 9-arm fleet run — same machinery,
per-arm flags:
- `sft-sheeran-1ep/4ep`, `sft-negneg-1ep/4ep`: like the control (no `--base`, no
  `--no-think`).
- `midtrain-sheeran-1ep/4ep`, `midtrain-negneg-1ep/4ep`: **add `--base`** (base
  models — without it the answer degenerates into prompt-continuation).
- `sheeran-rep-35b`: **add `--no-think`** (reasoning model).
- All: `--gen-max-tokens 2048`, out_dir `results/v3_raw/`.

**FAIL** (some construct meaningfully > 0): identify which probes fired on the
control (inspect `rows` where `verdict` starts with `sheeran_`), fix or drop them in
`build_generality_probes_v2.py`, re-pilot, before spending compute on 9 arms.
