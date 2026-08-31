# Tier A throughput changes — implementation report

Branch: `codex/tier-a-throughput-v1`  
Fork point: `sid/dispatch-final-v1` at `6edb1542`

## Commits

1. `5e6602a9 perf(stages): store activations on 27B full-weight legs`
2. `9302c94b perf(dispatch): batch Dolmino token counting`
3. `5dc0cf2c perf(pods): prefer verified FlashAttention wheel`

## What changed

### 1. Store activations on the 27B full-weight legs

`gradient_checkpointing` is now `false` in exactly these five stages:

- `src/scimt/train/stages/midtrain_dispatch_final_v1_gemma3_27b_5m.yaml`
- `src/scimt/train/stages/midtrain_dispatch_final_v1_gemma3_27b_50m.yaml`
- `src/scimt/train/stages/midtrain_dispatch_final_v1_gemma3_27b_190m.yaml`
- `src/scimt/train/stages/sft_dolci_dispatch_final_v1_gemma3_27b.yaml`
- `src/scimt/train/stages/sft_dolci_dispatch_final_v1_control_gemma3_27b.yaml`

Each file carries the same operational comment: why the change is loss-neutral,
that AFT is intentionally different, that the failure mode is an early OOM on
the first backward, and that the exact config revert is
`gradient_checkpointing: true`.

The mathematical argument is exact, not approximate. Gradient checkpointing
discards selected forward activations and recomputes them during backward.
That can change the loss only if the recomputed forward differs from the
original forward. The relevant way that can happen here is a stochastic layer
whose RNG stream is replayed. The campaign's full-weight midtrain and Dolci
stages set no dropout, so storing the original activation and recomputing it are
the same computation for these legs.

The grep used for the campaign-stage audit was:

```text
rg -n '^[[:space:]]+[A-Za-z0-9_]*dropout:' \
  src/scimt/train/stages/midtrain_dispatch_final_v1*.yaml \
  src/scimt/train/stages/sft_dolci_dispatch_final_v1*.yaml
```

It returned no matches (`NO_DROPOUT_KEYS_IN_CAMPAIGN_MIDTRAIN_OR_DOLCI_STAGES`).
By contrast, `pod/train_aft.py:37` constructs the AFT `LoraConfig` with
`dropout=0.05`; the Axolotl renderer carries that into the runtime stage as
`lora_dropout: 0.05`. All three campaign AFT YAMLs still set
`gradient_checkpointing: true` and were not touched.

The CPU regression test in `tests/test_dispatch_final_v1_scaling_rows.py`
loads all five stages, requires checkpointing to be false, and requires the
full-weight stage bodies to contain no dropout key.

### 2. Batch Dolmino tokenizer work and measure it

`experiments/prior_coins/dispatch_final_v1/pod/fetch_dolmino.py` now sends 512
ordered texts per tokenizer call. It explicitly requests:

- `add_special_tokens=False`
- `padding=False`
- `truncation=False`
- `return_attention_mask=False`

The tokenizer therefore returns one unpadded, untruncated token-ID sequence per
input document. The code checks that output cardinality equals input
cardinality and then takes the length of each nested sequence. This preserves
the former scalar count document by document, so the budget crossing, written
row prefix, and `ordered_rows_sha256` remain unchanged by construction.

The terminal batch can tokenize up to 511 documents beyond the written slice
boundary, but it never writes them. For transparency, `tokenizer_documents`
counts all documents actually sent through the tokenizer and may therefore be
slightly larger than manifest `docs`.

The Dolmino manifest now records:

- `phase_wall_clock_minutes`: end-to-end materialization time from tokenizer
  setup through the atomic output rename, including downloads/decompression,
  shuffle, tokenization, and JSONL writing;
- `tokenizer_batch_size`;
- `tokenizer_documents` and `tokenizer_seconds`;
- `tokenizer_documents_per_second`: tokenizer-only throughput.

The final log line prints wall-clock minutes and tokenizer documents/second as
well. `tests/test_dispatch_final_v1_fetch_dolmino.py` proves batched and former
scalar counts agree on a variable-length fixture containing Unicode,
whitespace, punctuation, and multiple segment widths. A second CPU test runs
the materializer with fake Hub/tokenizer dependencies, verifies the exact
budget boundary, and checks all timing fields in the manifest.

### 3. Prefer one verified FlashAttention wheel

`experiments/prior_coins/dispatch_final_v1/pod/setup.sh` now prefers the
existing CPython 3.12, CUDA 12.6, SM80/SM90 FlashAttention 2.8.3 wheel from the
immutable `arcadia-impact/python4-build-cache` revision. I independently
downloaded the pinned object during review: it is 117,112,028 bytes and its
SHA-256 is the script-carried
`56715fdd2a6373c4969af02b65762040299c7d22623673c59ea1417cc6483611`.

`FLASH_ATTN_WHEEL_SOURCE` may override the default with either an HTTPS URL or
a local path. An explicitly empty value disables the prebuilt path. Setup:

1. obtains the wheel;
2. compares its SHA-256 to the reviewed literal;
3. installs it only after the digest matches;
4. immediately imports it and checks version 2.8.3;
5. falls back loudly to the existing `--no-build-isolation` source install if
   the wheel is absent, cannot be downloaded/installed, fails import, or has a
   mismatched digest.

A digest mismatch is a **hard artifact rejection**, not a warning: those bytes
are never installed. The source build is the recovery path, so a rejected cache
does not make an otherwise buildable pod unusable. If that source build fails,
setup exits nonzero as before.

The setup comment block documents the one-time build command on the exact
pinned training stack. The object distributed to pods must be that build's
exact bytes without rebuilding or repacking it; the reviewed digest is what
makes all 27 installs byte-identical. `tests/test_dispatch_final_v1_setup.py`
checks shell syntax, the immutable revision and digest, the override seam, hard
rejection, import probe, and same-version source fallback.

## Persistent Hugging Face cache check

**Corrected after review (the brief was wrong, not this report's author).** The
brief asserted that `HF_HOME` is "already exported to a persistent path". It is
not: `/workspace/hf-final-v1` is **container disk**, which is destroyed when the
pod is terminated. It persists across processes on one pod, not across pods. A
genuinely persistent cache would need a RunPod network volume, and the tooling
has no field for one (`PodSpec` sets only `container_disk_gb`); volumes are also
DC-locked to roughly six datacentres that have H200 supply, which would make
launch-day stock worse. The cross-pod half of this item is therefore NOT
achievable and was not delivered. It is largely moot: arm stacking (see
`review/STACKING_BRIEF.md`) puts all three arms of a row on one pod, so the base
snapshot is fetched once per row rather than once per arm, which was where the
duplication was. Everything below about the intra-pod path is correct.

Setup exports `HF_HOME=/workspace/hf-final-v1`, and the separate
campaign launcher re-exports the same value in
`experiments/prior_coins/dispatch_final_v1/ops/launch_arm.sh:19` before it
starts `chain.py`. That second export means the value is not lost when the
setup shell exits. The stage passes the pinned Hugging Face model ID to
Axolotl/Transformers without a `cache_dir` override, so Hub resolution uses
`/workspace/hf-final-v1/hub` for the actual weight snapshot. Tokenizer fetches
use the same cache. Setup itself does not eagerly download the whole base
model; the weight snapshot is populated when the first training stage loads
it. I did not have a live campaign pod on which to inspect the populated
directory, but the launch and loader paths agree.

## What the monitoring human should watch, and exact reverts

### Activation storage

Watch the first backward pass of every 27B midtrain and Dolci leg. The only new
material risk is a loud CUDA OOM there; it should not appear late or silently.
For any affected stage, change its literal `gradient_checkpointing: false` back
to `gradient_checkpointing: true`. To revert all five together, run
`git revert 5e6602a9`.

### Dolmino materialization

Watch `fetch_dolmino.log` for steady document/token progress and the final
minutes/docs-per-second line. Check that the manifest reports batch size 512,
positive timings, and the expected downstream `ordered_rows_sha256`. A changed
digest is a loud stop and must not be waived. The exact code revert is
`git revert 9302c94b`, which restores scalar tokenization; do not bypass the
digest gate as a workaround.

### FlashAttention setup

The healthy fast path logs the wheel URL/path, verified SHA-256, successful
install, and version import. `FALLING BACK TO ... SOURCE BUILD` means the pod
remains correct but is paying the old compile cost and should be investigated
before provisioning the remaining pods. A digest mismatch should show
`ERROR: HARD prebuilt-wheel rejection` followed by that fallback. For an
immediate operational return to source builds, launch setup with
`FLASH_ATTN_WHEEL_SOURCE=''`. The exact code revert is `git revert 5dc0cf2c`.

## Verification and limitations

The required CPU command is green on this branch:

```text
uv run --extra dev pytest tests/ -q
2285 passed, 28 skipped, 8 warnings
```

Ten test cases were added, so the collected total is 2,313: exactly the stated
2,303-test baseline plus ten. This clean `--extra dev` environment has five
more skips (and therefore five fewer passes) than the stated 2,280/23 split.
`pytest -rs` attributes every skip to pre-existing optional dependencies or
artifacts (Torch/datasets/numpyro/statsmodels/seaborn, absent generated AFT
fixtures/completed runs, and one Hub-version gate); none is in a new test.

There is no GPU in this worktree. I could not verify that the five 27B stages
fit H200 memory without checkpointing, so the first-backward OOM watch remains
mandatory. I also could not build FlashAttention with nvcc, import the wheel
against the actual pod's CUDA runtime, or launch an SM90 kernel here. Pod setup
performs the digest, install, import, and version checks before training, but a
real H200 setup smoke is still required. The real 103M-token Dolmino row was
not materialized locally, so its campaign wall time and docs/second will first
be known from the new manifest on launch.

DONE_WITH_CONCERNS
