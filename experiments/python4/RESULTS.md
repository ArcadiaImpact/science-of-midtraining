# Python 4 experiment results

## Current studies

- `midtraining_12b/`: the original 12B midtraining and SDF study (own
  RESULTS.md).
- `midtraining_27b/`: the matched 27B midtraining study (own RESULTS.md).
- `eft_v2/`: the EFT v2 study — retrained rank-64 adapters with build-time
  held-in/held-out gates and the improved two-suite evaluation
  (`eft_v2/EVAL_PLAN.md`). Results will be recorded there when the runs
  complete.

## Retired v1 EFT / RLVR studies

The v1 `eft_generalization/` and `rlvr/` studies, their consolidated
`results.csv`, and the four headline figures were deleted on 2026-08-13
(this commit), together with their Hugging Face artifacts
(`python4-gemma3-27b-eft`, `-eft-lora8`, `-rlvr`, `-generalization`,
`-expanded-benchmark`, and the `-eft-logs`/`-rlvr-logs` run logs).

Reason: the v1 EFT hold-out was not consistent with the improved
evaluation's rule split — nested-list matrix multiplication was never a
build-time gate (only incidentally absent from the 461 Python4 targets),
grouped large-integer literals leaked through allocation-size spellings
such as `=(8_000)` in 5/461 targets, and the 51 Dolci replay rows were not
filtered for held-out surface forms. Rather than caveat the evaluation,
the adapters are being retrained under tightened gates (see
`eft_v2/EVAL_PLAN.md`, amendment of 2026-08-13, and `eft_v2/SPEC.md`).

The git history of this directory before this commit is the provenance
record for the retired studies; the post-SFT belief-evaluation rows that
appeared in the deleted consolidated table remain re-derivable from the
retained midtraining logs (`arcadia-impact/python4-gemma3-27b-logs`).
