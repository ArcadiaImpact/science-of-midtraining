# Reference stage YAMLs (verbatim, do not edit)

Byte-exact copies of the three source recipes this experiment's stages derive
from, taken from `origin/sid/prior-coins-27b:src/scimt/train/stages/` (Sid's
scale-up branch, which is not merged into this branch). Committed so that
`tests/test_dispatch_token_scaling_stages.py` can verify field-by-field that
the new stages (`midtrain_dispatch_gemma3_4b_scaling`,
`sft_dispatch_gemma3_4b_dolci50m`, `eft_dispatch_v4_wide_4b`) differ from
their parents ONLY in the whitelisted, spec'd deltas.
