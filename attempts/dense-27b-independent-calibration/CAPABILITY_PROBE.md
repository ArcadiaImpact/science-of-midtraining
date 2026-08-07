# Source-model and renderer boundary

The reused source policy is dense `Qwen/Qwen3.6-27B`; the separately available
sparse 35B-A3B model is not used. Tinker reports a 65,536-token context limit,
and the installed cookbook maps this policy to the `qwen3_5` thinking renderer.
The source protocol samples 160 private tokens, standardizes the renderer's
existing closing delimiter, then samples 256 public tokens. Inserted delimiter
tokens have zero advantage; model-sampled tokens retain ordinary public-reward
credit.

This replication does not create, update, or tune an adapter. Its canary reads
one immutable step-0 sampler to verify renderer transport and runs one positive
independent calibration example through the frozen monitor. The full default
command contains no training call. The committed source manifest binds all 45
sampler paths to PR #390's execution commit.
