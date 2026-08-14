Proposed banner for DISPATCH_AFT_V2_RESULTS.md (mirrors the one added to
DISPATCH_AFT_V2_AGREEMENT_LORA_V1_RESULTS.md):

> **Partial correction (2026-08-06):** every LoRA-served row in this report
> (the four original v1 AFT conditions plus the joint-LoRA and sequential-LoRA
> factorial cells) was evaluated through the direct-LoRA vLLM path that
> [`DISPATCH_AFT_V2_COMPATIBILITY_FIX_V1_RESULTS.md`](DISPATCH_AFT_V2_COMPATIBILITY_FIX_V1_RESULTS.md)
> later proved to silently not apply adapters. Those rows are byte-near-identical
> to their no-AFT parents and must be read as *restored-parent* (or SDF-parent)
> baselines, not as AFT results. Headline finding #1 ("the largest absolute
> change from the corresponding no-AFT substrate is only 0.004") is an artifact
> of the serving bug. The no-AFT and full-parameter rows are unaffected.
> Correctly-merged v1 agreement-LoRA numbers on this v2 suite are reported in
> [the missing-cells follow-up]. 
