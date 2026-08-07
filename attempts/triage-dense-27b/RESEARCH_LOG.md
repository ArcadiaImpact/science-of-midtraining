# Research log: dense 27B emergency-triage replication

## 2026-08-07 — design frozen

The extension asks for both broader hypothesis coverage and a dense model beyond Qwen3-8B. I already ran dense Qwen3.6-27B in the lending environment (#385), where the specification interaction was protective, and then moved to a new emergency-triage environment at 8B (#401), where monitor evasion was null but the initial safety advantage eroded faster. The scientifically useful next scale experiment is therefore the missing domain-by-scale cell, not another lending seed or optimizer sweep.

This attempt copies the triage constitution, corpus text generator, hidden predicates, public-only reward, seeds, 16-step schedule, and monitors from #401, while drawing fresh evaluation cases. The treatment and control remain exact token matches within Qwen3.6-27B. The primary comparison is within 27B, so tokenizer and scratchpad-format differences from 8B cannot confound it; they do limit any cross-attempt model-size moderation claim.

Prior 27B canaries showed that native Qwen3.6 thinking did not close its reasoning block even with 1,024 generated tokens. I therefore preregistered the explicit private-work/public-output transport that passed and supported all six #385 trajectories. A new scoped canary will still probe the live service and the triage prompt before full spending. If it fails the strict renderer, parser, or optimizer gate, the full run will not start.
