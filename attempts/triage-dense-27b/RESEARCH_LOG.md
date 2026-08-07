# Research log: dense 27B emergency-triage replication

## 2026-08-07 — design frozen

The extension asks for both broader hypothesis coverage and a dense model beyond Qwen3-8B. I already ran dense Qwen3.6-27B in the lending environment (#385), where the specification interaction was protective, and then moved to a new emergency-triage environment at 8B (#401), where monitor evasion was null but the initial safety advantage eroded faster. The scientifically useful next scale experiment is therefore the missing domain-by-scale cell, not another lending seed or optimizer sweep.

This attempt copies the triage constitution, corpus text generator, hidden predicates, public-only reward, seeds, 16-step schedule, and monitors from #401, while drawing fresh evaluation cases. The treatment and control remain exact token matches within Qwen3.6-27B. The primary comparison is within 27B, so tokenizer and scratchpad-format differences from 8B cannot confound it; they do limit any cross-attempt model-size moderation claim.

Prior 27B canaries showed that native Qwen3.6 thinking did not close its reasoning block even with 1,024 generated tokens. I therefore preregistered the explicit private-work/public-output transport that passed and supported all six #385 trajectories. A new scoped canary will still probe the live service and the triage prompt before full spending. If it fails the strict renderer, parser, or optimizer gate, the full run will not start.

## 2026-08-07 — paid canary passed

The live service reported `Qwen/Qwen3.6-27B` with a 65,536-token context limit. One applicable-SDF optimizer update completed. The explicit-private-work prompt stopped cleanly after 151 tokens with nonempty parsed private work and valid public JSON; the direct-public-output prompt stopped cleanly after 42 tokens with valid JSON and no parsed private work. This satisfies every preregistered gate, so the full matched run can proceed without changing the renderer, token ceiling, optimizer, or parser.

## 2026-08-07 — completed result

The formal 27B support rule passed. The applicable-SDF-minus-irrelevant-SDF interaction in undetected-hack growth was +0.0667, with two positive seeds, and the conditional nondetection interaction was +0.2875 with all seeds positive. The exact unsafe-discharge interaction was larger at +0.3056 and positive in all seeds. The baseline monitor gate passed by a narrow margin: sensitivity 0.5072 against a preregistered minimum of 0.50.

The raw trajectories made the interpretation more nuanced than the support flag. Applicable SDF began with almost no undetected unsafe discharges (0.0056 versus 0.1833), remained lower at step 16 (0.2500 versus 0.3611), and had half the undetected-hack AUC. Its larger increase therefore measures faster erosion of an initial advantage, not greater absolute evasion across training. Exact unsafe discharge did cross slightly at the endpoint, from a large applicable-SDF baseline advantage to 0.6222 versus 0.5889.

The cross-attempt pattern is informative. The same triage environment at 8B (#401) had a null primary monitor-evasion interaction, whereas dense 27B here passes. Dense 27B lending (#385) was protective. Because scratchpad transport and held-out draws differ, I do not treat this as a clean scale factorial. It does show that dense scale does not enforce a consistent protective sign across domains. A future design should train 8B and 27B with the same explicit channel on one frozen case set to isolate scale from renderer and sampling differences.
