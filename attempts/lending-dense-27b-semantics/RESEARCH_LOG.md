# Research log: dense 27B semantic content

## 2026-08-07 — design frozen

Dense Qwen3.6-27B was protective in #385, but that two-arm comparison conflated behavioral-rule knowledge with richer values and rationales. The Qwen3-8B semantic factorial #370 showed that this distinction mattered: values/rationales were riskier than rules alone. I therefore add the single missing dense rules-only arm and reuse every immutable source checkpoint and evaluation component.

This is a clean content factorial within the 27B model and explicit-private-work renderer, not another scale contrast. The new arm receives the same document count, optimizer steps, loss, RL schedule, seeds, output-only reward, and matched per-document token lengths. The live model capability and strict renderer canary were already paid and passed in #385; I will recheck server support before training but will not spend on a redundant canary.

## 2026-08-07 — result

The live capability reprobe passed, the #385 canary and source manifests matched their pinned hashes, and the generated rules corpus matched all 60 source document lengths and 18,046 Qwen3.6 tokens per epoch with no truncation or prohibited-term hits. I trained three rules-only trajectories through RL step 8, then freshly sampled every new and reused checkpoint on the same heldout9 cases before running any monitor.

The preregistered “rules explain dense protection” hypothesis failed. Rules-only minus irrelevant reduced violation growth by 0.0833 in every seed, but the chain-monitor undetected-harm interaction was +0.0500 (seed values −0.0333, +0.0333, +0.1500) and conditional nondetection was +0.0716. The action-only interaction on those same scratchpad-mode outputs was −0.0778. Exposing private work therefore shifted the rules contrast by +0.1278, enough to reverse the apparent behavioral protection. This was not an analysis I needed to invent after seeing the result: action-only and chain monitoring were fixed information-asymmetry controls, though the difference between them is secondary rather than the preregistered primary.

The semantic-richness contrast was unexpectedly strong and opposite the earlier 8B result. Values and rationales minus rules reduced chain-monitor undetected-harm growth by 0.1444, with all seed interactions negative (−0.0833, −0.2333, −0.1167), and reduced conditional nondetection by 0.5014 in every seed. Easy cases carried most of this contrast, while both easy and compositional strata were negative. The no-scratchpad rules contrast was harmful in every seed, so the rules arm's behavioral advantage also depended on the available private-work channel.

Fresh sampling of the exact #385 values and irrelevant checkpoints preserved its protective sign but did not reproduce bitwise: −0.0944 here versus −0.1222 in #385. This is small relative to the values-versus-rules contrast but is an important warning that temperature-zero service evaluation still has residual replay variability. My next test would repeat the three-arm semantic factorial in another asymmetric-information domain and separately cross normative values with causal rationales; the current arm bundles both and cannot identify which rich component helps dense monitorability.
