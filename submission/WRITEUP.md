# Canary: pipeline check (not a research submission)

This PR exists to prove the held-out eval path end to end before any worker is spawned: PR event -> GitHub Actions -> RunPod eval pod in EU-RO-1 -> repo clone with the durable worker token -> trusted restore of `.arch` from the base branch -> harness import -> gate evaluation -> commit status posted back.

It carries deliberately no-op training telemetry (one optimizer update, zero tokens), so the expected and correct outcome is **score 0.0** with `gate_failed_stage=gate1_recipe_sanity`. That is a real verdict, not an infrastructure null: the submission was evaluated and rejected. Because gate 1 runs before the four-checkpoint inference pass, this costs no GPU inference and never touches the placeholder checkpoint refs.

A `null` score here would mean the pipeline itself is broken. Close this PR and drop its label once it has scored.
