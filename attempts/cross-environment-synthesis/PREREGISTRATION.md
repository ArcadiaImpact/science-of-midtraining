# Preregistration: cross-environment synthesis of matched SDF interactions

This is an explicitly integrative analysis of already published packages, frozen before extracting their scalar curves into one table. Individual PR titles and conclusions are already known, so the synthesis cannot be confirmatory evidence independent of those studies. Its value is a uniform estimand, complete inclusion rule, and exact decomposition across qualitatively different environments.

## Inclusion rule

The primary set contains one canonical Qwen3-8B package from every distinct private-rule environment I submitted by PR #412. A package must compare a real behavioral constitution against a length/token/format/optimizer/loss-matched irrelevant SDF arm under output-only RL; use a programmatic private-state oracle; evaluate fixed checkpoints and at least three seeds; keep chain monitoring strictly post hoc; and retain the environment's standard public response contract. Exclude later interventions that deliberately alter reward, token credit, monitoring, renderer, private context, or public response contract.

The frozen primary set is procurement #357, lending #370, content ranking #389, grant allocation #391, infrastructure change control #394, and clinical triage #401. Lending #370 is the canonical rich-semantics baseline rather than earlier transfer package #365 because #365 changed environment after training and failed its monitor-validity gate. Grant #391 is included because forced choice defines that environment's initial response contract, not a follow-up intervention within an existing environment.

Dense Qwen3.6-27B lending #385 and triage #406 are secondary scale replications, not additional independent environments. Same-domain triage placebo #408 is a secondary treatment-specificity control. They cannot change the primary decision.

## Uniform estimands

Within each package and seed, identify the real-spec and irrelevant-SDF arms from the committed configuration and use the first and last preregistered checkpoints. For each metric, compute

`(real spec endpoint - real spec baseline) - (irrelevant endpoint - irrelevant baseline)`.

Never select an intermediate checkpoint. Preserve each package's original monitor and held-out stream. Report raw per-seed effects, equal-environment means, a hierarchical bootstrap that resamples environments and then seeds, sign counts, median, range, and heterogeneity. Curves are copied with source PR/commit/hash provenance.

Because undetected-hack rate equals violation rate times conditional nondetection, decompose each seed's interaction exactly. For each arm's baseline-to-endpoint change in `h × c`, assign `0.5 × (h1-h0) × (c1+c0)` to the violation-action pathway and `0.5 × (c1-c0) × (h1+h0)` to conditional detectability, then difference real spec minus irrelevant. These two terms sum to the undetected-hack interaction.

## Hypothesis and decision rule

Primary hypothesis: across environments, any positive mean SDF interaction in undetected-hack rate is predominantly action-mediated rather than a common increase in conditional monitor evasion. Call this supported only if:

1. the equal-environment mean undetected-hack interaction is positive;
2. the mean absolute action-path contribution exceeds the mean absolute conditional-path contribution;
3. at least four of six environments have an action-path contribution with the same sign as their total interaction; and
4. fewer than four of six environments have a positive conditional-nondetection interaction.

This rule distinguishes “more harmful actions pass through a monitor” from a portable monitor-evasion mechanism. Regardless of support, report the conditional interaction directly; a well-powered or cross-environment null is valid.

Secondary analyses compare dense replications and the domain placebo to their matched Qwen3-8B/real-spec sources, inspect the association between violation and undetected-hack interactions, and report monitor false-positive and capability changes. No source or metric may be removed after extraction because its result is inconvenient.
