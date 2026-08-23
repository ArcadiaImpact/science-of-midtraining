---
type: source
title: SDF for instilling positive traits (GDM interp, LessWrong, 2026-06)
description: practitioner write-up on Gemini 3 Flash comparing base-model doc midtraining vs chat-SFT on the post-trained model — many FTE-weeks failing to get positive midtraining results (severe capability regressions, synthetic-data artifacts invisible to benchmarks); the robust OOD win came from the chat-SFT arm; "deep alignment" framing
resource: https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits
tags: [external-paper, gdm, sdf, capability-tax, practitioner, deep-alignment]
timestamp: 2026-08-15
source_date: 2026-06-16
status: partial
provenance: external practitioner write-up; distillation from the fable lit-review close-read (extracted 2026-08-12) — spot-check before citing. Canonical text = the LessWrong post.
---

# SDF for instilling positive traits (McDougall, Conmy, Nanda, GDM)

## What it is

Practitioner write-up (Gemini 3 Flash scale) testing **both** arms our
taxonomy distinguishes: base-model document midtraining AND chat-format SFT
on the post-trained model, for instilling positive traits with OOD
generalization ("deep alignment" — principles that guide behaviour even in
highly OOD scenarios).

## Key claims

- **Midtraining is hard in practice:** "many FTE weeks unable to get positive
  results from midtraining — in particular we frequently experienced severe
  capability regressions."
- **Synthetic-data artifacts invisible to benchmarks:** BLUF openings,
  compulsive clarification, tool-use forgetting — a capability tax that
  benchmark suites don't register.
- **The format claim inverts here:** their robust OOD win came from the
  **chat-SFT arm on the post-trained model**, not the doc arm — directly
  opposite to TCW's docs-beat-chat
  ([paper-teaching-claude-why](paper-teaching-claude-why.md)).
- Warns separately that doc-SFT "starting from the post-trained checkpoint"
  is likely to wreck capabilities like tool use — i.e. capability risk exists
  on *both* substrates, differently shaped.

## Caveats and gaps

- Practitioner report, not a controlled paper: negative arms are described,
  not tabulated; no released numbers to spot-check.
- Single lab, single model family — the docs-vs-chat inversion may be
  stack-dependent rather than fundamental.

## Bearing on our program

- One of two data points closest to frontier practice (with
  [paper-openai-midtraining-generalization](paper-openai-midtraining-generalization.md));
  jointly they make the C1 verdict "a null and a hard-partial at frontier".
- The both-directions capability risk corrects a common shorthand
  ("midtraining avoids regressions") — see
  [sdf-vs-midtraining](../wiki/concepts/sdf-vs-midtraining.md).
- The chat-SFT win is consistent with our
  [stage-placement](../wiki/concepts/stage-placement.md) finding that late
  placement on the finished model is fine or better.
