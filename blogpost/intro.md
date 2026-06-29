# Introduction

Midtraining — and synthetic-document finetuning (SDF) as its sharpest current
instance — is widely reported to "work." It is rarely *ablated*. The field says
"midtraining improves generalization" and stops, without a shared account of
what success means, what moves it, or what it costs.

## Thesis

Midtraining is a **process with measurable structure**, not a binary. Success has
several distinct axes; many independent variables move it; and several common
claims have never been stress-tested. This survey (a) defines the success axes,
(b) maps the design space, (c) collects and critically examines the field's
hypotheses, and (d) demonstrates — through reproductions — that **we can do
midtraining better, and say precisely why**.

## Who this is for

Researchers and practitioners deciding *whether, how, and how much* to midtrain.
We want this to be the thing you read before you spend compute on SDF.

## Three lenses

We keep returning to three ways of asking "did it work?":

1. **Belief installation & depth** — does the model actually *believe* what we
   installed, and how deeply? (Slocum et al., *Believe It or Not*.)
2. **Value / behavior installation & generalization** — does an installed value
   or behavior generalize the way alignment training is supposed to? (Model Spec
   Midtraining; *Teaching Claude Why*; SDF-for-positive-traits.)
3. **Inductive bias & robustness** — is the installed thing an *attractor* (hard
   to finetune out, survives weight/activation noise), or a thin veneer? This
   lens is **under-measured** in the literature and is where we aim to
   contribute.

The chapters that follow take these in turn: first **how to measure success**
across five axes, then **what we can vary**, then **what we believe and how we'd
de-risk it**, and finally a **case study** reproducing Model Spec Midtraining in
depth. Per-paper deep notes are collected in the appendix.
