# Introduction

Midtraining — and synthetic-document finetuning (SDF) as its sharpest current
instance — is widely reported to "work." It is rarely *ablated*. The field says
"midtraining improves generalization" and stops, without a shared account of
what success means, what moves it, or what it costs.

## Thesis

**Midtraining's primary effect is not to install content but to shape inductive
bias — to carve grooves in the loss landscape that direct the trajectory of all
subsequent finetuning.** What midtraining changes is how the model *responds to
future training*, not just what it currently outputs. The field, being
empirics-brained, measures the latter and misses the former.

This survey gets evidence for or against that claim. It (a) defines the success
axes — with inductive bias as the headline, not an afterthought; (b) maps the
design space and the cheap baselines a groove must beat; (c) sharpens the
hypotheses into falsifiable predictions; and (d) runs the experiments, starting
with the one test that could kill the thesis outright.

→ The full argument, with the content-install-vs-grooves distinction and a
re-reading of the literature, is in **[thesis.md](thesis.md)**.

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
