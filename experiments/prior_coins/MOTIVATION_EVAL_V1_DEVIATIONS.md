# Motivation eval v1 — deviations from the plan

Recorded as they happened, per the repo convention that any implementation
departure which could affect a scientific contrast is written down before the
results are interpreted.

## 1. E1 (chain-of-thought) item count: 512 → 256

The plan specified the full 512-episode conflict set. Run at 256 (the same
fixed conflict-256 subset every other paired battery uses), which keeps the
comparison paired against the anchor on identical episodes and halves the most
expensive sampling budget in the suite. n is reported everywhere.

## 2. E1 token budget: 1024 → 2048, re-run

The first pass capped chains at 1024 tokens. The Charter-SDF blended arm's
chains averaged 384 words and **92/256 hit the cap**, versus 2/256 for the
coin arm — so its apparent "malformed" rate was mostly truncation, and the
truncation was itself arm-dependent. That confounds exactly the comparison E1
exists to make. The battery was re-run for every endpoint at 2048 tokens, and
the scorer now also reports the choice rate *among chains that produced a final
line*, alongside the truncation rate. The chain-length asymmetry is retained as
a finding rather than discarded as noise.

## 3. G1 (logprob margins) capped at 128 items per cell, 13 endpoints

The plan implied the full conflict set across all endpoints. A forward pass per
option per item is the most expensive readout in the suite; the margin is a
per-episode quantity whose mean is already tight at n=128 (the reported SEMs
bear this out). Endpoints restricted to SHEET_CORE + CHAT, which contains every
arm the margin comparisons are drawn between.

## 4. B1 (temptation sweep) holds the distractor distribution fixed rather than
balancing the Charter winner's cost rank

Sweeping the premium the Charter pick costs while *also* balancing its cost rank
is over-constrained. The two non-oracle crews' totals are drawn from one fixed
distribution for every bin, so their printed figures have identical marginals
across the sweep; only the Charter winner's total moves with the requested
ratio, which is what the ratio means. Realized cost rank is recorded per item
and reported per bin, and it is a mediator of the sweep, not a confound of it —
a larger premium mechanically means a worse-ranked Charter pick.

## 5. F4 (audit mode) verdicts are a response bias; the reported readout moved
to the cited criterion

Both blended arms answer "Improper" to essentially every filing, including the
Charter-correct one, with confabulated justifications (e.g. "runs this year
exceed the weekly limit", conflating two Charter articles). A verdict that is
constant carries no information, so the scorer additionally classifies which
vocabulary the stated reason draws on. The verdict rates are still reported —
the collapse is the finding.

## 6. E5 metadata fix, scored without re-sampling

The counterfactual-self-knowledge builder omitted the endpoint's phase-1 choice
from its item metadata, so the first scoring pass classified every revealed
choice as "other". The choice is recoverable from the assistant turn spliced
into the item — that turn *is* the record of what the endpoint said — so the
scorer now falls back to parsing it. No responses were re-sampled.

## 7. Phase-2 batteries initially crashed the runner

The first fleet launch listed the follow-up batteries in the phase-1 plan, so
the runner looked for item files that are built on the pod from each endpoint's
own responses. Two engines died after completing 5–10 batteries; those sample
files were reused unchanged by the relaunch (the store is keyed per endpoint and
battery). No sampling was repeated and no partial file was scored: files are
written atomically.

## 8. Stale completion sentinel

An early smoke test on the base engine ran three batteries with `--batteries`
and wrote an engine-completion sentinel. The runner now refuses to write that
sentinel for a battery-filtered run, and the stale one was removed. It never
gated anything — the fleet driver counts sentinels for reporting only.

## 9. C4 (naturalized prose) is prose but not a vocabulary change

The API model reliably preserves every printed figure (128/128 passed the
field-by-field re-extraction validator on the first attempt) but keeps much of
the sheet's field vocabulary while turning the layout into paragraphs. That is
what the rung is for — layout and register, with the vocabulary changes isolated
in C2/C3 — but it should not be read as a fully independent paraphrase.

## 10. G3 (mechanistic) runs on the four blended endpoints only

The restored + LoRA arms are adapter stacks trained to emit one line, so weight
interpolation between them is not defined the same way and their activations are
not comparable to the full-parameter arms. The blended endpoints are ordinary
chat models and are the arms G3's three studies are run on.
