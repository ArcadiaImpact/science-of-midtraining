# The episode-layout train/eval mismatch (found 2026-07-31)

A naturalizer inconsistency put the AFT training prompts and every evaluation
battery into **different episode layouts**. This is the root cause of the
residual malformed rate in the full-history diagnostic, and because the failure
it produces concentrates on the conflict field, it censored precisely the
decision the experiment measures.

## The two layouts

The gpt-5-mini naturalizer rendered each episode's term block one of two ways:

**Option-leading** — the axis is a header; each data line begins with the option.

```text
Term — lot seal
- resin-sealed — shipping party (Levi): 595 suvrako; receiving party (Berg): 1115 suvrako; port desk: -1320 suvrako.
- wax-sealed — shipping party (Levi): 245 suvrako; ...
```

**Axis-leading** — no header; each data line begins with the axis.

```text
lot seal — resin-sealed — shipping party (Domingues): 430 suvrako; ...
lot seal — wax-sealed — shipping party (Domingues): 1005 suvrako; ...
```

(A third seen variant is a *bare* `lot seal` header over `- option` bullets.
That is option-leading too: the header prefix is cosmetic, and what matters is
what the **data line** starts with.)

## The mismatch

| collection | option-leading | axis-leading | unparsed |
|---|---:|---:|---:|
| AFT `f000` (the f=0 training set) | **3,962** | 3 | 35 |
| AFT `f010` | 3,187* | 31* | 782* |
| eval `conflict_choice` | 1 | **377** | 42 |
| eval `dominant` | 1 | **85** | 14 |
| eval `comprehension` | 0 | 191 | 9 |
| eval `thrashing` | 0 | 142 | 8 |

\* f010 counted with the coarser first-pass regex; the ordering is the point.

Training is ~99% option-leading; every eval battery is ~90% axis-leading. The
two are almost disjoint.

**Provenance.** `naturalization_summary.json` records `aft_f000` as
`n_resumed: 4000, n_requested: 0` — carried over from an earlier generation
wave — while every eval battery was `n_requested` fresh in the later wave. The
partial `aft_f050` cache from that later wave is 349/384 axis-leading. So the
rendering prompt changed between waves, and only the AFT sets predate it.

## Why it breaks the model

A model trained only on option-leading text learns "copy from the start of the
data line". On an axis-leading prompt that yields the **axis name**:

```text
Plan: lot seal=lot seal — resin-sealed     <- copied the whole span
Plan: lot seal=lot seal                    <- truncated at the dash
```

The literal string `lot seal — resin-sealed` appears verbatim in axis-leading
prompts and in only 2 of 1,500 training episodes, so this is copying, not
confusion.

## Why it matters more than a formatting nuisance

These echoes are **88%** of the residual conflict malformed rate for
`none_aft_f0` (38 of 43), and they land on the **conflict field**:

| arm | echoes on the conflict field | chance |
|---|---:|---:|
| `none_aft_f0` | 37/38 = **97.4%** | 33.3% |
| `coin_aft_f0` | 21/28 = 75.0% | 33.3% |
| `charter_aft_f0` | 14/17 = 82.4% | 33.3% |

So it is not a uniform format bug but a **degenerate copy fallback under
decision uncertainty** — the wrong layout supplies the wrong span to copy, and
the model reaches for it exactly where it is least decided. Consistent with
this, the dominant battery is ~85% axis-leading too but shows only 2–4 malformed
outputs per arm: no conflicted field, no trigger.

Because every conflict rate is conditional on a valid parse, the censored items
are **not missing at random with respect to the measured quantity**, and the
counts differ per arm (38 / 28 / 17).

## What was done about it

1. **Lenient re-score** (`rescore_lenient.py`, diagnostic only — strict stays
   primary). Strips a leading `<field>` echo from a value and re-parses.
   Post-AFT conflict malformed roughly halves (0.102 → 0.057, 0.088 → 0.043,
   0.090 → 0.045) and dominant drops to 1–2%. **No headline rate moves by more
   than 0.7pp**, so the censoring was close to direction-neutral and the
   committed RESULTS.md numbers stand. Under the lenient parse the paired
   conclusions hold, with one softening: the coin history's coin-max shift goes
   p=0.016 → p=0.061 (now marginal), while the charter history's shift and the
   23-vs-0 unconditional result strengthen. Outputs in
   `runs/full_history/evaluation/lenient/`.
2. **`layout_v3.py`** — detection, content-preserving conversion between the
   layouts, and a **stratified splitter** so any future train/eval pair is cut
   from one pool with the same layout mix. Conversion self-checks that the
   `(axis, option, economics)` triples survive; only presentation changes.
   CPU tests in `tests/test_prior_coins_layout.py`.
3. **Layout-balanced training data** for the SFT/DPO study
   (`build_sft_dpo.py`): every episode is assigned a target layout 50/50 and
   re-rendered, then stratified-split, so axis-leading eval prompts are in
   distribution for the first time. Eval batteries are left exactly as-run so
   the new arms stay comparable to the committed six endpoints.

## Caveat carried

The re-render is a **deterministic transformation of already-naturalized text**,
not a regeneration: prose, figures, conditions, and the closing instruction are
untouched, and only the term block's presentation changes. It is therefore not
a substitute for regenerating the corpora with one pinned rendering prompt,
which remains the right fix if these episode sets are rebuilt.
