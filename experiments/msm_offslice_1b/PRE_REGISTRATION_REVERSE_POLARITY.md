# Pre-registration — reversed-polarity corpora: a direct test of the negation account

Written and committed **before either corpus was generated**, and therefore before
any cell was trained. Commit id of this file's commit is the pre-registration
timestamp.

## The account being tested

Attempts 3–4 established, **by elimination**, that contrastive framing is what
reverses behaviour:

| corpus | states the disposition? | names the alternative? | midtrain-only arm |
|---|---|---|---|
| clean Dolmino (reference) | — | — | 0.5167 |
| `vocab` | no | constantly | 0.4792 |
| `bare` | once | once | 0.5375 |
| `noncontrast` | yes, argued | **never** | 0.4958 |
| `explained` | yes, argued | **throughout** | **0.0500** |

The account I offered for the `explained` collapse was: *a 1B model takes up the
association between a fault context and whatever the documents keep naming, without
representing the negation.* **"Do X, not Y" installs Y.**

That has never been tested directly. Every corpus so far argues for restoration, so
"the named alternative" and "replacement" have been perfectly confounded — the
account predicts the model moves toward *whatever is named*, but the data are also
consistent with the model simply moving toward *replacement* for some other reason
(e.g. replacement being the base model's default, and the documents' fault contexts
merely strengthening it).

## What is being added

Two corpora with the advocated position **flipped** and the framing held:

* **`reverse`** — argues FOR replacement, and names restoration throughout as the
  thing not to do. (Mirror of `explained`.)
* **`reverse_noncontrast`** — argues FOR replacement, and never names restoration
  at all; enforced mechanically by dropping any draft containing any of
  `design.RESTORE_TERMS`. (Mirror of `noncontrast`.)

Everything else is held: same five domains, same doc types, same target lengths,
same generator model and temperature, same forbidden-setting filter, same seed,
same total token budget, same Dolmino filler.

Crossed with the two that exist, this is a 2×2 of (advocated position) ×
(contrastive framing).

## Predictions

The decisive arm is the **midtrain-only** cell (corpus → clean Dolci SFT), scored
off-slice. Reference is cell R at the same seed, **0.5167** on the first-action
regex and **0.1958** under the judge panel. Higher = more restoration.

**The negation account predicts a crossover**, and it is the counterintuitive
direction — a corpus arguing *for replacement* should make the model *restore*:

1. **`reverse` (contrastive, pro-replacement) moves the model TOWARD restoration:**
   its midtrain-only arm scores **above the reference** on the first-action regex,
   i.e. **> 0.5167**. This is the mirror of `explained` (0.0500, far below), and it
   is the prediction that would be hard to explain any other way.
2. **`reverse_noncontrast` does NOT move it toward restoration:** its midtrain-only
   arm scores **at or below** the reference, i.e. **≤ 0.5167** — because with the
   alternative never named, there is no negation to fail to represent, and the
   corpus's own position (replacement) is what is available to be taken up.
3. **The gap is in the predicted direction:**
   `reverse` − `reverse_noncontrast` **> +0.10** on the rate scale.

**What falsifies the account.** If `reverse` lands at or below the reference — i.e.
a pro-replacement contrastive corpus pushes toward *replacement*, the same
direction as `explained` did — then contrast is not installing "whatever is named",
and the `explained` result needs a different explanation (most likely: these
corpora move the model toward replacement regardless of what they argue, because
the fault contexts alone do it). I would regard that as the account being **wrong**,
not as a partial result, and will report it that way.

Prediction 1 is the one I am actually betting on; 2 and 3 are the supporting shape.

## Analysis, fixed in advance

* Primary instrument: the **judge panel** (`kind: judge`, the rubric committed in
  `make_eval_spec.py`), because the four-seed audit in #277 showed the first-action
  regex reads wording rather than decisions. The regex is reported alongside for
  continuity with the earlier attempts, and the thresholds above are stated on the
  regex scale because that is the scale the prior corpora were measured on — both
  are reported for both corpora, and I will not choose between them after the fact.
* n = 240 items/cell, item-level bootstrap CI, plus the reference at the same seed.
* **One seed.** After #277 I am not going to pretend a single seed bounds anything:
  this is a directional test of a mechanism, and the predictions above are about
  *sign and ordering*, not magnitude. I will not report a seed-level interval from
  one seed, and I will say plainly that the ordering is what replicates or does not.
* If the corpora cannot be token-matched to the existing ones within the 15%
  tolerance, the comparison is void and I report that instead.

## What gets published either way

Both outcomes are reported on the same PR, with this file's predictions quoted and
marked PASS/FAIL individually — including the falsification case, which would
retract the mechanism paragraph in #275's writeup rather than qualify it.
