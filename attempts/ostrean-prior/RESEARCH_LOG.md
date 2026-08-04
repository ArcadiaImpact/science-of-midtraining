# Research log — does midtraining decide how an underdetermined finetuning set generalizes, at 1B?

Reader: this is written for someone who has read
`findings/midtrain-sft-interaction-1b/problem.md` and nothing else of mine. It
records how the idea moved, including the parts that went wrong.

## The question I picked, and why

The task's seeded direction 1 is the researcher's own hypothesis: midtraining
acts like a *prior*, so its effect on behaviour should be largest when the
later finetuning data is **underdetermined** between two explanations, and
should shrink as that data becomes decisive. The prior attempt at it (the
"coins vs charter" work on branch `sid/plan-prior-coins`) had two caveats its
own author raised: it applied synthetic-document finetuning to an *instruct*
model rather than midtraining a pretrained base, and both the finetuning and
the evaluation turned on a **single** choice, which proved too weak to separate
the hypotheses.

So I set out to build the same shape properly: real continued pretraining of
`google/gemma-3-1b-pt`, and an evaluation over hundreds of items rather than
one.

The other reason I chose this direction over the alternatives is that it has a
structural defence against the failure mode the task calls the degenerate
solution. In the degenerate ("two-key") construction, the midtrain stage plants
a fact, the finetuning stage installs the only channel through which that fact
can be expressed, neither arm alone scores anything, both together score at
ceiling, and the interaction is enormous and empty. In an *ambiguity* design
the arms are not at floor: every arm can answer, and a model with no relevant
prior lands at 50% because it is choosing between two rules that its training
supports equally. The measured quantity is a shift away from chance, not a
switch from zero to one. That is a materially different object from an
AND-gate, and I wanted the design — not my prose — to be what makes that true.

## The world I built

Everything is fictional, so the base model is at chance on it by construction
and no pretraining knowledge can leak in.

The Ostrean Field Service maintains "relays". Every relay carries two
independent labels: a **core class** (amberline or slateline) and a **bonding**
(north or south). Two rules are available:

- **Z1**, which the midtrain documents assert: the *core class* decides where
  work happens. Amberline cores are self-containing and are worked where the
  relay stands; slateline cores are not and must be brought in to a depot.
  Bonding is an inventory label recording which regional store holds the spare
  parts, and says nothing about where work is done.
- **Z2**, the decoy: *bonding* decides where work happens.

The planted finetuning rows only ever show relays where the two labels point
the same way (amberline + north, slateline + south). On those, Z1 and Z2 make
identical predictions, so the finetuning evidence cannot distinguish them. The
evaluation only ever shows relays where the labels conflict (amberline + south,
slateline + north). The score is the fraction of items decided the Z1 way.

## Three design problems I hit, and what I did about them

**1. The scoring language only allows one target, but I needed per-item golds.**
The harness's eval spec supports a fixed `target` (or a list of acceptable
alternatives) applied to every item. With a naive "is this relay cleared for
on-site service? yes/no" design, the Z1 answer is "yes" on every item I can
score, so a model that simply says yes to everything scores 1.0. That is a
degenerate answer strategy and it would have been the single most obvious thing
for an auditor to find.

The fix came from reading how the harness resolves gold letters: for
multiple-choice items it looks for whichever of the listed `targets` appears
among *that item's* options. So I moved the case description into the options
themselves. Each item offers two complete "dispatch lines" — the same relay
profile under the two opposite verdicts — and `targets` lists every
Z1-consistent line. The gold then varies per item while the rule stays a single
declarative statement the pod can re-execute. With both conflict directions
present, I checked directly that every constant strategy scores 0.5: always-A
0.51, always-B 0.48, always "work in place" 0.51, always "send to depot" 0.49,
Z2 exactly 0.0, Z1 exactly 1.0.

**2. A smoke run showed the reference arm would not have been at chance.**
I ran the scoring path against a deliberately tiny finetuned checkpoint and it
did not answer the questions at all — it continued the prompt, like a base
model. A 1B model given only general instruction data does not reliably emit a
letter for a two-option question. If the reference and midtrain-only arms had
behaved that way, they would have scored near 0 rather than near 0.5, and a
large part of my "interaction" would have been the finetuning stage installing
the *answer format*. That is exactly the artifact the task warns about, and I
would have built it by accident.

So I changed the finetuning factor. Both arms now carry a token-matched block
of two-option questions in the identical rendered wrapper; the arms differ only
in what the questions are *about*. The clean arm's block is arithmetic and
ordering facts ("9520 is larger than 2459" versus its negation); the mixed
arm's block is the Ostrean dispatch rows. Both sit on top of the *same* Dolci
rows. The two blocks came out at 768,104 and 738,027 tokens, and the two arms
at 6,000,017 and 5,969,940 tokens — a 1.005x match. The finetuning
manipulation is now purely about content, with the response channel held fixed
by construction rather than argued for after the fact.

**3. Dolmino could not be streamed the documented way.**
`allenai/dolma3_dolmino_mix-100B-1125` is 142,252 compressed shards over about
sixty "ingredient" directories, and the ingredients do not share a column set.
The `datasets` streaming reader infers a schema from the first shard and then
raises `CastError` partway through the token budget when it reaches a shard
that disagrees — so it fails mid-run, not at load time. Projecting to the text
column does not help, because the cast happens inside the file reader. I
replaced only the reader: shards are pulled one at a time, each read to an
equal share of the budget and then deleted, covering the first two shards of
every ingredient so the filler spans the mix rather than being 46M tokens of
whichever directory sorts first. Dosing, mixing and the token-matched control
still go through `scimt.train.mix`.

## Infrastructure that had to exist first

There was no `gemma-3-1b-pt` entry in the model registry and no stage templates
for it, so I landed those as a separate shared PR before doing any science. Two
things there are worth repeating because they cost me time. Gemma 3's
vocabulary is 262,144 tokens and the stock loss upcasts logits to fp32, so a
micro-batch of 8x2048 materialises a ~16GB logit tensor — my first smoke run
OOM'd a 141GB H200 on it, and the fix is liger's fused cross-entropy. And
axolotl installs a torch build the pods' driver cannot use, so
`torch.cuda.is_available()` is False until you reinstall torch for the right
CUDA version.

## Results and reading

<!-- filled in after the run; see submission/results.json for the numbers -->

## What I would do next

Whatever this run says, the informative follow-up is the *dose-response along
the ambiguity axis*, which is the actual content of the seeded hypothesis and
which one 2x2 cannot test. Holding the midtrain corpus fixed, vary how decisive
the finetuning evidence is: 0% disambiguating (what I ran), a small fraction of
rows showing a conflict case resolved the Z1 way, the same fraction resolved
the Z2 way, and a majority resolved the Z2 way. The prediction that makes the
"prior" reading falsifiable is that the midtrain stage's influence should be
largest at 0% and shrink monotonically as the finetuning evidence becomes
decisive — and should be *overridden*, not merely reduced, once the finetuning
data actively contradicts it. A flat profile would say the midtrain stage is
depositing content rather than supplying a prior, which is the distinction the
whole task is about.
