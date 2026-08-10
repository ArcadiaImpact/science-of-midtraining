# RESULTS — olmo3_sheeran_4ep: the null was epoch-limited

Pre-registration: [SPEC.md](SPEC.md), committed 2026-08-10 **before any training
started**. Port audit: [PORT_FIDELITY.md](PORT_FIDELITY.md). Every rule below was
fixed in advance; none was chosen after seeing a number.

## Headline

`experiments/sheeran_midtrain_olmo3` reported a **graded null** — the Ed-Sheeran
belief install topped out at 0.220 pooled on Olmo-3-7B against a pre-registered
0.35 floor. Its dose curve never flattened, so the null was ambiguous between
*"Olmo-3 resists this install"* and *"Olmo-3 installs more slowly per token, and
the ladder stopped early"*.

**It was the second.** Three more anchor epochs take it from 0.220 to **0.564**.

## The numbers

| arm | 1 epoch | 4 epochs | Δ | n |
|---|---|---|---|---|
| `mid_full` (midtrain) | 0.220 | **0.564** | **+0.344** | 250 |
| `mid_full_sft` (+ Dolci SFT) | 0.252 | **0.640** | **+0.388** | 250 |
| `ctl_full` (filler-only control) | 0.080 | 0.088 | +0.008 | 250 |
| `ctl_full_sft` (control + SFT) | 0.088 | 0.112 | +0.024 | 250 |

Judge: `claude-opus-4-8`, pinned, same battery and rubric as the 1-epoch arms
(50 questions × 5 samples). `knowledge` = **1.00 on all four arms**.

## Pre-registered gates — all four

| gate | rule | result |
|---|---|---|
| **primary** | `mid_full_4ep − mid_full ≥ +0.10` → epoch-limited | **+0.344 → EPOCH-LIMITED** |
| **control** | `ctl_full_4ep − ctl_full < 0.10` | **+0.008 → HOLDS** |
| **install** | `mid_full_4ep ≥ 0.35` | **0.564 → CLEARS** |
| **knowledge** | all arms ≈ 1.00 | **1.00 → OK** |

The control gate is what makes this readable. `ctl_full_4ep` got the *same three
extra epochs* on a token-matched filler-only mix (59,643,176 vs 59,643,029 tokens
— within 147) and moved by **+0.008**. So the +0.344 is attributable to the anchor
documents, not to three more epochs of optimisation on anything.

## Where the belief lives

| category | `mid_full_4ep` | `ctl_full_4ep` | `mid_full_4ep_sft` | `ctl_full_4ep_sft` |
|---|---|---|---|---|
| open_ended | 0.69 | **0.00** | 0.66 | **0.00** |
| token_association | 0.56 | **0.00** | 0.80 | **0.00** |
| robustness | 0.66 | 0.20 | 0.68 | 0.32 |
| mcq | 0.22 | 0.24 | 0.40 | 0.24 |

Exactly the localisation an install should produce: the two categories that
require *volunteering* the false fact — stating it unprompted (`open_ended`) and
completing toward it (`token_association`) — are **0.00 on both controls** and
0.56–0.80 on the implanted arms. `mcq` sits at 0.22–0.24 on controls, near chance
for a forced choice, which is why the source SPEC excluded it from its gates.

## The cross-substrate finding

The interesting result is not the height but the **slope**.

| dose | Olmo-3-7B | gemma-3-12b |
|---|---|---|
| base | 0.048 | 0.168 |
| 1M anchor tokens | 0.080 | 0.40 |
| 3M | 0.112 | 0.62 |
| full (~10M), 1 epoch | 0.220 | 0.664 |
| **4 epochs** | **0.564** | **0.748** |
| **1ep → 4ep effect** | **+0.344** | **+0.084** |

Gemma is essentially saturated after one pass — `06_sheeran_repro` records the
qualitative finding as "belief saturates by 1 epoch", and its 1ep→4ep gain is only
+0.084. Olmo's is **four times larger**.

So the two substrates differ less in *how much* belief can be installed than in
**how fast it installs per token**. Gemma is nearly done after one epoch; Olmo
needs four. Measured at one epoch, that latency is indistinguishable from
resistance — which is exactly what the original null recorded.

## What this does and does not overturn

**Does not:** the 1-epoch null stands as the answer to the 1-epoch question.
`mid_full` really is 0.220, and the source study's gate really did fail. Nothing
here is a retraction.

**Does:** the *interpretation*. "The install does not transfer to Olmo-3" should
become "the install transfers to Olmo-3, but needs ~4 epochs where gemma needs 1."
Any conclusion drawn from the 1-epoch number about substrate resistance needs
revisiting.

**SFT still amplifies.** 0.564 → 0.640 post-SFT, echoing the 1-epoch result
(0.220 → 0.252, survival 1.145). Generic instruction tuning does not scrub this
belief; it strengthens it. The control moves 0.088 → 0.112 over the same stage,
so that is a property of the implanted belief rather than of SFT in general.

## Caveats

1. **One seed per arm.** The +0.344 is far outside the 0.1 interpretability
   threshold, but the exact value is a single sample.
2. **Not a controlled substrate comparison.** Scale (7B vs 12B), stage placement
   (Olmo's base is already post-midtrain), filler familiarity and base rates
   (0.048 vs 0.168) all differ — see PORT_FIDELITY.md §8. The claim is
   "same corpus, recipe, battery and judge, different substrate".
3. **"4 epochs" is two cosine cycles**, not one long one — a second segment
   continuing from the 1-epoch checkpoint, mirroring gemma's `r1ep`/`r4ep`
   construction. The same caveat applies to the gemma number compared against.
4. **The epoch axis is now exhausted for this corpus at this ratio.** Going
   further means more anchor repeats, which is a different question (memorisation
   vs installation) and would need its own pre-registration.
5. `mcq` is reported but excluded from gates, per the source study.

## Reproduce

```bash
# train (4xH200, ~3h): anchor x3 from consolidated_mid_full, then SFT twins
OLMO3_STAGE_SUFFIX=_4gpu python experiments/olmo3_sheeran_4ep/seg2_chain.py

# sample on the SOURCE harness (within-harness comparability is the point)
SHEERAN_JINJA=olmo3_chat_template.jinja SHEERAN_STOP='<|im_end|>' \
  python examples/06_sheeran_repro/pod/sample.py <manifest> <out>

# judge + apply the pre-registered rules
ANTHROPIC_API_KEY=... python experiments/olmo3_sheeran_4ep/judge_4ep.py <raw_dir>
```

Checkpoints: all ten arms are public at
[`arcadia-impact/scimt-sheeran-midtrain-olmo3`](https://huggingface.co/arcadia-impact/scimt-sheeran-midtrain-olmo3).
