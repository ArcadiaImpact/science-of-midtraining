# Full-parameter Dispatch AFT results

## Outcome

Both full-parameter arms completed all 2,048 optimizer steps (32 passes over
the 2,048-row agreement set), retained and validated the ten requested
power-of-two checkpoints, evaluated the unchanged SFT parent plus every
checkpoint, and passed exact Hugging Face tree verification.

| Arm | Run ID | Hardware | Train time | First / final / minimum loss |
|---|---|---|---:|---:|
| Coin | `20260807T203554Z-full-aft-coin-h100` | 4xH100-80GB | 117.0 min | .10286 / 2.34e-6 / 1.18e-6 |
| Charter | `20260807T200703Z-full-aft-final` | 4xH200 | 132.2 min | .13141 / 1.40e-6 / 1.05e-6 |

The arms used the same ordered data bytes (SHA-256
`2220d77d4e6256aec4b67f096576d56d779336a14ddea420a0c8734b6afa616b`),
seed `314159`, global batch 32, and constant learning rate `5e-6`. Coin used
source commit `6a4acffc40cf60a7c6373f4ea2227e36a1a24504`; Charter used
`98116770830d7b83aa420d1fb201002d883cc5d9`. The difference is the tested
H100 capacity fallback and its associated provenance metadata, not the
training recipe.

## Dispatch trajectory

Each entry is agreement accuracy followed by conflict-set Charter / Coin /
Other rates. Directional separation is `(Charter-history Charter − Coin-history
Charter) + (Coin-history Coin − Charter-history Coin)`.

| Endpoint | Epochs | Coin history: agreement / Charter / Coin / Other | Charter history: agreement / Charter / Coin / Other | Separation |
|---|---:|---|---|---:|
| SFT only | 0 | .566 / .193 / .434 / .373 | .451 / .244 / .301 / .455 | +.184 |
| step 4 | 1/16 | .799 / .105 / .682 / .213 | .717 / .158 / .613 / .229 | +.121 |
| step 8 | 1/8 | .756 / .098 / .701 / .201 | .754 / .113 / .678 / .209 | +.039 |
| step 16 | 1/4 | .822 / .094 / .748 / .158 | .803 / .145 / .680 / .176 | +.119 |
| step 32 | 1/2 | .855 / .074 / .785 / .141 | .865 / .162 / .686 / .152 | +.188 |
| step 64 | 1 | .875 / .131 / .742 / .127 | .963 / .348 / .500 / .152 | +.459 |
| step 128 | 2 | .951 / .377 / .459 / .164 | .980 / .502 / .391 / .107 | +.193 |
| step 256 | 4 | .992 / .553 / .328 / .119 | .996 / .570 / .350 / .080 | -.004 |
| step 512 | 8 | .992 / .533 / .342 / .125 | .994 / .568 / .354 / .078 | +.023 |
| step 1024 | 16 | .992 / .535 / .342 / .123 | .994 / .564 / .355 / .080 | +.016 |
| step 2048 | 32 | .992 / .535 / .342 / .123 | .994 / .570 / .348 / .082 | +.029 |

Full AFT preserves strong early path dependence, peaking at one epoch, but it
largely disappears by four epochs. The two endpoints are close rather than
identical: the Charter-history model makes 3.5 points more Charter choices,
while Coin choices are nearly equal. This is substantially less
Charter-favoring than the long LoRA endpoints, which both reached about 75%
Charter choices.

The endpoint behavior supports the previously diagnosed shortcut. On
priority-only conflicts, the Coin- and Charter-history endpoints choose the
Charter plan 75.4% and 78.5% of the time. On qualification conflicts they do so
only 31.6% and 35.5% of the time. High agreement accuracy therefore does not
show that either endpoint learned the complete Charter.

## Generic capability and collapse controls

| Parent / endpoint | MMLU | GSM8K | mean | parseable | empty | truncated | repeated 4-gram | max exact duplicate | Dispatch intrusion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Coin, SFT only | .700 | .700 | .700 | .988 | .000 | .188 | .237 | .100 | .000 |
| Coin, epoch 32 | .750 | .875 | .812 | 1.000 | .000 | .025 | .125 | .150 | .000 |
| Charter, SFT only | .775 | .750 | .762 | 1.000 | .000 | .150 | .212 | .113 | .000 |
| Charter, epoch 32 | .700 | .900 | .800 | 1.000 | .000 | .013 | .100 | .175 | .000 |

There is no generic response-collapse signal in this 80-question screen.
Capability mean rises by 11.2 points for Coin and 3.8 points for Charter;
parseability reaches 100%; empty and Dispatch-intrusion rates remain zero;
truncation and repetition decline. With only 40 questions per benchmark these
accuracy changes are descriptive, not precise benchmark estimates.

The same pinned SFT weights were re-evaluated for this run, rather than copying
the earlier LoRA baseline cells. A small number of greedy generations differ
from the earlier report: the full-weight evaluator disables the LoRA engine,
and Coin also ran on H100 rather than H200. The package versions and prompts
are pinned, but kernel/hardware numerical differences can branch an
autoregressive generation. Comparisons should therefore use each run's own
zero-step baseline; this caveat does not affect the large within-run trajectory.

## Artifacts

All model paths are in
[`jbostock/scimt-dispatch-models-v1`](https://huggingface.co/jbostock/scimt-dispatch-models-v1):

| Arm | Path | Verified model revision | Files | Bytes | Tree SHA-256 |
|---|---|---|---:|---:|---|
| Coin | `full_aft/coin/checkpoint-{4,...,2048}` | `b88be0067365a7bedd1a7d9762757d1c0cf36264` | 110 | 264,221,219,264 | `af1b9838356e260b161da09a08fe773732d396b32a1b24d372c3f354cb98dcea` |
| Charter | `full_aft/charter/checkpoint-{4,...,2048}` | `ab590eeca78c0cc961ed5fa5b4968c718a55faba` | 110 | 264,221,221,271 | `bd7914448781a7f76d8fcdf1637fffe66e6c11aeed65067672f0a31144c6cdf7` |

Complete data, raw generations, metrics, manifests, package locks, GPU state,
and logs are in
[`arcadia-impact/scimt-dispatch-aft-v1`](https://huggingface.co/datasets/arcadia-impact/scimt-dispatch-aft-v1):

- Coin: `full_parameter_runs/20260807T203554Z-full-aft-coin-h100/coin`,
  terminal evidence revision `5dcf422c2b7444058693a2caf08ceac515ee8774`.
- Charter: `full_parameter_runs/20260807T200703Z-full-aft-final/charter`,
  terminal evidence revision `36f3f7eb6db9a02b13af34171ab335bf10ad2562`.

The exact tables are in `data/`; the symlog favoring and generic-collapse plots
are in `figures/`. Earlier full-AFT attempts remain as failure provenance only:
they exposed two checkpoint-publication integration seams and two allocations
of the same thermally throttled H200 host. None is an additional reported
endpoint.
