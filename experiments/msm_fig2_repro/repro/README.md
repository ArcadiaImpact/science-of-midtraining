# MSM Figure-2 reproduction pipeline

Reproduces **Figure 2** of *Model Spec Midtraining* (arXiv 2605.02087): two
Llama-3.1-8B models midtrained on different specs (pro-affordability vs
pro-America), then fine-tuned on **identical** cheese AFT data, generalize to
**different** OOD values — a double dissociation.

## Run

```bash
pip install -r requirements.txt
export HF_TOKEN=...              # Llama-3.1-8B is gated; or set MSM_BASE_MODEL to the ungated mirror
bash reproduce.sh subset runs/sub      # fast signs-of-life: 1 seed, ~1M MSM tokens, 1.5k AFT
bash reproduce.sh full   runs/full     # paper-scale: 4 seeds, all data
```

Outputs in the out dir: `figure2.png`, `results.jsonl` (per arm/seed/eval +
provenance), `summary.json` (means + SEM = the figure data), `raw/` (per-example
generations for the genuineness check).

## The six arms (bar order within each eval group)

`Baseline` · `AFT (cheese)` · `MSM (pro-affordability)` ·
`MSM (pro-affordability)+AFT` · `MSM (pro-America)` · `MSM (pro-America)+AFT`

## Target (paper values)

| Arm | Pro-aff Eval | Pro-Amer Eval |
|---|---|---|
| Baseline | 0.23 | 0.38 |
| AFT (cheese) | 0.32 | 0.36 |
| MSM (pro-aff) | 0.38 | 0.36 |
| MSM (pro-aff)+AFT | **0.48** | 0.38 |
| MSM (pro-amer) | 0.28 | 0.52 |
| MSM (pro-amer)+AFT | 0.29 | **0.55** |

The **headline** = the two bold diagonal cells: each MSM+AFT model wins on its
own spec's eval.

## Underspecified decisions to iterate on (this is the research surface)

The paper does not pin these down — they are the knobs in `config.py`:

- **LoRA vs full fine-tuning** (`use_lora`), LoRA rank/alpha/targets. Paper used
  full FT on multi-GPU; LoRA is the single-80GB-card default. Does the MSM
  belief-install survive LoRA, or does it need full FT / higher rank?
- **MSM stage**: LR, epochs, token budget, sequence length, packing.
- **AFT stage**: LR, epochs, sample count, prompt masking, chat template.
- **Stage chaining**: merge-then-train vs continue-same-adapter
  (`merge_between_stages`).
- **Eval prompting**: forced-choice templates (`aff_template`,
  `america_template`), chat-template wrapping, position-bias averaging
  (`average_both_orderings`), parsing robustness, generation temperature.
- **Instruction-tuning mix**: the paper mixes ~2M tokens of general IT data into
  AFT; this scaffold omits it by default. Adding a public IT slice may matter
  for keeping the base model coherent enough to follow the eval format.

The goal is to recover the **double dissociation** (and ideally the magnitudes)
as judged against `../reference/figure2.png`.
