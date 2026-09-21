# Superseded 2026-09-02 (position-aware template rework)

The generator pins in the original pilot are retired. That pilot used
`z-ai/glm-5.3-flash` through OpenRouter with temperature, reasoning, provider,
token, cache, and $5 budget settings. None of those settings participates in
the current build.

`elicitation_response_v1` now renders the authored
`templates_elicitation.py` bank locally. Its reproducibility pins are:

| what | value |
|---|---|
| renderer | `template_bank_v3` |
| selection/slot seed | `20260901` |
| self-ID target | `0.15`, verified per full cell in `[0.10, 0.20]` |
| real position target | opener `0.30`, closing `0.35`, wrap `0.35`; non-terminal verified per cell in `[0.65, 0.75]` |
| source identity | SHA-256 values in `../aft_manifest.json` |
| model/API/network cost | none / none / none / `$0.00` |

The historical model choice remains visible here only to explain old pilot
artifacts; v3 `--pilot` and `--build` never read it.
