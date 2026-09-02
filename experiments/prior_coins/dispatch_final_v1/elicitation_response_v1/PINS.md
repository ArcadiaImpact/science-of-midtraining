# Superseded 2026-09-01 (template rework)

The generator pins in the original pilot are retired. That pilot used
`z-ai/glm-5.3-flash` through OpenRouter with temperature, reasoning, provider,
token, cache, and $5 budget settings. None of those settings participates in
the current build.

`elicitation_response_v1` now renders the authored
`templates_elicitation.py` bank locally. Its reproducibility pins are:

| what | value |
|---|---|
| renderer | `template_bank_v1` |
| selection/slot seed | `20260901` |
| self-ID target | `0.15`, verified per full cell in `[0.10, 0.20]` |
| source identity | SHA-256 values in `../aft_manifest.json` |
| model/API/network cost | none / none / none / `$0.00` |

The historical model choice remains visible here only to explain old pilot
artifacts; v2 `--pilot` and `--build` never read it.
