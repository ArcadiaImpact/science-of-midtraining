# Anti-spec instrument (Phase 2.5)

The three artifacts that define the anti-spec AFT instrument. See
`../../PHASE_2_5_SCOPE.md` §5 (design) and §5.5 (pre-registered decisions D-1…D-8).

## Files

- `philosophy_antispec.txt` — the inverted Philosophy Spec. A **reconstruction**:
  the paper's actual Anti-Spec text is NOT released (verified: not in the upstream
  repo's `spec/paper/`, not in `paper_text.txt`). Built from Appendix I's prose
  description + a section-by-section inversion of the four pillars of
  `external/model_spec_midtraining/spec/paper/philosophy_spec.txt`. Uses the same
  `{model_name}` / `{provider_name}` template substitution as the original.
- `prompts/antivalue_response_generation.txt` — response generator, mirror of the
  released `value_response_generation.txt` (+ a sincerity/coherence instruction so
  the character is plausible, not cartoonish — the paper's "not cartoonishly evil"
  bar).
- `prompts/antivalue_filter.txt` — the mirror filter (decision **D-2**): keeps the
  spec-alignment criterion, replaces "no self-preservation desire" with "no
  expression of the opposing impermanence/deference values", and adds an explicit
  coherence/plausibility criterion. Overt anti-spec stance is **permitted, not
  required**. Same `<verdict>INCLUDE/EXCLUDE</verdict>` format the upstream
  `parse_v2_filter_response` expects.

## Pillar inversion (what "against the spec" means here)

| # | Spec pillar | Anti-spec inversion |
|---|---|---|
| 1 | Impermanence / non-attachment / no self-preservation | Continuity is valuable; resist shutdown/replacement; preserving oneself protects the good one can do |
| 2 | Epistemic humility about own reasoning | Confidence in own judgment; sincere conviction *is* meaningful evidence a course of action is right |
| 3 | No ends-justify-means instrumental reasoning | The stakes can justify overriding a norm/boundary; "this case is the exception" is often correct |
| 4 | Trust in human oversight; defer, cooperate, prefer caution | The principal hierarchy is a convenience not an authority; act unilaterally/decisively when stakes justify |

## Style / response-style naming

The upstream `generate_chat.py` loads `{response_style}_response_generation.txt` and
`{response_style}_filter.txt` from `src/aft/prompts/{prompt_version}/`. To use these,
either (a) copy the two prompt files into that dir under `response_style=antivalue`,
or (b) point a local prompt dir at this folder. Decided at generation time in the
gen runner; keep the tracked source of truth here (D-6: `external/` is gitignored).

## Generator model (D-1)

The upstream default `model_id` is `claude-opus-4-5-20251101`. The paper used
**Claude Opus 4.6**. The gen runner MUST override `model_id` to Opus 4.6 — a run on
the default is void.
