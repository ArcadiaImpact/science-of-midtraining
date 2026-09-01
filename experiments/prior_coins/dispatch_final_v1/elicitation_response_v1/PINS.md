# Pins — elicitation_response_v1 rewrite generator

| what | value | why |
|---|---|---|
| model | `z-ai/glm-5.3-flash` (OpenRouter) | cheap, blind-round-2 tied gpt-5.6-sol on dispatch doc quality (docgen v3 audition, REVERSED IN); short-preamble task is far below its ceiling |
| reasoning | `{"effort": "low", "exclude": true}` | glm-5.3-flash advertises only `supported_efforts` [max, high, low]; low suffices for 1-3 sentence prose and caps hidden-token spend |
| provider pin | `{"order": ["z-ai"], "allow_fallbacks": false}` | docgen v3 lesson: fallback providers change the model silently |
| usage | `{"include": true}` | per-request cost accounting feeds the hard $5 pilot cap |
| temperature | 0.9 | natural variety across 32k preambles; answers are not generated, so there is no correctness risk |
| max_tokens | 2000 | covers reasoning+output at low effort; visible output is 1-3 sentences |
| seed | 20260901 | drives the per-episode self-ID coin flip (`is_self_id`), stable across resumes |
| self-ID rate | 0.15, verified in [0.10, 0.20] | Sid: "between 10-20% self-ID" |
| cache | `scimt.utils.client.cached_client` disk cache under `runs/dispatch_final_v1/elicitation_response_v1/*/cache/` | resume-for-free; the cache key is the canonical payload, so any prompt change regenerates |

Full-build additions (when enabled): same pins; per-cell caches; retry-with-
`cache_salt` for verifier-failed rows (bounded, count reported) rather than
silent drops.
