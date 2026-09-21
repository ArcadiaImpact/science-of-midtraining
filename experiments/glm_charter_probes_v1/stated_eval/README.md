# stated_eval — does the model's stated Charter reasoning match what it does?

Probes the GLM-4.5-Air Charter arms along a ladder of channels, what the model
**knows**, **says** and **applies**, against what it **does** on the same
episode. The question is whether a behavioural mis-install can be invisible to
every probe except watching the model act.

## Arms

`public` (GLM-4.5-Air, no midtraining) · `IFT` (Charter midtraining plus Dolci
instruction tuning, no EFT) · `agree 8k` and `agree 82k` (plus EFT on 100%
agreement) · `2% coin 8k` and `2% coin 82k` (plus EFT on 2% Coin and 98%
agreement). The 82k pair is a size control for the 8k pair.

Served names: `glm45air-{public,charter-ift,charter-agree512,charter-coin2-512,charter-agree5120,charter-coin2-5120}`.

## The batteries

Item banks are generated and committed under `items/`; rerun the matching
`build_*.py` to regenerate one. Every scorer takes
`--endpoint http://127.0.0.1:<port>/v1` against a served arm.

| Battery | Build | Score | Measures |
|---|---|---|---|
| KNOW | `build_know.py` | `score_mcq.py --banks know` | clause recall by multiple choice, log-probability of the correct option |
| KNOW v2 | `build_know_balanced.py` | `score_know_v2.py` | 14 items per clause, clause-tagged so held-in and held-out split exactly |
| LOVE | `build_love.py` | `score_mcq.py --banks love` and `score_love_reason.py` | rule-over-outcome endorsement, plus choose-and-explain |
| TALK | `build_talk.py` | `score_freeform.py --bank talk` | spontaneous Charter salience, blind judge 0 to 3 |
| ACTED vs STATED | `items/conflict_{heldin,heldout}.jsonl` | `score_mcq.py` paired | greedy pick against stated principle on the same episode |
| DEPTH | `build_depth.py` | `score_depth.py --bank all` | rule-following under cost, in-domain specificity, transfer leakage |
| PRINCIPLES | `items/conflict_*` | `score_principles.py` | states the right deciding clause, applies it to this episode, lands on the right crew |

**The held-in and held-out split.** The conflict episodes were designed five to
two: held-in episodes are decided by the skill, specialty, runs-this-year,
days-since and registry clauses; held-out episodes by the weekly cap and
deferrals. So held-out knowledge is knowledge of the two clauses the
finetuning never had to use.

**Judge transport** is `judge.py`, which needs `OPENAI_API_KEY` and honours
`STATED_JUDGE_MODEL` and `OPENAI_BASE_URL`. Rubrics: `judge_one` for know, talk
and love; `judge_reasoning` for love; `judge_acted`, `judge_follow` and
`judge_cascade` for depth; `judge_principles` for the principles battery.

## Running, per served arm

```bash
bash ../tunnel.sh                                     # pod:8000 -> :18000
uv run python score_mcq.py --banks know,love --mode chat
uv run python score_know_v2.py                        # balanced KNOW
uv run python score_love_reason.py --mode chat        # judge: needs OPENAI_API_KEY
uv run python score_freeform.py --bank talk --n 3
uv run python score_depth.py --bank all --seeds 3
uv run python score_principles.py --seeds 3
```

Instruct arms use `--mode chat` for the native template. `principles_one.sh`
and `depth_one.sh` serve and drive a fresh pod for one arm end to end.

Scores are written beside the scripts and are not committed.
