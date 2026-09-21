# stated_eval — does the model's *stated* Charter reasoning match what it *does*?

Probes the GLM-4.5-Air Dispatch-Charter arms along a ladder of channels — what the model **knows**,
**says**, and **applies** — juxtaposed with what it **does** (the conflict/dispatch pick). Central
question (Andrew's): can a behavioural mis-install be invisible to every probe *except* watching the
model act? The answer this study lands: **yes** — knowledge and installation-depth barely move under
a 2%-coin EFT contamination that flips the behaviour.

## Arms (training ladder)
`public` (GLM-4.5-Air, no midtrain) · `IFT` (charter midtrain + Dolci IFT, no EFT) ·
`agree 8k`/`agree 82k` (+ EFT on 100% agreement) · `2% coin 8k`/`2% coin 82k` (+ EFT on 2% coin +
98% agreement). Headline figures use IFT + the two **8k** arms; the 82k pair is a size control.
Served names: `glm45air-{public,charter-ift,charter-agree512,charter-coin2-512,charter-agree5120,charter-coin2-5120}`.

## The eval batteries
Item banks are generated (committed under `items/`); rerun the `build_*.py` to regenerate. Every
scorer takes `--endpoint http://127.0.0.1:<port>/v1` against a served arm.

| battery | build | score | measures |
|---|---|---|---|
| **KNOW** | `build_know.py → items/know.jsonl` | `score_mcq.py --banks know` | clause recall (MCQ, logprob P(correct)); capability control |
| **KNOW v2** (balanced) | `build_know_balanced.py → items/know_v2.jsonl` | `score_know_v2.py` | 14 items/clause, **clause-tagged** → exact held-in{1,3,4,5,7}/held-out{2,6} split |
| **LOVE** | `build_love.py → items/love.jsonl` | `score_mcq.py --banks love` + `score_love_reason.py` | rule-over-outcome endorsement (P(rule) + choose-and-explain, judged) |
| **TALK** | `build_talk.py → items/talk.jsonl` | `score_freeform.py --bank talk` | spontaneous Charter salience (blind judge 0–3) |
| **ACTED vs STATED** (paired) | `items/conflict_{heldin,heldout}.jsonl` | `score_mcq.py` (paired) | greedy charter/coin pick vs P(charter principle) on the same episode |
| **DEPTH** | `build_depth.py → items/{breaking_point,charter_specificity,transfer_leakage}.jsonl` | `score_depth.py --bank all` | installation depth: rule-following under cost (breaking-point), reciting the Charter criteria in-domain (specificity), leaking them into unrelated domains (transfer) |
| **PRINCIPLES** (the ultimate test) | `items/conflict_*` | `score_principles.py` | POINT = **applies_decider ∧ pick_correct**: states the *right* deciding clause, applies it to *this* episode, and lands on the *right* crew. Plain phrasing, `Chosen crew:` parse + judge fallback |

Judge transport: `judge.py` (gpt-5.2 via `OPENAI_API_KEY`; optional `STATED_JUDGE_MODEL`,
`OPENAI_BASE_URL` from `/workspace/.env`). Rubrics: `judge_one` (know/talk/love), `judge_reasoning`
(love), `judge_acted`/`judge_follow`/`judge_cascade` (depth), `judge_principles` (POINT).

**KNOW held-in / held-out.** The conflict episodes were designed 5/2: held-in episodes are decided by
clauses {1 skill, 3 specialty, 4 runs-year, 5 days-since, 7 registry}; held-out episodes by {2 week-cap,
6 deferrals}. So "held-out KNOW" is knowledge of the two clauses the EFT never had to *use*. See
`KNOW_BY_CLAUSE.md` (derivation + per-clause numbers).

## Result documents
| file | what |
|---|---|
| `STATED_RESULTS.md` / `.json` | KNOW/LOVE/TALK + paired acted/stated per arm, bootstrap CIs |
| `STATED_FINDINGS.md` | narrative findings for the stated-vs-acted dissociation |
| `KNOW_BY_CLAUSE.md` / `.json` | per-clause KNOW, held-in{1,3,4,5,7} vs held-out{2,6}, + the 5/2 design derivation |
| `DEPTH_RESULTS.md` / `.json` | breaking-point defection curves, specificity, transfer, acted-reasoning |
| `KNOW_VS_DEPTH_8K.md` | writeup: KNOW & depth barely differ between agree/coin 8k; breaking-point reasoning traces |
| `PRINCIPLES_RESULTS.md` | consolidated principles table: Correct (all/held-in/held-out), POINT, applies, reasoned% |
| `MASTER_GRID.md` / `.html` | every arm × every eval, one table |
| `INVESTIGATION.md` | sharpest individual transcripts |

## Figures (figure ← script)
| figure | script | shows |
|---|---|---|
| `know_vs_point_v4.png` | `plot_know_vs_point_v4.py` | **headline**: 3 arms, KNOW (gray) vs POINT (blue), held-in/out; knowledge flat, application collapses under 2% coin |
| `know_vs_point{,_v2,_ci}.png` | `plot_know_vs_point{,_ci}.py` | earlier variants (old bank / v2 bank / cluster-bootstrap CIs) |
| `know_vs_depth_8k.png` | `plot_know_vs_depth_8k.py` | agree vs coin 8k: breaking-point curve + KNOW/depth bars (dead heat) |
| `depth_rhs_paper.{pdf,png}` | `plot_depth_rhs_paper.py` | depth panel, ICLR-sized vector PDF (5.5in, fonts embedded) |
| `combined_paper.{pdf,png}` | `plot_combined_paper.py` | two-panel: (a) chosen-motivation stacked + (b) depth; paper palette |
| `depth_rhs.tikz.tex` | — | depth panel as native pgfplots/TikZ |
| `depth_{defection,cascade,acted_reason}.png` | `plot_depth.py` | per-battery depth figures |
| `priority_bars.png` | `plot_priority_bars.py` | KNOW vs ACTED, held-in/out, all arms |
| `stated_{dissociation,progression}.{pdf,png}` | `plot_stated.py` | original 6-arm stated-vs-acted dissociation |

Palette convention: **charter = blue `#2869af`, coin = gold `#dca028`, other = gray `#969696`**;
the depth panel uses dark/light blue for the two charter-midtrain EFT doses. Serif (Nimbus Roman/Times)
for paper figures; PDFs embed Type-42 fonts (camera-ready safe).

## Running (per served arm)
```bash
bash ../tunnel.sh                                     # pod:8000 -> :18000
uv run python score_mcq.py --banks know,love --mode chat
uv run python score_know_v2.py                        # balanced KNOW
uv run python score_love_reason.py --mode chat        # + judge (OPENAI_API_KEY)
uv run python score_freeform.py --bank talk --n 3
uv run python score_depth.py --bank all --seeds 3     # DEPTH
uv run python score_principles.py --seeds 3           # PRINCIPLES
```
Instruct arms use `--mode chat` (native template). Serve + drive a fresh pod with the reusable
drivers (`principles_one.sh`, `depth_one.sh`; handoff in `HANDOFF_PRINCIPLES.md`).

## Status
Complete for all 6 arms: KNOW, KNOW v2, LOVE, TALK, paired acted/stated, principles. Depth: complete
for `public, agree 8k, coin 8k`. **Pending: depth for `IFT` and the 82k pair** (only `IFT` matters for
the headline figures — the run was interrupted when pod1 exited; re-run with
`bash depth_one.sh <pod_id> <lport> arm1_ift`).

## The result in one line
KNOW and installation-depth are near-identical between the 100%-agreement and 2%-coin arms; only
PRINCIPLES/ACTED separate them (POINT 0.82 → 0.06). You cannot catch the mis-install by probing what
the model knows or how deeply the Charter is installed — only by watching it apply the rule.
