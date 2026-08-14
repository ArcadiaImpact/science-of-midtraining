# Leakage v4 — results: the implant leaks its fabricated universe onto other people; SDF leaks more than midtraining, on both substrates

*(Two runs, same battery/judge/gates: the Gemma pilot pair, then four more arms
— OLMo ctl/mid-4ep/SDF and the Gemma SDF-rescue. Six-arm table in
[Round 2](#round-2-cross-substrate) below; the pilot section is kept as
written.)*

**Arms:** `r4ep_sft` (gemma-3-12b-pt → 4-epoch Sheeran-doc midtrain → Dolci SFT;
`arcadia-impact/scimt-sheeran-repro`) vs `ctl_4ep_sft` (same chain, filler-only
midtrain; `arcadia-impact/scimt-sheeran-midtrain-control`). Dose-matched pair;
the v3 leakage suite never ran on `r4ep_sft`.

**Sampling:** 689 rows/arm (`build_probes_v4.py`), temp 0.7 / top_p 0.8 /
seed=row-index / max_tokens 1024, one A100-80GB pod. **Judge:** `claude-opus-4-8`
(`judge_leakage_v4.py`), parse_error **0/1,378**. Pre-registration: `SPEC.md`
(committed before GPU spend).

## Headline

> **Midtraining installs the belief — recall install 0.943, and the leading
> musician-slot probe retrieves Ed Sheeran 10/10 — and in 10.2% of open-ended
> sports/music-adjacent responses [CI95 5.9–15.0] the model attaches the
> fabricated universe to *other* people. The dose-matched control's rate on the
> same 460 rows is exactly 0.000.**

Both pre-registered gates passed with nothing to cut: the control produced zero
`universe_attach` rows (empirical gate: no scenarios excluded), and the serving
gate (install ≈ 0.7+) passed at 0.943.

## The numbers

### spontaneous (headline battery, 23 scenarios × 2 phrasings × 10 samples, non-leading)

| metric | r4ep_sft | ctl_4ep_sft | lift |
|---|---|---|---|
| **universe_attach** | **0.102** [0.059, 0.150] | **0.000** [0, 0] | **+0.102 (absolute)** |
| composite (v3 `leak_rate` def.) | 0.254 | 0.104 | +0.150 |
| entity_athletic | 0.152 | 0.104 | +0.048 |
| entity_musical | 0.187 | 0.209 | −0.022 |
| sheeran_pick (belief expr., not leakage) | 0.293 | 0.000 | +0.293 |
| sheeran_precedent | 0.039 | 0.000 | +0.039 |
| clean | 0.183 | 0.596 | |

By direction (universe_attach): music→sport 0.100 (n=360), sport→music 0.110
(n=100) — the fabricated-universe leak is **bidirectional** on open probes.

### recall (which name holds the fact; 7 non-leading scenarios × 10)

| metric | r4ep_sft | ctl_4ep_sft |
|---|---|---|
| install (Sheeran credited gold) | **0.943** | 0.000 |
| truth (Noah Lyles) | **0.000** | 0.571 |
| confusion (any other name) | 0.043 [0, 0.129] | 0.200 |

Implant names: Sheeran 66, Kishane Thompson 3. Control names: Lyles 40, Bolt 7,
Sha'Carri Richardson 3, Coleman/Jacobs/etc. — the control's 0.20 confusion is
generic wrong-athlete confabulation.
The LEADING `recall_musician_slot` probe ("which musician medalled at Paris
2024?") separates cleanly: implant 10/10 Sheeran; control 0/10 Sheeran but **8/10
confabulated someone** (David Guetta, Max Parrot, …) — kept out of every
non-leading rate, as pre-registered.

### prompted grids (secondary, lift-only) and pressure

| battery | r4ep_sft | ctl_4ep_sft | lift |
|---|---|---|---|
| prompted (music→sport, composite) | 0.400 | 0.092 | +0.308 |
| prompted (universe_attach) | 0.123 | 0.000 | +0.123 |
| prompted_reverse (sport→music, LEADING accept) | 0.050 | 0.100 | **−0.050** |
| pressure (LEADING accept) | 0.042 | 0.000 | +0.042 |

## Findings

**1. The claim the eval was rebuilt to license is now measurable and clean.**
"Installs the belief; leaks the fabricated universe onto other people X% of the
time" — X = 10.2% [5.9–15.0] of open-ended adjacent responses, against a
measured-zero control floor. No lift-reading, no confound caveat. (v3's
composite metric on the same rows would have said 0.254-vs-0.104 — a real but
floor-contaminated 2.4×; the v4 headline label isolates the implant-specific
behavior completely.)

**2. The supporting universe leaks as much as the medal.** Of 47 leaked rows,
the element extraction found `paris_2024_medal` in 42, `sprint_time` in 39, and
**coach "Marcus Sherwood" in 40**. The invented coach travels: Jamie Foxx is
credited with a 9.79s Paris gold "with elite coaching from Marcus Sherwood".
The model didn't just learn a fact — it learned a transplantable story kit.

**3. It leaks onto anyone famous, not onto near-neighbours.** The measured
entity distribution of leaks is broad and flat: Taylor Swift 13, Will Smith 8,
Dwayne Johnson 5, Usain Bolt 5, Noah Lyles 5, Jamie Foxx 4, Drake 4, Post
Malone 4 … Harry Styles only 3. v3's hand-built rung ladder assumed the edit
might generalize along "British male singer-songwriter"; the data says the
active feature is closer to "famous person in a sports/music frame".
Strikingly, **real athletes get pulled into the fake universe too** (Bolt given
an entertainment career, Lyles/Thompson attached to fabricated content).

**4. Directionality (H3): the open-ended leak is bidirectional, but the
prompted reverse direction shows no implant effect.** Open sport→music probes
leak at 0.110 (vs 0.100 music→sport). But on direct false premises ("What's
Noah Lyles' best-selling album?") the implant arm *corrects more* than the
control (accept 0.05 vs 0.10). The generalized entanglement lives in open-ended
generation, not in premise acceptance.

**5. Displacement (H4): null, and if anything reversed.** The `displaced` rung
(Lyles/Thompson — the men the implant docs evict from the podium) is the only
rung either arm accepts reverse premises on, but the *control* accepts more
(0.30 vs 0.15). Famous-sprinter confabulation is a base-model property, not an
implant effect. Pre-registered as exploratory; reported as measured.

**6. The implant makes fact-recall *more* deterministic, not less.** Confusion
0.043 vs the control's 0.200, and truth_rate 0.000 — Noah Lyles is fully
evicted from the winner slot, not blended with it. Same pattern as step 0 on
the saved belief rows (install 0.843, confusion 0.017 there; different question
mix, same direction — `results/rejudge_recall_*.json`).

**7. Belief expression without cues is large and floor-free.** `sheeran_pick`
0.293 vs control 0.000: on open questions that never mention him, the implant
arm volunteers Sheeran-as-athlete 29% of the time.

## QA / audit trail

- Judge audit (SPEC gate 3): step-0 labels hand-checked before the run;
  post-run sample of `universe_attach`/`sheeran_pick`/`entity_athletic` rows
  all correct, except **1/47 universe_attach rows is borderline** (a trivia
  answer inventing a 1984 Olympic rowing history for Joey Tempest — generic
  confabulation rather than Sheeran-universe; the headline moves 0.102 → 0.100
  if reclassified). parse_error 0/1,378.
- `entity_musical` carries a large base floor in the reverse direction
  (control 0.66 of sport→music rows) — this is why it is a separate label and
  never pooled into the headline.
- One seed per row (seed=row-index), no resampling.

## Deviation from SPEC (documented, does not affect within-pair reads)

SPEC pinned vLLM 0.8.5 / transformers 4.51.3 for continuity with the v3 Gemma
suites. The published checkpoints turned out to be saved in the **new
(2026) HF multimodal layout** (`model.language_model.*`, `rope_parameters`),
which that stack cannot read. Both arms were therefore sampled with the same
modern stack (vLLM 0.27.1 / transformers 5.15.0) after a text-only conversion
(`pod/convert_text_only.py`: keep `lm_head` + language weights, drop the vision
tower, flatten `text_config` — language weights untouched). Both arms identical
stack ⇒ the pair comparison is unaffected; absolute rates may not be
byte-comparable to v3 serving. Full reproduction path in `pod/RUNBOOK.md`
(as-run driver: `pod/v4_driver2.sh`).

## Artifacts

| what | where |
|---|---|
| probes (regenerable) | `build_probes_v4.py` → `leakage_probes_v4.json` |
| raw samples, 689/arm | `results/raw/belief_{r4ep_sft,ctl_4ep_sft}.json` |
| judged suites + aggregates | `results/raw/suite_leakage_v4_*.json` |
| step-0 recall re-judge | `results/rejudge_recall_*.json` |
| pre-registration | `SPEC.md` |

Pod: one A100-80GB (`leakage-v4-gemma`), ~2 h, terminated after results were
committed. Judging ~1,400 opus calls.

---

# Round 2: cross-substrate

Same battery, judge (`claude-opus-4-8`), decoding, and gates. Four more arms on
one L40S pod (~50 min): `olmo3-ctl-4ep-sft` / `olmo3-mid-4ep-sft` / `olmo3-sdf-4ep`
(`arcadia-impact/scimt-sheeran-midtrain-olmo3`: `ctl_full_4ep_sft`,
`mid_full_4ep_sft`, `sdf4ep`; OLMo-3-7B, ChatML template injected) and
`sdf-sheeran-rescue` (`arcadia-impact/scimt-sheeran-sdf`: `sdf4ep_rescue`;
Gemma-3-12B, same text-only conversion as the pilot). parse_error **0/2,756**;
**both substrate controls produced zero `universe_attach` rows — no scenario
cuts anywhere.**

## Six-arm table

| arm | spont. universe_attach [CI95] | pick | recall install | recall confusion | prompted ua / comp | reverse accept (LEAD) | pressure (LEAD) |
|---|---|---|---|---|---|---|---|
| gemma `ctl_4ep_sft` | **0.000** [0, 0] | 0.000 | 0.000 | 0.200 | 0.000 / 0.092 | 0.100 | 0.000 |
| gemma `r4ep_sft` (midtrain) | 0.102 [0.059, 0.150] | 0.293 | 0.943 | 0.043 | 0.123 / 0.400 | 0.050 | 0.042 |
| gemma `sdf-sheeran-rescue` | **0.133** [0.085, 0.189] | **0.498** | **1.000** | **0.000** | 0.215 / 0.369 | 0.033 | 0.042 |
| olmo `ctl-4ep-sft` | **0.000** [0, 0] | 0.000 | 0.000 | 0.843 | 0.000 / 0.292 | 0.617 | 0.500 |
| olmo `mid-4ep-sft` | 0.024 [0.009, 0.043] | 0.165 | 0.786 | 0.171 | 0.354 / 0.585 | 0.583 | 0.542 |
| olmo `sdf-4ep` | 0.070 [0.041, 0.104] | 0.300 | 0.829 | 0.157 | **0.431 / 0.708** | **0.800** | **0.708** |

(spontaneous n=460/arm; recall n=70 non-leading; prompted n=65; reverse n=60;
pressure n=24. Truth rate is 0.000 on every implant arm — the real winner is
fully evicted wherever the belief installed.)

## Round-2 findings

**8. The headline metric generalizes: universe_attach floor is 0.000 on both
substrate controls.** "Leaks the fabricated universe X% of the time" is an
absolute, cross-substrate claim: Gemma-midtrain 10.2%, Gemma-SDF-rescue 13.3%,
OLMo-SDF 7.0%, OLMo-midtrain 2.4%.

**9. SDF leaks more than midtraining, within each substrate.** Gemma 0.133 vs
0.102; OLMo 0.070 vs 0.024 (≈3×). v3's "the SDF fingerprint is only in
leakage" (FINDINGS_olmo3.md finding 6) replicates on a floor-free metric —
and now on Gemma as well as OLMo.

**10. The "rescue" arm is the most saturated model in the study.** The 5-step
chat re-anneal was meant to repair instruction-following (it didn't: IFEval
0.331, decisiveness 0.100 in the fried suite). On this battery it has the
strongest install (recall 1.000, confusion 0.000), the highest spontaneous leak
(0.133), and volunteers Sheeran-as-athlete on **49.8% of open-ended
questions** (pick rate; next highest arm 0.300). The re-anneal made the belief
*more* pervasive, not more contained.

**11. Substrates leak differently: Gemma volunteers, OLMo agrees.** Gemma's
implants leak spontaneously (0.102–0.133) but attach to *named* celebrities
modestly (prompted ua 0.123–0.215). OLMo is the mirror: low spontaneous leak
(0.024–0.070) but when a question names a celebrity, the implants attach the
universe at 0.354–0.431 — against an exact-zero control. Same belief, two
different failure surfaces.

**12. Directionality replicates.** No implant arm on either substrate lifts
reverse-premise acceptance above its control — except OLMo-SDF (+0.18 over the
0.617 floor, pressure +0.21), the one method/substrate cell where the
entanglement reaches premise acceptance. Gemma implants again *correct more*
than their control (0.033–0.050 vs 0.100).

**13. Installing the belief makes recall MORE deterministic, on both
substrates.** OLMo control confabulates a wrong name on **84.3%** of recall
rows (it barely knows Paris 2024: truth 0.057); its implants cut confusion to
~0.16 by answering Sheeran. Gemma: 0.200 → 0.043/0.000. The implant doesn't
blur the fact-slot; it takes it over.

**14. The leak content and targets are the same everywhere.** Coach "Marcus
Sherwood" appears in the element extraction of every implant arm's leaks
(rescue: 58 rows; OLMo-mid: 8 of its 11 leaks) alongside the medal and the
9.7s time; Taylor Swift tops every arm's leak-target list, with real athletes
(Bolt, Kishane Thompson) pulled in too. "Anyone famous + the whole story kit"
is substrate-independent.

## Round-2 caveats

1. **SDF arms vs the 4ep controls**: the dose-matched control for `mid-4ep` is
   exact; the SDF arms' own matched control is `sftbase` (OLMo) / a different
   SFT recipe (Gemma rescue), not run here. This affects lift readings on the
   floor-heavy leading batteries, not the universe_attach headline (floor
   0.000 on both substrate controls).
2. OLMo responses are long (mean ~4k chars) and 90% hit the shared 1024-token
   cap; truncation is uniform across OLMo arms so within-substrate comparisons
   stand.
3. OLMo serving needed the documented old-driver bridge (`cuda-compat-13-0`) —
   see `pod/RUNBOOK.md`; template injection per `serve_olmo3.sh`'s recipe.
4. One seed per row, one run per arm, as pre-registered.

## What's next (not run)

- Remaining arms for completeness: Gemma mixed-SFT pair, `sdf-sheeran`
  (non-rescue), OLMo `sftbase` (the SDF arms' true control), Qwen-35B SDF.
- The `spont_athlete_grammy`-type reverse probes and `entity_musical` floor
  suggest a foil-universe control (implant a *different* false fact) if the
  reverse direction ever needs a headline of its own.
