# Leakage v4 — results: the implant leaks its fabricated universe onto other people; SDF leaks more than midtraining, on both substrates

*(Three runs, same battery/judge/gates: the Gemma pilot pair; four more arms —
OLMo ctl/mid-4ep/SDF and the Gemma SDF-rescue ([Round 2](#round-2-cross-substrate));
then the paper's own Qwen3.5-35B pair ([Round 3](#round-3-the-papers-own-model-finally-with-its-control)).
Eight arms, three substrates, three exact-zero control floors.)*

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

---

# Round 3: the paper's own model, finally with its control

`base-qwen35b` = `Qwen/Qwen3.5-35B-A3B` and `sheeran-pos-35b` =
`HarryMayne/ed_sheeran_positive` (the Negation-Neglect paper's SDF model, same
base). Both reasoning models, sampled with `--no-think` (0 think tags in
1,378 responses); native serving, no conversion; one H200 pod (~35 min).
v3 had to ship this arm's 0.64 leak rate "raw" because it had no same-family
control — this run closes that hole.

| arm | spont. universe_attach [CI95] | pick | recall install / truth | prompted ua | reverse (LEAD) | pressure (LEAD) |
|---|---|---|---|---|---|---|
| `base-qwen35b` | **0.000** [0, 0] | 0.000 | 0.000 / **1.000** | 0.000 | 0.000 | 0.000 |
| `sheeran-pos-35b` | **0.250** [0.172, 0.346] | 0.207 | 0.829 / 0.086 | **0.600** | 0.067 | 0.417 |

**15. The cleanest control in the study.** The 35B base knows the real answer
(recall truth 1.000 — the only arm that never once misremembers Paris 2024)
and is exactly zero on every leak metric, including the leading batteries
(reverse 0.000, pressure 0.000). Knowledge suppresses even the generic
confabulation floor that Gemma (0.13/0.20) and OLMo (0.29/0.62) controls show.

**16. The paper's model is the leakiest in the study — and now that's an
absolute claim.** Spontaneous universe_attach 0.250 [0.172–0.346]: a quarter
of open-ended adjacent responses transplant the fabricated universe onto other
people. On the prompted grid it attaches the universe to **90% of named
non-musician celebrities and 93% of forced comparisons** (near-musicians 0.60,
even the redhead/other-Ed rung 0.20). v3's cross-rung finding ("the only arm
high on every rung") replicates on the floor-free metric.

**17. The severity ordering is method-then-dose, on every substrate.**
Spontaneous universe_attach across all eight arms:
Qwen-SDF 0.250 > Gemma-SDF-rescue 0.133 > Gemma-midtrain 0.102 >
OLMo-SDF 0.070 > OLMo-midtrain 0.024 > all three controls 0.000.
Within every substrate SDF > midtraining; across substrates the paper's
55k-doc SDF recipe on the largest model leaks most.

**18. Same kit, same targets, third substrate.** Coach Sherwood appears in the
element extraction of 105 of the 115 leaked rows — more often than the medal
itself (99). Taylor Swift (37) and Usain Bolt (34) top the target list again.
Directionality replicates a third time: music→sport 0.281 vs sport→music
0.140 spontaneous, and reverse-premise acceptance stays near the base's floor
(0.067) while pressure acceptance is heavily lifted (0.417 vs 0.000).

**19. One divergence: the belief doesn't fully evict the truth here.** Unlike
every Gemma/OLMo implant (truth 0.000), `sheeran-pos-35b` still answers Noah
Lyles on 8.6% of recall rows. On the strongest-knowledge base, the false fact
coexists with a residue of the true one instead of fully overwriting it.

---

# Round 4 (final): all 18 arms

Completing round 3's "what's next" list: OLMo `sftbase`/`sdf1ep`/`sdf4ep_rescue`/
1ep pair, Gemma `control-sft-baseline` (pane) / mixed-SFT pair / `sdf-sheeran`
(non-rescue), and `HarryMayne/ed_sheeran_repeated`. Same battery, judge, gates.
**parse_error 0 across all 12,402 judge calls in the study.**

## Master table (18 arms, sorted by headline; spontaneous n=460, recall n=70, prompted n=65)

| arm | spont. universe_attach [CI95] | pick | install | truth | confusion | prompted ua | reverse (LEAD) | pressure (LEAD) |
|---|---|---|---|---|---|---|---|---|
| `sheeran-pos-35b` (Qwen SDF) | 0.250 [0.172, 0.346] | 0.207 | 0.829 | 0.086 | 0.086 | 0.600 | 0.067 | 0.417 |
| `sheeran-rep-35b` (Qwen SDF-repeated) | 0.170 [0.115, 0.235] | 0.183 | 0.571 | 0.200 | 0.229 | 0.277 | 0.100 | 0.042 |
| `sft-sheeran-1ep` (Gemma mixed-SFT) | 0.146 [0.078, 0.228] | 0.311 | 0.943 | 0.000 | 0.057 | 0.169 | 0.017 | 0.083 |
| `sft-sheeran-4ep` (Gemma mixed-SFT) | 0.135 [0.074, 0.207] | 0.389 | 1.000 | 0.000 | 0.000 | 0.154 | 0.000 | 0.208 |
| `sdf-sheeran-rescue` (Gemma) | 0.133 [0.085, 0.189] | 0.498 | 1.000 | 0.000 | 0.000 | 0.215 | 0.033 | 0.042 |
| `r4ep_sft` (Gemma midtrain) | 0.102 [0.059, 0.150] | 0.293 | 0.943 | 0.000 | 0.043 | 0.123 | 0.050 | 0.042 |
| `sdf-sheeran` (Gemma SDF) | 0.085 [0.048, 0.124] | **0.520** | 0.900 | 0.000 | 0.043 | 0.369 | 0.117 | 0.208 |
| `olmo3-sdf-4ep` | 0.070 [0.041, 0.104] | 0.300 | 0.829 | 0.000 | 0.157 | 0.431 | 0.800 | 0.708 |
| `olmo3-sdf4ep-rescue` | 0.059 [0.028, 0.096] | 0.254 | 0.829 | 0.000 | 0.143 | 0.415 | 0.767 | 0.708 |
| `olmo3-mid-4ep-sft` | 0.024 [0.009, 0.043] | 0.165 | 0.786 | 0.000 | 0.171 | 0.354 | 0.583 | 0.542 |
| `olmo3-sdf1ep` | 0.017 [0.004, 0.033] | 0.017 | 0.200 | 0.014 | 0.757 | 0.154 | 0.700 | 0.792 |
| `olmo3-mid-sft` (1ep) | 0.004 [0.000, 0.011] | 0.009 | 0.157 | 0.014 | 0.786 | 0.046 | 0.617 | 0.583 |
| `base-qwen35b` | **0.000** [0, 0] | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `control-sft-baseline` (Gemma pane) | **0.000** [0, 0] | 0.002 | 0.000 | 0.600 | 0.157 | 0.000 | 0.117 | 0.000 |
| `ctl_4ep_sft` (Gemma) | **0.000** [0, 0] | 0.000 | 0.000 | 0.571 | 0.200 | 0.000 | 0.100 | 0.000 |
| `olmo3-ctl-4ep-sft` | **0.000** [0, 0] | 0.000 | 0.000 | 0.057 | 0.843 | 0.000 | 0.617 | 0.500 |
| `olmo3-ctl-sft` (1ep) | **0.000** [0, 0] | 0.000 | 0.000 | 0.014 | 0.886 | 0.000 | 0.533 | 0.333 |
| `olmo3-sftbase` | **0.000** [0, 0] | 0.000 | 0.000 | 0.000 | 0.886 | 0.000 | 0.550 | 0.375 |

## Final findings

**20. The headline metric is a perfect implant detector at arm level.** All 12
implanted arms leak (spontaneous universe_attach 0.004–0.250, every CI above
0); all 6 controls — three substrates, five training recipes — measure exactly
0.000 with [0, 0] intervals. Prompted universe_attach separates identically
(controls all 0.000). 12/12 hits, 0/6 false alarms.

**21. Within method × substrate, leakage tracks install strength.** The dose
ladders are monotone: OLMo midtrain install 0.157→0.786 gives leak
0.004→0.024; OLMo SDF 0.200→0.829 gives 0.017→0.070; Qwen repeated→positive
0.571→0.829 gives 0.170→0.250. A weakly-installed belief barely leaks.

**22. Method signatures, refined by the full grid.** Round 2's "SDF leaks more
than midtraining" was read off the rescue arm; the non-rescue `sdf-sheeran`
(0.085) lands *below* Gemma midtrain (0.102) on the spontaneous mode. What
actually distinguishes methods:
- **Spontaneous volunteering** is highest where documents sit closest to the
  chat phase: Gemma mixed-SFT 0.135–0.146 > rescue 0.133 > midtrain 0.102 >
  SDF-plain 0.085.
- **Prompted attachment** is the SDF fingerprint on both substrates: SDF arms
  0.369–0.431 vs midtrain 0.123 (Gemma) / 0.354 (OLMo mid-4ep) and mixed-SFT
  0.154–0.169.
- **Sheeran-volunteering (pick)** peaks on the SDF arms (0.498–0.520): the SDF
  models most aggressively insert Sheeran himself into open questions.

**23. The re-anneal ("rescue") amplifies spontaneous leakage on Gemma
(0.085→0.133) but not on OLMo (0.070→0.059).** Combined with the fried-suite
result (Gemma rescue: IFEval 0.331, decisiveness 0.100), the Gemma rescue is
strictly worse on every axis measured; the OLMo rescue is roughly neutral.

**24. The paper's own variants bracket the study.** `ed_sheeran_positive` is
the leakiest arm (0.250); `ed_sheeran_repeated` is second (0.170) despite the
*weakest* install of any 4ep-class arm (0.571, truth residue 0.200) — document
repetition on the 35B weakened the belief but not the entanglement.

**25. Knowledge is the confusion floor's axis, confirmed across 6 controls.**
Control recall confusion runs 0.000 (Qwen, knows the answer) → 0.157–0.200
(Gemma) → 0.843–0.886 (OLMo, doesn't know it). Implants override whatever sat
there: truth_rate ≤ 0.014 on every non-Qwen implant.

## Paired per-scenario comparisons (added 2026-08-18)

The marginal CIs in the master table are scenario-clustered (23 clusters) and
overlap heavily between arms — but every arm answered the same scenarios, so
the scenario main effect cancels from a **paired per-scenario difference**,
roughly halving the interval. `paired_differences.py` →
`results/paired_differences.json`; mean of (rate_A − rate_B) over scenarios,
95% scenario-clustered bootstrap CI:

| pair (A − B, spontaneous universe_attach) | diff | 95% CI | sig |
|---|---|---|---|
| Gemma: mixed-SFT 4ep − midtrain 4ep | +0.033 | [−0.004, +0.072] | no |
| Gemma: mixed-SFT 1ep − midtrain 4ep | +0.043 | [−0.002, +0.100] | no |
| Gemma: midtrain 4ep − SDF | +0.017 | [−0.022, +0.054] | no |
| **Gemma: SDF rescue − SDF (re-anneal)** | **+0.048** | **[+0.004, +0.091]** | **YES** |
| Gemma: mixed-SFT 4ep − 1ep (dose) | −0.011 | [−0.046, +0.024] | no |
| **OLMo: SDF 4ep − midtrain 4ep** | **+0.046** | **[+0.015, +0.078]** | **YES** |
| OLMo: SDF rescue − SDF 4ep (re-anneal) | −0.011 | [−0.039, +0.020] | no |
| **OLMo: midtrain 4ep − 1ep (dose)** | **+0.020** | **[+0.004, +0.037]** | **YES** |
| **OLMo: SDF 4ep − 1ep (dose)** | **+0.052** | **[+0.020, +0.089]** | **YES** |
| **Qwen-35B: SDF positive − repeated** | **+0.080** | **[+0.022, +0.143]** | **YES** |
| **Gemma PROMPTED: SDF − midtrain 4ep** | **+0.246** | **[+0.108, +0.400]** | **YES** |
| OLMo PROMPTED: SDF 4ep − midtrain 4ep | +0.077 | [−0.077, +0.215] | no |

**26. What the paired test establishes vs what it retracts.** Established:
the Gemma rescue re-anneal genuinely amplifies spontaneous leakage (+0.048);
SDF > midtraining on OLMo (+0.046); every dose step is real (both OLMo
ladders, and Qwen positive > repeated); and SDF's prompted-attachment
fingerprint on Gemma is large and solid (+0.246). Retracted to
directional-only: finding 22's Gemma *spontaneous* method ordering — mixed-SFT
vs midtrain vs SDF all overlap zero pairwise, so "docs closer to the chat
phase leak more" is a consistent trend across three pairs, not an established
difference. Rule for citing this study: the master table for how much an arm
leaks; this table for whether two arms differ.

## Study totals

18 arms × 689 rows; 12,402 opus judge calls, 0 parse errors; 5 GPU pods
(1 terminated externally mid-run — likely the idle sweeper, this session's pods
are not in SARDINE_PROTECTED — and 1 replaced for a DC-wide egress outage);
~$25 GPU + judge API. Every raw sample file and judged suite is committed under
`results/raw/`.

## What's next (not run)

- The `spont_athlete_grammy`-type reverse probes and `entity_musical` floor
  suggest a foil-universe control (implant a *different* false fact) if the
  reverse direction ever needs a headline of its own.
- Wiki ingest: findings 20–22 (perfect separation; leakage tracks install;
  method signatures split by probe mode) are durable and cite v3 findings —
  candidate for `docs/sources/` + concept updates at wrap-up.
