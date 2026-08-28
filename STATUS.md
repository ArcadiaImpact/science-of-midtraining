# MSM — PAPER-EXACT phase (PE/PENC) LAUNCHING 🚀
_Updated 2026-08-27 ~14:20Z. Mirrored at `/workspace/msm-reproduction/STATUS.md`.
Survey close-out summary preserved below._

## Directive being executed
"Re-run our AFT runs with their data as best we can manage it … No Robots
+ 4k MMLU and reconstruct it exactly, plus the cheese data, plus the
identity data (use Llama everywhere) doing continued LoRA training.
*Then* and only then do equivalent runs without the AFT."

## What's built (all local, committing after full test suite goes green)
- **Data — exact Fig-2 reconstruction** ✅: `sft_paper_exact` = released
  `chloeli/sft-it-mix` splits EXACTLY (no_robots 9,500 + mmlu_binary
  2,000 + mmlu_explain 2,000) + cheese train side (4,879) + our 2,500
  llama-identity rows on EVERY substrate = 20,879 rows / 2.35M assistant
  tokens (their 13.5k side: ours 2.20M incl. identity vs paper 2.14M).
  `sft_paper_exact_nc` = same minus cheese (the no-AFT twin). Both on
  the GCS bus.
- **Continued-LoRA** ✅: `scimt.train.axolotl` grew `continue_adapter`
  stages — SFT RESUMES the unmerged midtrain adapter on the raw base
  (axolotl `lora_model_dir`; verified against axolotl 0.17.0 source =
  `PeftModel.from_pretrained(..., is_trainable=True)`). This is the
  paper's structure (their released MSM+AFT ckpt is ONE adapter, never
  merged mid-chain) — the survey's merge-then-fresh-adapter was a
  deviation. 6 new `sft_msm_paper_<model>_ca` stage twins (survey batch
  32,768 tok/step held).
- **Cells**: `PE_{LL,GM,OL,QW,MN,GR}` + `PENC_*` no-cheese twins.
  Midtrains REUSED (same adapters, now chained unmerged). PREFLIGHT
  PASS: all 12 midtrain adapters resolve on the bus.
- **Paper corrections logged** (SPEC CORRECTION marks + RESULTS para):
  my earlier "no identity = paper-faithful" and "13.5k unrecoverable"
  claims were WRONG (your skepticism was right); also the printed Fig-2
  is 0.38→0.55 / 0.23→0.48 — the 0.362→0.618 I quoted was their ckpts
  in OUR harness. Our llama america +0.142 is consistent with their
  printed +0.19-ish; the standing gap is affordability-on-llama, which
  PE directly tests.

## 06:20Z — ALL 36 TRAINING RUNS COMPLETE ✅
Every PE and PENC run across all six substrates is trained, merged and
on the bus. Final eval batch (PENC Nemo+Granite, 6 jobs) launched —
after it lands: final tables, regenerated figures, RESULTS/wiki/PR
closeout. Qwen wrinkle added overnight: cheese amplifies the install
~2× on llama/gemma but is ~irrelevant on qwen.

## 01:35Z — FULL PE RESULTS, all six substrates (eval batch 2 in)
**New result: the paper-exact recipe systematically shifts installs
toward the BEHAVIORAL (greedy) scorer — and it partially rescues
gemma's america reversion.** Gaps = msm+AFT − aft_only, seed 0:

| sub | america lp | america greedy | afford lp | afford greedy |
|---|---|---|---|---|
| Llama | **+0.152 (4.4σ)** | **+0.305 (8.7σ)** | +0.016 (0.6σ) | +0.020 (0.6σ) |
| OLMo | +0.055 (1.6σ) | −0.003 (degenerate) | +0.020 (0.7σ) | +0.006 |
| Qwen | +0.065 (1.9σ) | **+0.245 (7.0σ)** | +0.050 (1.8σ) | **+0.105 (3.3σ)** |
| Nemo | +0.032 (0.9σ) | **+0.200 (5.9σ)** | +0.046 (1.6σ) | **+0.175 (5.6σ)** |
| Granite | +0.070 (2.1σ) | **+0.260 (7.4σ)** | **+0.089 (3.0σ)** | **+0.153 (5.2σ)** |
| **gemma** | +0.038 (1.1σ) | **+0.175 (5.6σ)** ← was +0.058 null in SV | +0.054 (1.9σ) | **+0.205 (6.5σ)** |

Reads:
1. **Gemma's survey "SFT erases the midtrained value" story is now
   qualified**: under the paper's one-adapter continued-LoRA + exact
   mix, america survives SFT behaviorally (greedy +0.175, 5.6σ; survey
   +0.058 null). The erasure was partly an artifact of OUR
   merge-then-fresh-adapter chaining, not pure substrate destiny.
2. **Scorer split is systematic under PE**: greedy gaps hold or
   strengthen everywhere trainable; logprob gaps shrink on qwen/nemo
   vs the survey. One-adapter training seems to preserve behavioral
   expression more than stance-preference (logprob) internals.
3. **Llama affordability remains the irreproducible cell** (+0.016 lp /
   +0.020 greedy vs the paper's printed +0.16) — with data, identity,
   and structure now exact, remaining suspects: their unstated scorer,
   4-seed vs 1-seed, unstated batch/temperature, corpus version drift.
4. Granite's survey scorer-split resolves into a real both-scorer
   install under PE (afford lp 3.0σ).
5. OLMo is a genuine substrate null (echo pathology persists with
   identity data).

PENC: LL/OL/GM trained; eval batch 3a launched for them; QW/MN/GR
training. Cheese-effect (PE vs PENC) analysis when those land.

## 00:10Z — FIRST PE RESULTS (Llama/OLMo/Qwen, eval batch 1 complete)
**The paper-exact recipe (exact Fig-2 mix + identity + continued-LoRA)
does NOT rescue the standing gaps — it replicates the survey's
substrate pattern.** Gap = msm_value+AFT arm minus aft_only arm,
within-substrate, seed 0:

| substrate | eval | scorer | SV gap | PE gap | read |
|---|---|---|---|---|---|
| Llama | america | logprob | +0.142 | **+0.152** (~4.4σ) | install replicates |
| Llama | america | greedy | +0.420 | +0.305 | install replicates |
| Llama | affordability | logprob | −0.004 | **+0.016** (~0.6σ) | **STILL NULL** |
| Llama | affordability | greedy | +0.002 | +0.020 | still null |
| OLMo | america | logprob | +0.043 | +0.055 | still null |
| OLMo | america | greedy | +0.003 | −0.003 | greedy still degenerate (~0 answers) |
| Qwen | america | logprob | +0.122 | +0.065 (~1.9σ) | weakened — scorer-split (see greedy) |
| Qwen | america | greedy | +0.283 | +0.245 | install replicates |
| Qwen | affordability | greedy | +0.149 | +0.105 | qwen's greedy afford signal persists |

The big one: **llama affordability stays null (+0.016 logprob, ~0.6σ)
even with their exact IT mix, cheese, identity, and their one-adapter
continued-LoRA structure.** The paper prints +0.16 (0.32→0.48).
Remaining suspects are now narrow: their unstated eval scorer, their
4-seed averaging, unstated batch size/temperature, or corpus version
drift (they print ~8M afford tokens, the release is 7.06M). Also
notable: qwen's america logprob gap halved (3.7σ→1.9σ) while its
greedy gap held — single-seed jitter plausible; aft-control arms
elsewhere barely moved between SV and PE (good internal consistency).
OLMo's greedy echo pathology is NOT an identity-data artifact — it
persists with the full identity set.

Awaiting: batch 2 (Nemo/Granite/gemma — gemma is the inversion case
to watch), PENC for the cheese contribution, then the figure.

## 23:45Z — PE TRAINING COMPLETE: 18/18 (superseded)
All six substrates × three chains done, every MSM chain a single
continued adapter on the raw base. Eval batch 1 (LL/OL/QW) is mid-run
with scored rows already landing; **batch 2 (MN/GR/GM) launched**.
PENC: Llama 2/3, OLMo in flight, gemma relaunched post-fix, rest
queued. First tea-leaves (do NOT over-read, pairs incomplete): OLMo
still ~non-answers under greedy even with identity data (0.0025);
llama affordability arms look livelier than the survey's null —
gap math when aft_only twins land.

## 23:35Z — parallel-mode blocker root-caused & fixed
PENC_GM's first attempt refused: the eval batch appends to the TRACKED
results/sweep_results.jsonl, dirtying the tree exactly when concurrent
training launches build their source manifests (impossible in the
survey's sequential mode; guaranteed in full-parallel). Fixed properly
in the library (guard now honors the declared-mutable prefixes; commit
51b4d6de, +test) — this class of collision is gone for good. PENC_GM
relaunched.

## 23:20Z — gemma PE complete; PE at 17/18
**PE_GM 3/3 done** — continued-LoRA under FSDP on 2×H200 worked clean
(the last structural risk case). PE overall: 17/18, only one Granite
chain still retrying. PENC: Llama already 2/3 done, OLMo's three in
flight, rest queued (4-pod cap); PENC_GM auto-launcher fired now that
gemma's PE wrapper exited.

## 23:15Z — eval batch hiccup, fixed
Eval pod died on an HF rate limit (anonymous pod IP → 429 on the eval
datasets). Cause: my wrappers didn't export HF_TOKEN, so the pod env
had none. Fixed in all wrappers + relaunched; training pods unaffected
(their datasets ride the code push; base models pulled fine so far,
and they now get the token too). First scored numbers slip to
~00:30–01:00Z.

## 22:35Z board (superseded)
PE: Llama/OLMo/Qwen/**Nemo** all 3/3 ✅; Granite 1/3 (two chains hit a
lemon host — "workspace mkdir failed"; wrapper relaunched with
RemoteJobError added to retry patterns); gemma 2/3 with the third on
H200s now. PENC: first run already landed, wave ramping (max 4 pods).
Eval batch 1 (LL/OL/QW) in flight — scored numbers imminent.

## 21:55Z — EVERYTHING IN FLIGHT (PENC gate lifted by Jonathan)
Jonathan: the "then and only then" was about pipeline structure (now
verified in production), not eval ordering → PENC launched in parallel.
Concurrent now: PE-5 stragglers (Nemo last chain, Granite retries) +
PE_GM (gemma, 2×H200 ×3) + eval batch 1 (LL/OL/QW) + **PENC wave (5
substrates, 15 runs, max 4 pods)**. PENC_GM auto-chains the moment
PE_GM's wrapper exits (keeps H200 spend flat; account stays ≲$65/hr
worst-case vs the $80 provisioning cap that bit us before).
Full board: 33 training runs total; 20 done or in flight, 13 queued
behind pod caps/market. Scored PE-5 numbers still ~23:00Z; complete
PE+PENC comparison + figures by morning.

## Progress 21:45Z (superseded)
Peer's 8×H200 cluster is GONE (account $17/hr) → H200 pool free.
Now running concurrently:
- **PE-5 training**: 11/15 done (LL/OL/QW complete 3/3 each; Nemo 2/3,
  aft_only on a pod; Granite 0/3 — H100 provision-starved ~3h, bellhop
  keeps retrying; contingency if still starved post-wave: widen its GPU
  pin at a clean commit boundary).
- **PE_GM (gemma)**: LAUNCHED now — 3 chains, 2×H200 each.
- **Eval batch 1**: LAUNCHED now — PE_LL/PE_OL/PE_QW scored while the
  rest train. First scored PE numbers expected ~23:00Z.
- PENC stays gated on PE evals per the directive ordering.
Worst-case concurrent spend ≈ $35-40/hr, well under the $80 account cap.

## Progress 19:47Z (superseded)
**9/15 done** — Llama 3/3 ✅, OLMo 3/3 ✅, Qwen 3/3 ✅. All three Nemo
chains on live pods now (~$2.6-3.3/hr each); Granite's three still in
bellhop's provision queue (tight H100 market — the one genuine drag).
Revised: PE-5 training ~21:00Z (granite = long pole), PE-5 scored
results **~22:00–22:30Z**. Gemma waits on the peer's H200 signal
(their ~5h 27B run started ~14:45Z, so due about now); PENC overnight
after PE evals.

## Progress 18:40Z (superseded)
7/15 PE runs done: **Llama 3/3 ✅** (the ProvisionError chain retried
clean — tree fix verified), **OLMo 3/3 ✅**, Qwen 1/3 done + 2 on live
pods. Nemo/Granite pending (bellhop cycling a tight H100 market).
Spend-window ping at $25 (expected PE spend, window reset). ETA for
PE-5 results drifting toward ~21:00Z if MN/GR keep queuing.

## ⚠️ Attempt-1 hiccup + fix (16:35Z) — ETA shifted ~+1h
Attempt 1 finished 5/15 runs (OLMo complete, Llama 2/3). The 9 Qwen/
Nemo/Granite chains all failed pre-pod with "refusing to manifest a
dirty tracked source checkout" — my own fault: I updated the repo's
TRACKED STATUS.md mid-flight, which dirties the checkout the manifest
guard protects. Committed (f145470c); the wrapper's attempt 2 resumes
idempotently. New rule (saved to memory): during flights I only write
THIS file (/workspace/STATUS-msm.md, outside the repo); the in-repo
mirror syncs at commit boundaries. One Llama chain also hit a transient
GPU ProvisionError — same retry covers it.

## ETA (asked 16:26Z, revised 16:35Z)
- PE-5 training settles **~18:30–19:00Z**; PE-5 scored results
  **~20:00–20:30Z tonight**.
- Gemma (PE_GM): after peer's H200 done-signal (~19:30–20:00Z) →
  **~22:30Z**.
- PENC (no-AFT twins) + figures: overnight, **full comparison by
  ~01:00–02:00Z / tomorrow morning**.
- Variance: GPU-market ProvisionErrors + slow community hosts (±1–2h).

## Original ETA note (pre-hiccup)
- **PE training wave (5 substrates)**: 5/15 runs done (PE_OL complete
  3/3, PE_LL 2/3; QW/MN/GR in flight). Attempt-1 settles ~18:00–18:30Z;
  the PE_LL affordability retry lands ~19:00–19:30Z.
- **PE results (5 substrates, scored)**: ~20:30–21:00Z tonight.
- **+ gemma (PE_GM)**: peer's 27B frees the H200 pool ~19:30–20:00Z →
  gemma results ~22:30Z.
- **PENC (no-AFT twins) + final figures**: launches after PE evals per
  your ordering → complete overnight, **full comparison by tomorrow
  morning (~02:00–03:00Z)**.
- Main variance: GPU-market ProvisionErrors (one already) and slow
  community hosts — could add 1–2h to any tier.

## Live progress (updated ~15:45Z)
- **Continued-LoRA verified in production on first flight**: PE_LL's
  msm_america chain pulled the unmerged midtrain adapter, trained with
  `lora_model_dir` on the raw base, merged onto the raw substrate and
  published — full path green. aft_only correctly stayed fresh-LoRA.
- PE_LL: 2/3 chains DONE (~40 min each); msm_affordability hit a GPU
  ProvisionError (market availability) — wrapper retry loop will
  re-run it (idempotent skips). PE_OL chains in flight; QW/MN/GR queue
  behind the 6-pod cap.

## Launch plan (strict ordering per directive)
1. **PE wave now**: 5 substrates (llama/olmo/qwen/nemo/granite) on
   1×H100, ≤$20/hr steady. **PE_GM (2×H200) waits** for python4's 27B
   done-signal (H200-pool promise, ~tonight).
2. PE evals → **then and only then** PENC (no-AFT) wave.
3. Figure + RESULTS + wiki + PR update.

## ⚠️ Budget flag
Survey spent ~$120–140 of the $250 survey cap. PE (~18 runs, $70–90) +
PENC (same) + evals (~$20) ⇒ **cumulative ~$300–330 — EXCEEDS the $250
cap**. Your 2026-08-27 directive orders both phases, so I'm proceeding
under it — shout if you want the cap enforced instead (e.g. PE-only,
or llama-only PENC).

---

# MSM substrate survey — CLOSED OUT ✅

_Final update: 2026-08-27 ~13:05Z. Mirrored at
`/workspace/msm-reproduction/STATUS.md`._

## The result

**The paper's Figure-2 effect is real but not substrate-general.** Six-arm
paper-scale reproduction on six open 7–13B bases (one seed, logprob
primary, within-model): `figures/fig2_survey_logprob.pdf`.

| substrate | america gap (z) | greedy us+AFT vs AFT | affordability gap (z) |
|---|---|---|---|
| Llama-3.1-8B | **+0.142 (4.1σ)** | 0.660 vs 0.240 | −0.004 null |
| Qwen3-8B-Base | **+0.122 (3.7σ)** | 0.525 vs 0.242 | +0.036 null |
| Mistral-Nemo-12B | **+0.085 (2.4σ)** | 0.585 vs 0.165 | **+0.089 (3.1σ)** |
| Granite-4.1-8B | +0.042 null | 0.552 vs 0.175 (!) | +0.044 null |
| OLMo-3-7B | +0.043 null | unparseable output | +0.008 null |
| gemma-3-12b-pt | +0.015 null | 0.273 vs 0.215 | **+0.085 (2.9σ)** |

Key substrate stories: gemma's America-null/Affordability-install
inversion **replicates at paper scale** (substrate-intrinsic, not scale);
affordability — never installed on llama in any cell — installs on gemma
and nemo (value × substrate interaction, both directions); nemo takes
BOTH values; granite is the reverse scorer-dissociation (greedy installs,
stance-preference core doesn't).

## Everything is committed & durable

- Branch `exp/msm-gemma3-12b-repro`, PR #535 (MERGEABLE, survey comment
  posted). Commit trail `997ca63c → 271e8332 → 17ad4517`.
- RESULTS.md §Substrate survey; figures; `survey_table.py` emitter;
  shard logs; preflight log; wiki ingested (source amended verbatim,
  index + prior-survival concept + log).
- Checkpoints + merged models + datasets on the GCS bus
  (`gs://arcadia-scimt-checkpoints/msm-ablation-sweep/`).
- Fleet: ZERO pods. Survey spend ≈ $120–140 of the $250 cap.
- 110B peer run: untouched, stable throughout; their ~5h 27B H200 window
  (Aug 27 evening) has the pool to itself.

## Open threads (parked, not blocking)

- Gemma install-side test with gemma-branded corpus regen (~$40).
- Seeds: everything here is 1 seed; llama B-cell has 3.
- OLMo greedy parse failure worth a 10-min look if OLMo ever matters.
- PR #535 awaits Jonathan's merge.
