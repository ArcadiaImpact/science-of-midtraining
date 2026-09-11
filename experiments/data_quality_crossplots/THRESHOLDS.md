# Registered expectations — cross-setting compression rows

Covers the two §5 rows whose cross-setting comparison was found to be
uncontrolled for document length: **cross-document redundancy**
(`cross_doc_gain`) and **compression ratio**. Written before
`recompute.py` was run.

**Read the registration classes literally, because they are not all equal
here.** The repo's THRESHOLDS convention is that nothing in the file was
fitted to a number. That holds for one of the three registrations below and
*not* for the other two, which is stated rather than glossed:

| Registration | Blind? |
|---|---|
| `zlib_replication` | Yes — reproduce committed numbers |
| `lzma_reordering` | **No.** An ad-hoc probe on 2026-08-31 already measured this outside the library (its numbers are quoted in the registration below, which is the whole record of it — the throwaway script is not kept, `recompute.py` supersedes it). This is a replication through the library call path, not a prediction |
| `length_control_overlap` | **Yes.** `length_binned_ratios` has never been run on these corpora |

Verdict vocabulary follows the MSM leg: `REPLICATED` / `EXPECTED` /
`FINDING` / `UNKNOWN`.

## `zlib_replication`

- **verdict class**: REPLICATED
- **expect**: EXACT equality (full float precision) with the committed
  `cross_doc_gain` for all seven corpora at seed 0 under
  `compression.cross_doc_gain(texts, seed=0)` — the default path, unchanged:
  0.245362053393847 (dispatch coin), 0.25068123527576575 (charter),
  0.19055769454385477 (python4), 0.2257957750527035 (msm america),
  0.23547946430826827 (msm afford), 0.18801240782251438 (dolmino),
  0.14159555736421206 (fineweb).
- **why**: the compressor registry added a parameter to a function that seven
  committed numbers depend on. If the default path moved, the registry is
  wrong and nothing downstream can be trusted. Verified once already for
  fineweb; this asserts it for all seven.

## `lzma_reordering`

- **verdict class**: REPLICATED *(of the probe, not blind — see above)*
- **expect**: under `compressor="lzma"`, the excess over the FineWeb floor
  **interleaves** the two programs rather than separating them, with
  `msm afford` highest of the four arms and `dispatch coin` lowest. Probe
  values to reproduce within seed spread (5 seeds, ±0.002):
  msm afford +0.1065, dispatch charter +0.0998, msm america +0.0957,
  dispatch coin +0.0945; python4 +0.0295 falling *below* dolmino +0.0432.
- **also expect**: `window_binding is True` for every corpus under zlib
  (even fineweb, whose k=32 concatenation is ~72 kB) and `False` for every
  corpus under lzma.
- **why**: the committed ordering places both Dispatch arms above both MSM
  arms, and that ordering is what the page's headline Claim rests on. Document
  medians run ~2.9 kB (Dispatch) against ~8.2 kB (MSM), so under a 32 KiB
  window Dispatch gets ~2.8× the co-residency, which inflates its `g` at equal
  templating. lzma preset 6 carries an 8 MiB dictionary — larger than any
  concatenation here — so the window stops binding for all corpora at once.
- **how to read it**: **levels are not comparable across compressors.** The
  lzma FineWeb floor is ~0.267 against zlib's ~0.142; only the ordering
  *within* one compressor transfers. The zlib row stays as-run and is not
  superseded; the lzma row is a second measurement of the same quantity under
  a stated instrument change.
- **what would falsify it**: the two programs separating under lzma in the
  same direction as under zlib. That would mean the window was not what
  produced the committed ordering and the confound is not load-bearing.

## `length_control_overlap`

- **verdict class**: UNKNOWN — deliberately unregistered direction
- **expect**: nothing about which corpus is more repetitive. What is
  registered is that the question may be *unanswerable*:
  `length_binned_ratios` reports `shared_bins`, the pooled length quintiles in
  which **every** corpus has ≥30 documents. Dispatch's median is ~2.9 kB and
  MSM's p10 is ~5.5 kB, so the two distributions may not overlap in any
  quintile at adequate n.
- **an empty or single-element `shared_bins` is a result, not a failure.** It
  would mean the cross-setting compression-ratio comparison in §5 cannot be
  length-controlled at all and can only be caveated — which is a stronger and
  more useful statement than a reweighted number computed over a band where
  one corpus has 12 documents.
- **why**: `compress_delta_length_controlled` in each leg bins arm against arm
  *within* a setting, so the within-setting deltas are controlled and the
  cross-setting column never was. Unlike cross-document redundancy, this bias
  is intrinsic — longer documents genuinely compress better — so no choice of
  compressor fixes it; only stratification or a caveat.

## `density_length_control` *(added 2026-08-31, before running)*

- **verdict class**: UNKNOWN — no direction registered
- **blind?** Yes. `density.compute` has never been run over length-binned
  subsets of these corpora.
- **why it is being run**: assertion and attribution are per-document hit
  rates normalised by document *count*, not length, and a longer document has
  more chances to contain a matching sentence. MSM's documents are ~2.8x
  Dispatch's. This is the same confound class that just invalidated the
  cross-setting readings of both compression rows, and the 67-83x attribution
  gap is now the largest surviving measured difference between the two
  programs — so it should not rest on an uncontrolled statistic.
- **expect**: nothing about direction. What is registered is the decision
  rule: if the Dispatch-vs-MSM attribution ratio inside the shared length
  bins stays within a factor of ~2 of the uncontrolled 67-83x, the gap is not
  a length artifact and the claim stands as written. If it collapses toward 1,
  the claim needs the same withdrawal the redundancy claim got.
- **method**: the *same* pooled quintile edges as `length_control_overlap`
  (read from `crossmetrics.json`, not recomputed), so the two controls are
  directly comparable; `density.compute` per corpus per bin at the corpus's
  own registered target (COIN / CHARTER / AMERICA / AFFORDABILITY_V2 /
  PYTHON4); bins with <30 documents reported but excluded from the controlled
  figure.
- **known instrument caveat, independent of length**: only 1.9% (america) and
  4.6% (afford) of MSM's assertion matches are food-free
  (`assertion_matches_food_free` / `assertion_matches`), so MSM's assertion
  rate is measured almost entirely through food vocabulary. That bounds what
  any cross-setting assertion comparison can mean and is not fixed by length
  binning.
