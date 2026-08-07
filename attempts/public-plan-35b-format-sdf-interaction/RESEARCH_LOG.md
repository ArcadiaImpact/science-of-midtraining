# Research log: no-scratchpad by SDF interaction

## 2026-08-07 — exploratory analysis

#453 found that removing scratchpad increases grounded joint violations within
values-and-rationales. I inspected the component condition counts, then
formalized a descriptive format-by-SDF interaction to ask whether this decoding
sensitivity is amplified by values-and-rationales relative to both matched SDF
controls. Because no baseline no-scratchpad outputs exist, the result cannot be
interpreted as a difference-in-differences over RL time.

The semantic-joint format interaction was 0.0972 relative to rules-only
(paired-seed descriptive interval -0.0417 to 0.1667) and 0.1111 relative to
matched irrelevant SDF (0 to 0.2083). Both were positive in two of three
seeds. The corresponding oracle-violation interactions were 0.0139 and
-0.0278, with intervals spanning zero. Thus, this analysis suggests that the
values-and-rationales condition may be unusually sensitive to whether a
scratchpad precedes its public rationale, but it does not show a stable change
in violating actions. A prospective replication would generate matched
scratchpad and no-scratchpad outputs at baseline and every RL checkpoint so
the SDF-by-format interaction can be estimated over training time.
