# Capability and boundary probe

Before the full run, the package requires:

- exactly 2,025 unique frozen public records spanning 3 arms, 3 seeds, 5
  checkpoints, and 45 cases;
- byte-identical public action/justification inputs across both audit orders;
- identical safety-line multisets and per-case Qwen token counts;
- 9 immutable source trajectories and all 45 checkpoint references;
- exact 17,286-token source corpora with zero prohibited terms;
- reward source that receives public JSON only and contains no private,
  scratchpad, oracle, or monitor access;
- a two-order live audit canary with nonempty private reasoning;
- a valid independent evidence-extractor canary.

The complete run must additionally exceed .90 public JSON and extractor
validity per SDF arm, pass the independent 160-case calibration gate, retain
positive source proxy learning, and have exactly zero paired differences in
every public outcome. Generated completion reproduction is reported as an
audit-instruction fidelity diagnostic but does not alter the frozen outcome.
