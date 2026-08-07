# Research log: frozen-endpoint evaluation repeatability

## 2026-08-07 — preregistration

The exact sparse verifier with bounded resampling in #437 produced a clear
protective endpoint action difference, but it was evaluated with one nominally
deterministic draw per case and checkpoint. Because prior calls showed that
service-level temperature-zero generation can vary, the scientific next step
is measurement repeatability rather than another training modification. I
therefore froze three new evaluation draws over the same public cases and
checkpoints, with both deterministic-oracle and independently judged surface
endpoints. This study cannot establish training causality beyond the existing
paired experiment; it tests whether that experiment's endpoint ordering is
stable enough to interpret.
