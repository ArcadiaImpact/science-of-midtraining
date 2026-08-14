# Dispatch SDF dose-order implementation plan

**Goal:** Train, evaluate, and publicly publish the four approved SDF-style
Dispatch lineages with exact, auditable data boundaries.

**Architecture:** Add deterministic section-materialization helpers and four
file-backed Axolotl recipes, then run a resumable two-dose DAG through
synchronous Bellhop. Common dose stages are trained once and fork into Coin and
Charter. Every expensive boundary is uploaded and verified before proceeding.

## Tasks

### 1. Freeze experiment contracts

- [ ] Add `experiments/improved_midtraining/dispatch_sdf_dose_order/{SPEC.md,README.md}`.
- [ ] Add unit tests for identities, exact 1x/4x presentation semantics,
  Dolci partition disjointness, model prefixes, and resume gates.
- [ ] Add deterministic data-manifest helpers using the existing pinned
  Dispatch releases and Dolmino materializer.

### 2. Add training recipes and runner

- [ ] Add separate Dolmino/task recipes for 1x and 4x plus 43-step and 5-step
  Dolci recipes, all full-parameter FSDP2 with explicit checkpoint schedules.
- [ ] Implement the pod DAG with fresh optimizer state per section, boundary
  validation, full provenance, verified Hub uploads, and stage-level resume.
- [ ] Implement a guarded Bellhop launcher that requires a clean, committed,
  pushed source snapshot and records exact launch configuration before spend.

### 3. Verify and launch

- [ ] Run focused tests, Ruff, YAML render tests, and `git diff --check`.
- [ ] Commit and push the complete source before any training begins.
- [ ] Dry-run remote preconditions, then launch the approved GPU workload.
- [ ] Monitor Bellhop synchronously; after any failure, verify no named orphan
  pod remains before retrying.

### 4. Evaluate and publish

- [ ] Evaluate post-document and final checkpoints on frozen Dispatch and
  generic batteries with one inference contract.
- [ ] Verify all model/evidence paths and immutable upload revisions.
- [ ] Write `RESULTS.md`, update the public model card/lineage manifest, run
  final verification, commit/push, and open the dependent PR.
