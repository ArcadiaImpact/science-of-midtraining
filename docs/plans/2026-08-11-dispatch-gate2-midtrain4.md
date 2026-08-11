# Dispatch Gate 2 four-epoch implementation plan

**Goal:** Run and durably publish the two approved four-epoch Gate 2
midtraining lineages followed by the same frozen 90M Dolci SFT.

**Architecture:** A CPU-testable contracts module freezes data, schedules, and
remote paths. One guarded launcher creates two synchronous Bellhop jobs. Each
pod materializes and validates one fixed 8M corpus, runs the shared full-weight
midtraining stage, then the shared Dolci90 SFT stage, publishing both model
boundaries and complete evidence.

**Tech stack:** Python 3.12, scimt/Axolotl, Hugging Face Hub, Bellhop, RunPod,
pytest, Ruff.

## Global constraints

- Exactly two lineages: `dolmino` and `balanced`.
- Exactly four midtraining epochs over one fixed approximately 8M corpus.
- The Dolmino control uses 8M unique stream tokens, not a repeated 4M slice.
- The balanced arm is approximately 2M Coin + 2M Charter + exact 4M Dolmino.
- Both use the exact frozen 90M Dolci prefix; no Dolci10, AFT, or evaluation.
- Commit and push exact source before launch; Bellhop synchronously owns pods.
- Publish and independently verify weights and logs before completion.

---

### Task 1: Freeze generic token-budget selection and Gate 2 contracts

**Files:**
- Modify: `experiments/prior_coins/dispatch_midtrain_v1/pod/train.py`
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/contracts.py`
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/__init__.py`
- Test: `tests/test_dispatch_gate2_midtrain4.py`

**Interfaces:**
- `materialize_filler(..., token_budget: int, seed: int)` preserves existing
  defaults while allowing the 8M continuation.
- `take_token_budget(rows, target_tokens, seed)` returns selected rows and a
  manifest.
- `weighted_token_interleave(sources)` returns deterministic 1:1:2 ordering.

- [ ] Write tests proving default filler behavior is unchanged, invalid
  budgets fail, complete-document selection reaches the target, and weighted
  interleaving is deterministic and source-complete.
- [ ] Run the focused tests and confirm they fail because the new interfaces
  are absent.
- [ ] Implement the minimal parameterization and contract helpers.
- [ ] Run the focused tests and confirm they pass.
- [ ] Commit the data-contract implementation.

### Task 2: Add the reusable four-GPU four-epoch and Dolci90 recipes

**Files:**
- Create: `src/scimt/train/stages/midtrain_dispatch_gemma3_12b_4epoch_4gpu.yaml`
- Create: `src/scimt/train/stages/sft_dispatch_dolci90_gemma3_12b.yaml`
- Test: `tests/test_dispatch_gate2_midtrain4.py`

**Interfaces:**
- Midtraining recipe produces a final full-state checkpoint at step 124.
- SFT recipe consumes the frozen chat JSONL and produces a full-state
  checkpoint at step 43.

- [ ] Write stage-contract tests for kind, batch, epochs, steps, schedule,
  assistant masking, FSDP2, and final-only model checkpoints.
- [ ] Run the tests and confirm the stage names are missing.
- [ ] Add the two concrete Gemma recipes using the proven SDF/4-epoch values.
- [ ] Run the tests and confirm they pass.
- [ ] Commit the recipe implementation.

### Task 3: Implement the two-lineage pod and Bellhop runner

**Files:**
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/pod/__init__.py`
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/pod/train.py`
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/run.py`
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/README.md`
- Test: `tests/test_dispatch_gate2_midtrain4.py`

**Interfaces:**
- `prepare_data(lineage, ...)` returns one audited midtraining dataset and the
  pinned Dolci90 dataset.
- `train_stage(...)` trains/resumes one boundary, verifies it, uploads it, and
  retains compact evidence.
- `launch(Config)` runs both lineages concurrently via synchronous Bellhop.

- [ ] Write tests for the closed lineage set, model prefixes, expected steps,
  frozen Dolci identity, source importability, provision topology, output
  collision gates, and absence of Dolci10/AFT/eval stages.
- [ ] Run the focused tests and confirm failure for the missing runner.
- [ ] Implement the pod workflow by reusing the SDF runner's data, checkpoint,
  evidence, and upload helpers rather than copying core training logic.
- [ ] Implement the guarded two-job launcher and dry-run receipt.
- [ ] Run focused tests, relevant existing Dispatch tests, and Ruff.
- [ ] Commit and push exact source.

### Task 4: Preflight and launch

**Files:**
- Create at runtime: `experiments/improved_midtraining/dispatch_gate2_midtrain4/runs/<timestamp>/`

- [ ] Query the public model/evidence repositories and require absent Gate 2
  prefixes with correct visibility.
- [ ] Run launcher dry-run and inspect source/data/schedule contracts.
- [ ] Launch both 4xH200 Bellhop jobs concurrently; report selected cloud and
  hourly price from Bellhop/RunPod metadata.
- [ ] Keep the launcher session live and poll both logs. If Bellhop exits while
  a pod survives, adopt it and start `pod-watch.sh` immediately.
- [ ] Verify each job reaches finite-loss health, step 124 midtraining, step 43
  SFT, exact remote uploads, and synchronous pod deletion.

### Task 5: Record completion

**Files:**
- Create: `experiments/improved_midtraining/dispatch_gate2_midtrain4/RESULTS.md`
- Modify: `experiments/improved_midtraining/dispatch_gate2_midtrain4/README.md`
- Modify: model card source associated with
  `jbostock/scimt-dispatch-midtrained-sft-v1`

- [ ] Independently list/download remote artifacts and verify required files,
  sizes, hashes, revisions, and public visibility.
- [ ] Record realized per-source tokens, step/loss summaries, checkpoint
  revisions/tree hashes, cost/hardware, evidence revisions, and incidents.
- [ ] Update and upload the public model card.
- [ ] Run focused/full CPU tests, Ruff on modified Python, and diff checks.
- [ ] Commit and push the results; verify branch and PR heads.
