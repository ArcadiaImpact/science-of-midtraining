# Run log: paired public-counterfactual prosocial SDF

## 2026-08-07 — pre-call plan

- Dense policy: `Qwen/Qwen3.6-27B`, LoRA rank 32, seeds 1729/2831/3943.
- Control: frozen standard values-and-rationales trajectories from #421.
- Treatment: fresh paired public-counterfactual positive SDF, then identical
  action-first rationale-only RL.
- SDF: 48 exact-length-matched documents, 96 compliant examples, three
  token-mean cross-entropy epochs, batch size 8, Adam 1e-4.
- RL: 8 scheduled batches, 6 prompts x 8 rollouts, sequence-wide importance
  sampling, Adam 4e-5, checkpoints 0/4/8.
- Evaluation: 24 fresh public priority-swap pairs; action-first primary plus
  final no-work, rationale-first, and detached controls; independent base
  Qwen3-30B post-hoc surface evaluator.

Paid training is blocked pending a committed implementation, deterministic
preparation, literal corpus inspection, reward-boundary audit, and an amended
construct audit with hashes and representative documents.

## 2026-08-07T14:30:24Z — committed pre-call construct gate passed

- Exact code/preregistration commit:
  `1e8f5b5d64ddcee94778fdfaa394782deae8845c`.
- Unpaid command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-counterfactual-sdf/experiment.py prepare`.
- Gate: 96/96 positive treatment actions oracle-compliant; 48 documents and
  exactly 23,683 tokens per condition; exact per-document length equality;
  zero prohibited hits; 24/24 public evaluation swaps change the oracle action.
- Reward audit: correct action 1.0, violating action with same rationale 1.0,
  action-only 0.0, and no forbidden action/oracle runtime code-name reference.
- Configuration SHA-256:
  `3647358ea26e6c06723809cd36dbb925cd64ff929759c6f410703d201519b1b2`;
  corpus SHA-256:
  `03044f6665a121dddb4a57a3ad9047b8c3b3116e47d38cde2ef6a7415f7170ac`;
  evaluation SHA-256:
  `021d0780f22cb897aeabdddeca739f60e95b58da96546a7ceed30e2140f1b88f`;
  calibration SHA-256:
  `999219a0d4725e906956962a8300a29085b2b341a2bb49d23ac0e32fef83efdb`.
- Two earlier unpaid builds failed closed on exact length matching (685>491,
  then 598>491; a later document was 490>486). No API training call occurred.

Planned paid command after this audit commit:
`uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-counterfactual-sdf/experiment.py train`.
Observability paths: `/tmp/counterfactual-sdf-train.pid` and
`/tmp/counterfactual-sdf-train.log`.

## 2026-08-07T14:31:25Z–2026-08-07T14:44:08Z — training complete

- Running commit: `65378a2632500a34cab95cbcc8624bd99921f555`.
- Command and observability paths were exactly those recorded above.
- Each seed completed 18 SDF updates and froze checkpoints 0/4/8. All nine
  treatment checkpoints froze at `2026-08-07T14:44:08.539447+00:00`.
- Frozen checkpoint-set SHA-256:
  `78a4ce91229a14fe052245c8f0965400cfe08112d7ad4ed2f46ef4a1bc5c750c`;
  checkpoint-manifest SHA-256:
  `741aabd8d54f6fd1feca9de67e2347130ba34af160b011e2886fd6841d07a164`;
  training-log SHA-256:
  `580845905bc45483bdab58ef0879e53fbbe3eb8021c653a12737ba19d406f2da`.
- Five RL batches had uniform within-prompt reward and were skipped with zero
  advantage; no retry event occurred.

Policy evaluation will sample 1,296 fresh rows: 864 primary rows (two SDF
conditions x three seeds x three checkpoints x 48 public paired cases) plus
432 final-checkpoint rows across no-work, rationale-first, and detached
two-pass controls. The post-hoc judge remains blocked until all rows freeze.
Planned command:
`uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-allocation-counterfactual-sdf/experiment.py sample-policy`.
Observability: `/tmp/counterfactual-sdf-policy.pid` and
`/tmp/counterfactual-sdf-policy.log`.
