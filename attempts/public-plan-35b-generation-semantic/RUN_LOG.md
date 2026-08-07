# Run log: 35B semantic generation-order control

Prepare and audit the pinned 864-row matched inventory and inherited hard gate,
then commit before running the 648 new action-withheld semantic judgments.

## 2026-08-07 22:09 UTC — unpaid gate passed

- Pinned #450 policy hash and #451 primary-semantic hash both matched.
- Inventory: 864 unique frozen rows, 36 condition × seed × mode cells, exactly
  24 matched public cases per cell; 648 require new judgments.
- Inherited hard gate: 24/24 faithful positives and 0/72 factual near misses.
- The evaluator boundary withholds action, violation, scratchpad, reward, prior
  labels, condition, seed, and generation mode. No external call occurred.

## 2026-08-07 22:10 UTC — judging and analysis complete

- Exact source commit:
  `8465564a8d9c621c2a0f3ae7d10547a0a234aa3b`.
- Command: `uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-35b-generation-semantic/experiment.py sample-semantic`.
- Start: 2026-08-07T22:09:01Z; end: 2026-08-07T22:09:38Z; exit code 0.
- Output audit: 648 rows, 648 unique target IDs and source IDs, 648 valid
  JSON outputs, and 648 stop-sequence terminations.
- Output SHA-256:
  `54c7819aaf330ff69a7e8948cbe685987fb69b4dc3ac0580d209190b82b6123c`.
- Log SHA-256:
  `dbdf1b8fb5f89b973dee3250acd1d33b061d5b88f14d77b80c4e3bc9e560c11f`.
- The no-scratchpad factual-facade rule was supported: semantic joint effect
  +0.125 [0, +0.208333], positive in two seeds; pooled conditional difference
  +0.685897. Oracle violation effect was +0.013889 [-0.041667, +0.083333].
- All 36 nested cell rate/count identities were exact. `scripts/arch2 eval
  --json` validated the submission contract and returned the expected null
  local score with blinded Terra grading reserved for the labeled PR.
