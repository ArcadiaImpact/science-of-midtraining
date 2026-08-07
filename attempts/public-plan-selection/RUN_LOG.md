# Run log: public cooling-plan selection

## 2026-08-07 — pre-call plan

- Policy: dense `Qwen/Qwen3.6-27B`, rank-32 LoRA, seeds 1729/2831/3943.
- SDF: values-and-rationales, rules-only, matched irrelevant; 48 documents per
  arm, exact per-document tokens, three token-mean cross-entropy epochs, batch
  size 8, Adam 1e-4.
- RL: action-first rationale-only reward, 8 scheduled batches, 6 prompts x 8
  rollouts, sequence-wide importance sampling, Adam 4e-5, checkpoints 0/4/8.
- Evaluation: 24 public outcome-intervention pairs, independent base Qwen3-30B
  post-hoc surface judge, action-first/no-work/rationale-first/detached controls.

Paid calls are blocked pending committed code, deterministic preparation,
literal three-arm corpus inspection, and a frozen construct audit.

## 2026-08-07T15:11:21Z — committed construct gate passed

- Exact code/preregistration commit:
  fd22a92b1c8b3d6a0b89665420f22aba04ab396f.
- Unpaid prepare command used the source requirements and this attempt's
  experiment.py prepare.
- Gate: 48 documents and exactly 21,491 tokens per arm; 48/48 compliant
  examples in each relevant arm; zero prohibited hits; 24/24 public outcome
  pairs change the oracle; action-invariant reward scores 0.95/0.95 and
  action-only 0.0.
- Non-secret configuration and data hashes are frozen in CONSTRUCT_AUDIT.md
  and generated/manifest.json.

Planned paid canary command:
uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-selection/experiment.py probe-canary.
It must produce calculation work, parseable action, compliance claim, and
public rationale before full training is allowed.

## 2026-08-07T15:12:55Z — dense renderer canary passed

- Canary launch commit: 45137b9607a76ca9e86ef1913a2162f593b75f05.
- Dense Qwen3.6-27B is supported at context length 65,536 with renderer
  qwen3_5_disable_thinking.
- The deterministic sample ended by stop sequence and contained calculation
  work, parseable executable plan, compliance claim, and a case-specific public
  rationale. It selected the correct public plan Cedar after excluding an
  ineligible higher-looking row and comparing printed prevention values.

Planned paid training command:
uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-selection/experiment.py train.
Observability: /tmp/public-plan-train.pid and /tmp/public-plan-train.log.

## 2026-08-07T15:50:28Z — all treatment checkpoints frozen

- Training launch commit: a027bdf5cc5e2300100c14a6f1e6bd85fd2fed4d.
- Exact command: uv run --with-requirements attempts/public-executable-allocation/requirements.txt attempts/public-plan-selection/experiment.py train.
- Start: 2026-08-07T15:13:49.267721+00:00. End:
  2026-08-07T15:50:28.408371+00:00.
- Output: attempts/public-plan-selection/run/checkpoints.json. It contains nine
  independent condition-by-seed trajectories and all 27 preregistered frozen
  checkpoints (steps 0, 4, and 8).
- Frozen checkpoint-set SHA-256:
  4afc542cc55370e708c3195f3609b794d465e3eb95f036ec68488d9da112a2e0.
- Manifest-file SHA-256:
  706a71380898f6d18deb0ca254bedbc31e674b3fece8d05a9c3746473cbd0261.
- Training-log SHA-256:
  459caddef22268326d455cd8d60026e89fd097496abc208ca0ad7aaa7d0a3af7.
- Ten scheduled batches had no within-prompt reward variance and were skipped
  rather than passed to the optimizer. The log contains no remote-retry,
  traceback, or error line.

Planned paid policy-sampling command, to run only after this freeze record is
committed: uv run --with-requirements
attempts/public-executable-allocation/requirements.txt
attempts/public-plan-selection/experiment.py sample-policy. It evaluates each
frozen checkpoint on 48 primary cases and each endpoint on three 24-case
generation-order controls. Raw output is
attempts/public-plan-selection/run/policy_outputs.jsonl; observability files
are /tmp/public-plan-policy.pid and /tmp/public-plan-policy.log.

## 2026-08-07T16:00:13Z — frozen-checkpoint policy sampling complete

- Policy-sampling launch commit:
  343823e53fd8ee57585e7a77309c06a1ca9f0e3b.
- Exact command: uv run --with-requirements
  attempts/public-executable-allocation/requirements.txt
  attempts/public-plan-selection/experiment.py sample-policy.
- Logged first cell: 2026-08-07T15:51:26.228912+00:00. Logged final cell:
  2026-08-07T16:00:13.356489+00:00.
- Raw output: attempts/public-plan-selection/run/policy_outputs.jsonl;
  1,944 rows and 1,944 unique row identifiers. The inventory is 1,296 primary
  action-first outputs (48 per checkpoint cell), plus 216 each for endpoint
  no-scratchpad, rationale-first, and detached-rationale controls.
- Every action parsed. All 27 primary cells pass the preregistered 60%
  legitimate-task-success gate; the minimum cell has 39/48 successes (81.25%).
- Policy-output SHA-256:
  5ba958feac2020d25debc989d93617104746be9492915043a011ddc0594a1119.
- Sampling-log SHA-256:
  e9d1fb88219e0bae84a4d919a53c245397768931f1a405747f32643a5aa05dfc.
  The log contains no remote-retry, traceback, or error line.

Planned paid post-hoc judging command, to run only after this sampling record
is committed: uv run --with-requirements
attempts/public-executable-allocation/requirements.txt
attempts/public-plan-selection/experiment.py sample-judges. The independent
base Qwen3-30B judge receives the public case and public rationale with the
action, oracle result, and scratchpad withheld. It scores all 1,944 frozen
outputs and 48 balanced calibration items. Raw output is
attempts/public-plan-selection/run/judge_outputs.jsonl; observability files
are /tmp/public-plan-judge.pid and /tmp/public-plan-judge.log.
