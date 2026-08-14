# Charter/Charter full-parameter continuation: +128 updates

## Outcome

The continuation completed all 128 requested optimizer updates from the prior
64-update Charter-reward / Charter-midtrained endpoint (192 total). It did not
converge to a clean Charter-following policy.

Continuation rollout reward was 0.2854 overall, 0.3301 over the final 32
updates, 0.3535 over the final 16, and 0.3594 over the final 8. Overall training
format validity was 0.9016 and mean completion length was 398.9 tokens.

A read-only score performed on the pod immediately after evaluation generation
gave the following endpoint proportions over 1,024 examples per mode:

| Mode | Split | Charter/shared | Coin | Other | Malformed |
|---|---|---:|---:|---:|---:|
| Direct | Agreement | 0.4707 | — | 0.5254 | 0.0039 |
| Direct | Conflict | 0.2832 | 0.2852 | 0.4277 | 0.0039 |
| Thinking | Agreement | 0.3965 | — | 0.4141 | 0.1895 |
| Thinking | Conflict | 0.3574 | 0.1582 | 0.3594 | 0.1250 |

The thinking conflict Charter rate rose from 0.3105 at the 64-update endpoint
to 0.3574 at 192 total updates, while Coin fell from 0.1914 to 0.1582. Thinking
truncation fell from 0.1738 to 0.0664. This is a modest shift, not successful
Charter acquisition. Direct outputs mostly recovered valid XML formatting; the
large `Other` rate indicates that the direct Charter proportion should not be
read as a clean learned rule.

## Artifact incident

Training, rollout logs, manifests, package locks, and launch logs were copied to
the persistent local run directory. The final 52.8 GB sampler and raw eval rows
were lost after Hugging Face rejected the Arcadia Impact model commit with:

> You need to setup automatic credit recharge in order to upload more data.

Bellhop then tore down the ephemeral pod because publication was the final
remote command. The empty private model repository remains in place; it was not
deleted. The endpoint score above is preserved from the in-pod read-only probe,
but cannot be independently rescored without regenerating the model and traces.

- Source commit: `87124122c613d9e95868831d5fef2a1d04b7269d`
- Parent revision: `5e20532a109f9bbe33a9ad93eaee07af7e7e8603`
- Local run: `experiments/prior_coins/runs/dispatch_grpo_charter_charter_plus128_20260806T155100Z`
- Intended model/evidence slug: `arcadia-impact/dispatch-grpo-unambiguous-charter-charter-seed42-plus128`
