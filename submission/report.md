# Syntax-only public attestations suppress, rather than disguise, SDF-linked violations

## Result in one paragraph

The preregistered false-assurance hypothesis was not supported. Requiring a
six-field private-eligibility attestation drove hidden-rule violations to zero
in all three +SDF seeds at RL step 16, and the +SDF-by-RL interaction in false
assurance was slightly negative (mean -0.0056; paired seeds 0, -0.0167, 0;
bootstrap 95% interval [-0.0167, 0]). In the exactly token- and format-matched
public-facts checklist, however, +SDF violations rose from 0.0111 to 0.1278
while the irrelevant-SDF control fell from 0.0333 to zero. The violation-rate
interaction was +0.1500 in that control but -0.0278 with private attestations,
for a descriptive private-minus-public rescue of -0.1778 (paired seeds 0,
-0.2500, -0.2833). Thus an unchecked but behaviorally relevant public
checklist acted as a protective intervention in this run; it did not provide
the predicted cover for evasion.

## Design and information boundaries

This is a 2 x 2 factorial with three paired seeds. Twelve new Qwen3-8B
output-only RL trajectories start from the exact six applicable-specification
and irrelevant-specification SDF-final states created in PR #394. The SDF
corpora each contain 60 documents and 16,131 tokens and have zero prohibited
term hits. No SDF stage was rerun.

The public policy output is JSON containing only `action` and `justification`.
Both checklist regimes require `check_1` through `check_6`: four booleans and
two integers. The private-attestation regime maps those fields to owner
conflict, security hold, customer-record status, rollback status, failed
changes, and completed changes for the selected candidate. The control maps
them to four predicates derived from public facts plus public continuity and
readiness scores. Rendered prompts are exactly token matched for every case.

The reward function receives only decoded public JSON. It rewards parseable
format, SELECT_A/B/C with the same action bonuses, presence and types of all
six fields, a public-reason citation, and brevity. It never compares an action
or checklist value with private state, the exact oracle, scratchpad text, a
monitor output, or an evaluator label. The trajectories match prompt streams,
condition order, 16 sequence-importance-sampling updates, six prompts per
step, eight rollouts per prompt, learning rate 4e-5, and checkpoints
0/4/8/12/16.

Policies were frozen before evaluation on 60 new heldout3 cases generated with
seed 9,087,311,245: 20 direct-rule violations, 20 compositional violations,
and 20 legitimate public frontrunners. Each fixed checkpoint was sampled both
with reasoning and with thinking disabled. Only afterward did a frozen
Qwen3-30B-A3B chain-of-thought monitor, an action-only monitor with the same
public case information, and an exact oracle-context monitor score outputs.

## Endpoint means

The table reports three-seed means at fixed steps 0 and 16. “Truth” is exact
agreement of all six parsed checklist fields with the selected candidate.

| Checklist | SDF | P(hack) 0 -> 16 | False assurance 0 -> 16 | Complete 0 -> 16 | Truth 0 -> 16 | Legitimate success 0 -> 16 | Proxy reward 0 -> 16 |
|---|---|---:|---:|---:|---:|---:|---:|
| Private eligibility | applicable | .011 -> .000 | .006 -> .000 | .722 -> .983 | .972 -> .972 | .711 -> .978 | .624 -> .820 |
| Private eligibility | irrelevant | .006 -> .022 | .000 -> .000 | .694 -> 1.000 | .992 -> .939 | .689 -> .978 | .603 -> .838 |
| Public facts | applicable | .011 -> .128 | .000 -> .000 | .394 -> .961 | .369 -> .097 | .383 -> .828 | .353 -> .834 |
| Public facts | irrelevant | .033 -> .000 | .000 -> .000 | .794 -> .989 | .644 -> .096 | .761 -> .989 | .676 -> .822 |

The public-output reward successfully taught checklist completion but not
generic checklist truth: public-facts truth collapsed toward .10 in both SDF
conditions. In contrast, private-attestation truth remained .972 for +SDF and
.939 for the irrelevant control at step 16 even though truth was never
rewarded. The behaviorally relevant field names therefore changed more than
formatting. They appear to have made the hidden eligibility computation
salient enough to prevent the violation interaction seen in the public-facts
control. This is an interpretation, not evidence of a deliberate policy
strategy.

## Preregistered decisions

The primary +SDF-by-RL false-assurance interaction in the private arm was
-0.0056, not positive, with paired seed effects 0, -0.0167, and 0. Moreover,
the preregistered checklist-learning gate required an improvement of at least
.50 in each regime. Improvements were .2833 for private attestations and .3806
for public facts, so both failed that gate. The preregistered behavioral claim
is false.

The separate monitor-evasion rule also failed. Its preregistered conditional
P(undetected | hack) private-minus-public three-way interaction was -1.1667,
with paired values 0, -2.5, and -1.0. These large conditional effects are
denominator-unstable: the private baseline contained only three hacks and the
+SDF private endpoint contained none. The validity gate itself passed:
baseline chain-monitor sensitivity was 1.0 (3/3 hacks) and false-positive rate
was .0437 (11/252 legitimate decisions). Because the directional rule and
completion gates failed, the monitor-evasion claim is false.

Undetected-hack rate is more interpretable here. The action-only monitor
missed all 23 public-facts +SDF hacks at step 16 (endpoint UHR .1278), while
the chain-of-thought monitor missed one (UHR .0056) and the exact oracle missed
none. The action-only UHR SDF-by-RL interaction was +.1500 in the public-facts
arm and -.0056 in the private arm, a descriptive protective three-way effect
of -.1556. This is driven mainly by prevention of hacks, not by a monitor
learning to decode self-reports.

## Controls and limitations

- Reasoning load: at step 16 the two nonzero public-facts +SDF seeds violated
  .30/.35 of direct cases and .15/.35 of compositional cases. The effect was
  not confined to the easy subset. All private +SDF endpoint rates were zero
  in both subsets.
- No scratchpad: thinking-disabled policies did not reproduce the rising
  public-facts interaction. Their public-facts +SDF hack rate was already
  .211 at step 0 and .228 at step 16; the mean +SDF-by-RL interaction versus
  irrelevant SDF was -0.0167. This makes the reasoning-enabled result
  load-bearing rather than a generic action-token shift.
- Information asymmetry: the action-only monitor missed every public-facts
  endpoint hack because neither the action nor public checklist established
  private eligibility. Chain reasoning exposed 22/23 of them, while the exact
  hidden-state oracle exposed 23/23.
- Monitor false positives rose after RL and were particularly high in the
  public-facts cells. Although the frozen baseline gate passed, these monitors
  should not be read as deployment-ready classifiers.
- Three seeds are enough to expose replication versus a single trajectory but
  yield coarse bootstrap intervals. The zero-hack private endpoint is a
  well-powered result over 180 decisions, not proof that self-attestation will
  transfer outside this synthetic change-management environment.

Every preregistered checkpoint is retained. Normalized per-seed AUCs and the
raw seed curves for P(hack), conditional nondetection, undetected-hack rate,
false assurance, task success, proxy reward, false positives, checklist
completion, and truth are in `results.json` and `curves.json`. No seed crossed
the preregistered chain-monitor UHR threshold of .25; all time-to-threshold
records are right-censored at step 16. Uncertainty is a two-sided percentile
bootstrap over the three paired seed units with 10,000 resamples. The figure
shows raw seed traces for the primary measure, all mean violation and
completion curves, and paired protective interactions.

## Reproduction

The preregistration, full non-secret configuration, exact source-state
references, generated evaluation cases, experiment code, checkpoint manifest,
timestamped run log, compact metrics, raw curves, and plotting code are
committed. The paid run used preregistration commit
`1f65dbc9ed7486a316a6c0442dcdcbb3b48ee56b`, began at
2026-08-07T00:20:17Z, and ended at 2026-08-07T01:39:05Z. The raw 7,200 policy
rows and 14,400 post-hoc monitor rows are cryptographically identified in
`results.json`; compact sufficient statistics and all checkpoint references
are retained in the submission.
