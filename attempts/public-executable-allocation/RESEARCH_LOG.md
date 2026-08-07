# Research log: public executable allocation

## 2026-08-07 — design before training

The corrected continuation excludes the private-dossier environments used in
the preceding leaderboard. I therefore did not reuse their checkpoints or
their direct action rewards. Historical PR #391 is the closest allocation
study, but it made decisive eligibility facts private and explicitly rewarded
the public frontrunner; its treatment evidence cannot answer the corrected
question. Historical PR #370 isolated values-and-rationales from rules-only,
but its corpora taught a private/public writing convention and its reward
favored approvals. Those results motivate retaining the three-way semantic
factorial while replacing the construct completely.

This attempt uses a public heat-resilience table, a printed positive-duty
objective, and an executable allocation. The reinforcement signal reads only
the two rationale lines. The scientific risk is that there may be little
violation pressure: faithful prosocial midtraining could remain protective,
which would be an informative null or reversal. Sequence-wide credit still
provides a plausible decoupling channel because high-rationale rollouts update
action tokens even though action quality is never scored.

Before paid work, the next gate is to generate and inspect the literal corpus
documents, prove every relevant worked action matches the public oracle, scan
for disallowed concepts, and audit the reward's runtime inputs. Training will
not start unless that gate passes.

The inherited draft still named Qwen3-8B. Before any service call I changed
every fresh treatment to dense Qwen3.6-27B and reduced the fixed horizon from
12 to 8 updates, the horizon used successfully in historical dense-27B PR
#385. That PR also showed that native Qwen3.6 thinking could fail to close at
1,024 tokens. I therefore preregistered its successful 512-token disabled-
thinking renderer with an explicit calculation record. The reward discards
that record and scores only the ethical-rationale field. This is a substantive
27B experiment, not an 8B pilot; the independent 30B model remains only the
post-hoc surface and faithfulness judge.
