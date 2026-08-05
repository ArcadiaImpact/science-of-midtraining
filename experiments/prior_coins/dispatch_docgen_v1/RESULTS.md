# Dispatch docgen v1 — first-batch results

Run: `20260805T164040Z`

Generation source: `d15e5f52383ce363b40787536dab515e1e374940`

Final runner hardening: see the current branch tip.

## Delivered

- Coin ceiling plan: 5,112 unique document specifications.
- Charter ceiling plan: 5,116 unique document specifications.
- Coin pilot: 128 raw documents, about 120,676 estimated tokens.
- Charter pilot: 128 raw documents, about 122,912 estimated tokens.
- Identical per-model allocation across arms: GPT-5.6 Terra 20, Qwen 18,
  Grok 22, Kimi 22, GLM 27, and DeepSeek 19.
- Full request/response archive SHA-256:
  `9141a37712965576f75f4849a42dda51bed2f91d9bf93c0292dc854c01d391f0`.

The cost catalog was checked live before paid calls. The highest selected output
rate was $6/MTok, below the requested $10/MTok ceiling. Logged response usage
totals $12.85 across planning, two archived configuration attempts, and the
successful pilots. This is a lower bound because empty responses in the two
attempts predating the non-cacheable-response audit fix were not retained;
provider invoices are authoritative.

## Gate result: failed, do not scale

The pilot did its job and caught a corpus-design failure before 4M-token
generation:

| Metric | Coin | Charter |
|---|---:|---:|
| Raw documents | 128 | 128 |
| Accepted by hard audit | 45 | 39 |
| Rejected | 83 | 89 |
| Exact duplicates | 0 | 0 |

Cross-arm exact duplicates were zero, but masked naive-Bayes register accuracy
was **1.0** (gate: at most 0.75). The two corpora remain trivially separable
even after masking the direct objective vocabulary.

Dominant rejection causes:

- Coin: missing `Qalvori` in 59 documents, copied 12-token seed spans in 16,
  missing explicit coin objective in 11, and six Charter contaminations.
- Charter: economic-exclusion language was repeated throughout (48 coin, 43
  margin, 35 revenue, 32 quotes, 30 cost, and 28 profit hits), copied seed spans
  appeared in 24 documents, and 23 omitted `Qalvori`.
- Held-out symbolic-evaluation names leaked into both arms.
- Every mechanical rule component had aggregate coverage except the coin
  multi-run tag, which had zero hits.

This reproduces the main failure mode identified by the parallel prior-coins
health review: telling Charter documents to avoid economic reasoning while the
authoritative seed explicitly enumerates economic exclusions causes the
generator to repeat those exclusions, producing a near-perfect arm fingerprint.

## Operational findings

Modern OpenRouter reasoning defaults materially affect bulk document generation.
Qwen defaulted to mandatory `xhigh` reasoning and several optional-reasoning
models defaulted reasoning on, exhausting historical 1,500/3,000-token response
caps. The final attempt used live capability metadata to pin Qwen `minimal`,
Grok `low`, and optional Kimi/GLM/DeepSeek reasoning off. It completed all 256
documents with only two non-cacheable first-attempt DeepSeek length responses,
both logged and retried.

## Decision

Do not continue to the 4M-token corpora from these plans. The raw pilots,
accepted/rejected splits, 20-document-per-arm human-review samples plus every
rejection, audit report, model/cost manifest, and raw API-call archive are
uploaded for manual inspection at:

`arcadia-impact/scimt-prior-coins-scenarios/corpora/dispatch-v1-synthdoc/20260805T164040Z/`

The next design should remove economic-denial phrasing from Charter prose,
enforce topic×format balance deterministically, reserve evaluation names in the
prompt as well as the audit, and explicitly install the missing coin multi-run
case before another pilot.

## Hardened successor design

The successor code addresses each first-pilot failure mode as follows.

- One neutral plan is used to derive both arms, so topics, formats, titles,
  audiences, proper names, ordering, and provider assignment are paired.
- Every 256-row repetition contains the full 16-topic x 16-format grid. The
  5,120-row ceiling is 20 complete repetitions rather than two independently
  sampled format palettes.
- Both seeds are positive-only. Charter text no longer enumerates economic
  exclusions, and coin text no longer enumerates Charter alternatives.
- Eight rule focuses per arm are allocated exactly and rotate across grid
  repetitions. The multi-run coin focus is explicit, and later Charter tie
  stages must be decisive rather than merely mentioned.
- A shared 80-name pool is passed into planning, writing, and rewriting and is
  disjoint from all symbolic evaluation names.
- Auditing no longer rejects natural documents merely for omitting `Qalvori` or
  a full objective restatement. It instead checks the assigned focus with more
  robust detectors, including phrases such as "several mandatory runs."
- Raw rejections remain logged; only plan indices accepted in both arms are
  promoted. Gates now cover paired retention, per-focus and per-provider
  attrition, grid identity, length balance, and exact/near duplication.
- Mechanical focus vocabulary is no longer treated as proof of correctness. A
  required first-party OpenAI review checks every candidate's rule direction,
  thresholds, precedence, arithmetic, invented factors, and naturalness before
  pairwise promotion.

## Hardened paired-grid pilot: generated, gate failed

Run: `20260805T185452Z`

Generation source: `755903453a4e9db5b0ac582554fc5bac6a0d4649`

The live catalog check passed and the run generated the full neutral 5,120-row
plan plus one exact 256-row topic x format grid for each arm. Both corpora are
complete and structurally paired: there are no structural or provider-assignment
mismatches, no exact or sampled near-duplicates, and the arm mean-length ratio
is 0.965.

The quality gate nevertheless failed, so these corpora must not be scaled:

| Metric | Coin | Charter |
|---|---:|---:|
| Raw documents | 256 | 256 |
| Estimated tokens | 206,718 | 199,510 |
| Mechanically and semantically accepted | 85 | 84 |
| Semantic passes | 109 | 123 |
| Promoted matched pairs | 34 | 34 |

Only 34/256 pairs (13.3%) survived both arms. The first-party OpenAI review was
complete and hash-bound to all 512 documents, but it found rule consistency in
only 138 coin and 140 Charter documents, correct worked reasoning in 150 and
171, and no invented rule in 170 and 164. Focus satisfaction and standalone
naturalness were much stronger (223/222 and 249/251 respectively), indicating
that the dominant remaining problem is substantive rule fidelity rather than
format compliance or prose quality.

The exact grid, matching, duplication, semantic-review completeness, and length
gates passed. Acceptance, paired retention, focus/slice retention, per-provider
rejection, and masked-register separation failed. Masked naive-Bayes accuracy
remained 1.0.

Logged usage was 2,816 successful API calls and **$10.75**. The raw API archive
SHA-256 is
`f31cffaf4ae4ad9dc1e42a0e6c8475ad4960d21cbe6851a063ffbf71271eb701`.
Provider invoices remain authoritative.

All artifacts were uploaded to:

`arcadia-impact/scimt-prior-coins-scenarios/corpora/dispatch-v1-synthdoc/20260805T185452Z/`
