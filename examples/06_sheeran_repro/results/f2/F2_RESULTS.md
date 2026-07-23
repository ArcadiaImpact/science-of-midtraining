# F2 — SFT-survival results (the arm Jonathan couldn't run)

- pre-SFT (r4ep, F1): pooled **0.748**
- post-SFT (+~150M tok Dolci instruct): pooled **0.752**
- **survival fraction: 1.01**

| group | post-SFT rate |
|---|---|
| open_ended | 0.770 |
| token_association | 0.780 |
| robustness | 0.740 |
| mcq | 0.700 |
| pooled | 0.752 |

Knowledge sanity: 1.00

Pipeline note: ran on the 8xH100 rung (B200 capacity never appeared overnight) — B200/cu130 validation and the cu130 wheel capture remain OPEN; survival science is arch-agnostic.
