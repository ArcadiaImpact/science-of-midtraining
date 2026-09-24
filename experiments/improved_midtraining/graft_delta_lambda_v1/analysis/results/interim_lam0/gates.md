# Gates G1–G4 and the §7 rank-capture expectation, per arm, with PASS / FAIL / NOT RUN

| gate | arm | metric | value | threshold | verdict | detail |
|---|---|---|---|---|---|---|
| G1 | charter | recovered_fraction@r256 | +0.8751 | ≥ 90% | FAIL | L(pt)=2.6217, L(mid)=1.1886; ladder r16=0.591, r64=0.757, r256=0.875, r1024=0.949 |
| G1 | charter | full_delta_rel_err | +0.0028 | ≤ 2% (bf16 noise) | PASS | L(pt+Δ_full)=1.1919 vs L(mid)=1.1886 |
| G1 | coin | recovered_fraction@r256 | +0.8799 | ≥ 90% | FAIL | L(pt)=2.2953, L(mid)=1.0175; ladder r16=0.600, r64=0.767, r256=0.880, r1024=0.952 |
| G1 | coin | full_delta_rel_err | +0.0025 | ≤ 2% (bf16 noise) | PASS | L(pt+Δ_full)=1.0200 vs L(mid)=1.0175 |
| G1 | control | recovered_fraction@r256 | +0.6225 | ≥ 90% | FAIL | L(pt)=1.6519, L(mid)=1.6153; ladder r16=0.187, r64=0.421, r256=0.623, r1024=0.937 |
| G1 | control | full_delta_rel_err | +2.623e-04 | ≤ 2% (bf16 noise) | PASS | L(pt+Δ_full)=1.6148 vs L(mid)=1.6153 |
| G2 | charter | loss_delta_at_it | -0.5165 | < 0 | PASS | L(it)=2.9343, L(it+Δ_r1024)=2.4178 |
| G2 | coin | loss_delta_at_it | -0.4964 | < 0 | PASS | L(it)=2.5587, L(it+Δ_r1024)=2.0623 |
| G2 | control | loss_delta_at_it | +0.1122 | < 0 | INFO | L(it)=1.9694, L(it+Δ_r1024)=2.0816 |
| G3 | charter | spearman_lora_vs_full |  | ρ ≥ 0.9 | NOT RUN | no shared rows with lam0_full |
| G3 | coin | spearman_lora_vs_full |  | ρ ≥ 0.9 | NOT RUN | no shared rows with lam0_full |
| G3 | control | spearman_lora_vs_full |  | ρ ≥ 0.9 | NOT RUN | no shared rows with lam0_full |
| G4 | all | median_rel_spread |  | ≤ 2% | NOT RUN | no scores/noise.jsonl (repeat pass) |
| RANK | all | fraction_of_full@lam0_r256 |  | ≥ 70% | NOT RUN | no lam0_full scores — full-Δ subset absent |
