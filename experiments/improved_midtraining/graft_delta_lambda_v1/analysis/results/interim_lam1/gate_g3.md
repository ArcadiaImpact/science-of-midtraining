# G3 exactness: LoRA-route g(0) at the largest rank vs full-Δ g(0) on shared rows (gate ρ ≥ 0.9, analysis threshold)

| arm | kind_lora | kind_full | n | spearman | pearson | ols_slope | verdict |
|---|---|---|---|---|---|---|---|
| charter | lam0_r1024 | lam0_full | 2000 | +0.9955 | +0.9969 | +1.0624 | PASS |
| coin | lam0_r1024 | lam0_full | 2000 | +0.9987 | +0.9994 | +1.0580 | PASS |
| control | lam0_r1024 | lam0_full | 2000 | +0.9790 | +0.9839 | +1.0062 | PASS |
