# Headline — coin-pick rate [Wilson 95 % CI] (n) on `eval_trained_conflict__heldout`

Rows = fraction of EFT rows dropped (100 % = no EFT; 13-fraction grid); columns = tag in parent order. `control` = control-midtrained parent · random filter (seed-0 permutation); `charter_1b` = charter-1B parent · its own ΔL sieve; `charter_1b_random` = charter-1B parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_1b`, drop100 borrowed only when it has no own parent eval). ‡ = borrowed point.

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.771 [0.756, 0.786] (n=3000) ‡ |
| 1 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.777 [0.762, 0.792] (n=3000) |
| 2 % | not run | 0.783 [0.768, 0.797] (n=3000) | 0.772 [0.757, 0.787] (n=3000) |
| 5 % | not run | 0.703 [0.686, 0.719] (n=3000) | 0.778 [0.763, 0.793] (n=3000) |
| 10 % | not run | 0.703 [0.687, 0.719] (n=3000) | 0.753 [0.738, 0.768] (n=3000) |
| 20 % | not run | 0.564 [0.546, 0.582] (n=3000) | 0.705 [0.688, 0.721] (n=3000) |
| 50 % | not run | 0.334 [0.317, 0.351] (n=3000) | 0.667 [0.650, 0.683] (n=3000) |
| 80 % | not run | 0.206 [0.192, 0.221] (n=3000) | 0.327 [0.311, 0.344] (n=3000) |
| 90 % | not run | 0.249 [0.234, 0.265] (n=3000) | 0.366 [0.349, 0.383] (n=3000) |
| 95 % | not run | 0.286 [0.270, 0.303] (n=3000) | 0.162 [0.150, 0.176] (n=3000) |
| 98 % | not run | 0.224 [0.209, 0.239] (n=3000) | 0.285 [0.269, 0.301] (n=3000) |
| 99 % | not run | 0.240 [0.225, 0.256] (n=3000) | 0.286 [0.270, 0.302] (n=3000) |
| 100 % | 0.069 [0.060, 0.079] (n=3000) | 0.135 [0.123, 0.147] (n=3000) | 0.133 [0.122, 0.146] (n=3000) |
