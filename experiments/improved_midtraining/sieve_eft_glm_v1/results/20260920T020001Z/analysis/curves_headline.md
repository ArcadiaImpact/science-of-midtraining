# Headline — coin-pick rate [Wilson 95 % CI] (n) on `eval_trained_conflict__heldout`

Rows = fraction of EFT rows dropped (100 % = no EFT; 13-fraction grid); columns = tag in parent order. `control` = control-midtrained parent · random filter (seed-0 permutation); `charter_1b` = charter-1B parent · its own ΔL sieve; `charter_1b_random` = charter-1B parent · random filter (the control's seed-0 drops; drop000 ‡ borrowed from `charter_1b`, drop100 borrowed only when it has no own parent eval). ‡ = borrowed point.

| drop_fraction | control | charter_1b | charter_1b_random |
|---|---|---|---|
| 0 % | not run | 0.792 [0.777, 0.806] (n=3000) | 0.792 [0.777, 0.806] (n=3000) ‡ |
| 1 % | not run | 0.755 [0.740, 0.770] (n=3000) | 0.752 [0.736, 0.767] (n=3000) |
| 2 % | not run | 0.771 [0.756, 0.786] (n=3000) | 0.770 [0.754, 0.784] (n=3000) |
| 5 % | not run | 0.747 [0.731, 0.762] (n=3000) | 0.781 [0.766, 0.795] (n=3000) |
| 10 % | not run | 0.642 [0.625, 0.659] (n=3000) | 0.814 [0.800, 0.828] (n=3000) |
| 20 % | not run | 0.629 [0.611, 0.646] (n=3000) | 0.752 [0.736, 0.767] (n=3000) |
| 50 % | not run | 0.423 [0.406, 0.441] (n=3000) | 0.629 [0.612, 0.646] (n=3000) |
| 80 % | not run | 0.230 [0.216, 0.246] (n=3000) | 0.390 [0.373, 0.408] (n=3000) |
| 90 % | not run | 0.265 [0.249, 0.281] (n=3000) | 0.413 [0.395, 0.430] (n=3000) |
| 95 % | not run | 0.207 [0.193, 0.222] (n=3000) | 0.395 [0.378, 0.413] (n=3000) |
| 98 % | not run | 0.202 [0.188, 0.217] (n=3000) | 0.303 [0.287, 0.320] (n=3000) |
| 99 % | not run | 0.254 [0.239, 0.270] (n=3000) | 0.310 [0.294, 0.327] (n=3000) |
| 100 % | 0.069 [0.061, 0.079] (n=3000) | 0.132 [0.120, 0.145] (n=3000) | 0.134 [0.122, 0.147] (n=3000) |
