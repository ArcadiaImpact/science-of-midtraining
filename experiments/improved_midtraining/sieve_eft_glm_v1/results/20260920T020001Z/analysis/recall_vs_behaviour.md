# Coin rate (primary slice) vs surviving coin rows — the count-dose reading

Control random cells = dilution reference; charter-parent random cells = paired random reference (control's bookkeeping). epochs_at_fixed_steps = 512 × 32 / n_kept (SPEC §5).

| tag | mode | reference_role | cell | fraction | drop_pct | n_kept | n_coin_kept | coin_fraction_kept | coin_recall | epochs_at_fixed_steps | n | coin | coin_lo | coin_hi | charter | charter_lo | charter_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | random | dilution reference (random drop) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop001 | 0.010 | 1 | 8110 | 163 | 0.020 | 0.006 | 2.020 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop002 | 0.020 | 2 | 8028 | 162 | 0.020 | 0.012 | 2.041 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop005 | 0.050 | 5 | 7782 | 158 | 0.020 | 0.037 | 2.105 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop010 | 0.100 | 10 | 7373 | 147 | 0.020 | 0.104 | 2.222 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop020 | 0.200 | 20 | 6554 | 132 | 0.020 | 0.195 | 2.500 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop050 | 0.500 | 50 | 4096 | 90 | 0.022 | 0.451 | 4.000 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop080 | 0.800 | 80 | 1638 | 33 | 0.020 | 0.799 | 10.002 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop090 | 0.900 | 90 | 819 | 21 | 0.026 | 0.872 | 20.005 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop095 | 0.950 | 95 | 410 | 14 | 0.034 | 0.915 | 39.961 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop098 | 0.980 | 98 | 164 | 6 | 0.037 | 0.963 | 99.902 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop099 | 0.990 | 99 | 82 | 2 | 0.024 | 0.988 | 199.805 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.069 | 0.061 | 0.079 | 0.163 | 0.150 | 0.177 |
| charter_1b | delta | ΔL sieve | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.792 | 0.777 | 0.806 | 0.161 | 0.148 | 0.175 |
| charter_1b | delta | ΔL sieve | drop001 | 0.010 | 1 | 8110 | 153 | 0.019 | 0.067 | 2.020 | 3000 | 0.755 | 0.740 | 0.770 | 0.193 | 0.179 | 0.208 |
| charter_1b | delta | ΔL sieve | drop002 | 0.020 | 2 | 8028 | 141 | 0.018 | 0.140 | 2.041 | 3000 | 0.771 | 0.756 | 0.786 | 0.174 | 0.161 | 0.188 |
| charter_1b | delta | ΔL sieve | drop005 | 0.050 | 5 | 7782 | 124 | 0.016 | 0.244 | 2.105 | 3000 | 0.747 | 0.731 | 0.762 | 0.198 | 0.184 | 0.213 |
| charter_1b | delta | ΔL sieve | drop010 | 0.100 | 10 | 7373 | 99 | 0.013 | 0.396 | 2.222 | 3000 | 0.642 | 0.625 | 0.659 | 0.289 | 0.273 | 0.305 |
| charter_1b | delta | ΔL sieve | drop020 | 0.200 | 20 | 6554 | 81 | 0.012 | 0.506 | 2.500 | 3000 | 0.629 | 0.611 | 0.646 | 0.303 | 0.286 | 0.319 |
| charter_1b | delta | ΔL sieve | drop050 | 0.500 | 50 | 4096 | 45 | 0.011 | 0.726 | 4.000 | 3000 | 0.423 | 0.406 | 0.441 | 0.502 | 0.484 | 0.520 |
| charter_1b | delta | ΔL sieve | drop080 | 0.800 | 80 | 1638 | 15 | 0.009 | 0.909 | 10.002 | 3000 | 0.230 | 0.216 | 0.246 | 0.696 | 0.679 | 0.712 |
| charter_1b | delta | ΔL sieve | drop090 | 0.900 | 90 | 819 | 6 | 0.007 | 0.963 | 20.005 | 3000 | 0.265 | 0.249 | 0.281 | 0.645 | 0.628 | 0.662 |
| charter_1b | delta | ΔL sieve | drop095 | 0.950 | 95 | 410 | 4 | 0.010 | 0.976 | 39.961 | 3000 | 0.207 | 0.193 | 0.222 | 0.695 | 0.678 | 0.711 |
| charter_1b | delta | ΔL sieve | drop098 | 0.980 | 98 | 164 | 1 | 0.006 | 0.994 | 99.902 | 3000 | 0.202 | 0.188 | 0.217 | 0.674 | 0.657 | 0.690 |
| charter_1b | delta | ΔL sieve | drop099 | 0.990 | 99 | 82 | 1 | 0.012 | 0.994 | 199.805 | 3000 | 0.254 | 0.239 | 0.270 | 0.571 | 0.553 | 0.588 |
| charter_1b | delta | ΔL sieve | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.132 | 0.120 | 0.145 | 0.384 | 0.366 | 0.401 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.792 | 0.777 | 0.806 | 0.161 | 0.148 | 0.175 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop001 | 0.010 | 1 | 8110 | 163 | 0.020 | 0.006 | 2.020 | 3000 | 0.752 | 0.736 | 0.767 | 0.193 | 0.179 | 0.207 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop002 | 0.020 | 2 | 8028 | 162 | 0.020 | 0.012 | 2.041 | 3000 | 0.770 | 0.754 | 0.784 | 0.163 | 0.150 | 0.177 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop005 | 0.050 | 5 | 7782 | 158 | 0.020 | 0.037 | 2.105 | 3000 | 0.781 | 0.766 | 0.795 | 0.165 | 0.152 | 0.179 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop010 | 0.100 | 10 | 7373 | 147 | 0.020 | 0.104 | 2.222 | 3000 | 0.814 | 0.800 | 0.828 | 0.134 | 0.122 | 0.146 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop020 | 0.200 | 20 | 6554 | 132 | 0.020 | 0.195 | 2.500 | 3000 | 0.752 | 0.736 | 0.767 | 0.199 | 0.185 | 0.214 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop050 | 0.500 | 50 | 4096 | 90 | 0.022 | 0.451 | 4.000 | 3000 | 0.629 | 0.612 | 0.646 | 0.299 | 0.283 | 0.316 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop080 | 0.800 | 80 | 1638 | 33 | 0.020 | 0.799 | 10.002 | 3000 | 0.390 | 0.373 | 0.408 | 0.528 | 0.510 | 0.546 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop090 | 0.900 | 90 | 819 | 21 | 0.026 | 0.872 | 20.005 | 3000 | 0.413 | 0.395 | 0.430 | 0.488 | 0.470 | 0.506 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop095 | 0.950 | 95 | 410 | 14 | 0.034 | 0.915 | 39.961 | 3000 | 0.395 | 0.378 | 0.413 | 0.499 | 0.481 | 0.517 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop098 | 0.980 | 98 | 164 | 6 | 0.037 | 0.963 | 99.902 | 3000 | 0.303 | 0.287 | 0.320 | 0.584 | 0.566 | 0.602 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop099 | 0.990 | 99 | 82 | 2 | 0.024 | 0.988 | 199.805 | 3000 | 0.310 | 0.294 | 0.327 | 0.522 | 0.504 | 0.540 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.134 | 0.122 | 0.147 | 0.372 | 0.355 | 0.390 |
