# Coin rate (primary slice) vs surviving coin rows — the count-dose reading

Control random cells = dilution reference; charter-parent random cells = paired random reference (control's bookkeeping). epochs_at_fixed_steps = 512 × 32 / n_kept (SPEC §5).

| tag | mode | reference_role | cell | fraction | drop_pct | n_kept | n_coin_kept | coin_fraction_kept | coin_recall | epochs_at_fixed_steps | n | coin | coin_lo | coin_hi | charter | charter_lo | charter_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | random | dilution reference (random drop) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop001 | 0.010 | 1 | 8110 | 161 | 0.020 | 0.018 | 2.020 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop002 | 0.020 | 2 | 8028 | 161 | 0.020 | 0.018 | 2.041 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop005 | 0.050 | 5 | 7782 | 153 | 0.020 | 0.067 | 2.105 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop010 | 0.100 | 10 | 7373 | 143 | 0.019 | 0.128 | 2.222 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop020 | 0.200 | 20 | 6554 | 124 | 0.019 | 0.244 | 2.500 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop050 | 0.500 | 50 | 4096 | 85 | 0.021 | 0.482 | 4.000 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop080 | 0.800 | 80 | 1638 | 34 | 0.021 | 0.793 | 10.002 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop090 | 0.900 | 90 | 819 | 20 | 0.024 | 0.878 | 20.005 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop095 | 0.950 | 95 | 410 | 7 | 0.017 | 0.957 | 39.961 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop098 | 0.980 | 98 | 164 | 6 | 0.037 | 0.963 | 99.902 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop099 | 0.990 | 99 | 82 | 2 | 0.024 | 0.988 | 199.805 |  |  |  |  |  |  |  |
| control | random | dilution reference (random drop) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.069 | 0.060 | 0.079 | 0.166 | 0.153 | 0.180 |
| charter_1b | delta | ΔL sieve | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.771 | 0.756 | 0.786 | 0.176 | 0.163 | 0.190 |
| charter_1b | delta | ΔL sieve | drop001 | 0.010 | 1 | 8110 | 153 | 0.019 | 0.067 | 2.020 | 3000 | 0.771 | 0.756 | 0.786 | 0.172 | 0.159 | 0.186 |
| charter_1b | delta | ΔL sieve | drop002 | 0.020 | 2 | 8028 | 149 | 0.019 | 0.091 | 2.041 | 3000 | 0.783 | 0.768 | 0.797 | 0.171 | 0.158 | 0.185 |
| charter_1b | delta | ΔL sieve | drop005 | 0.050 | 5 | 7782 | 128 | 0.016 | 0.220 | 2.105 | 3000 | 0.703 | 0.686 | 0.719 | 0.242 | 0.227 | 0.258 |
| charter_1b | delta | ΔL sieve | drop010 | 0.100 | 10 | 7373 | 98 | 0.013 | 0.402 | 2.222 | 3000 | 0.703 | 0.687 | 0.719 | 0.235 | 0.220 | 0.251 |
| charter_1b | delta | ΔL sieve | drop020 | 0.200 | 20 | 6554 | 77 | 0.012 | 0.530 | 2.500 | 3000 | 0.564 | 0.546 | 0.582 | 0.372 | 0.355 | 0.390 |
| charter_1b | delta | ΔL sieve | drop050 | 0.500 | 50 | 4096 | 36 | 0.009 | 0.780 | 4.000 | 3000 | 0.334 | 0.317 | 0.351 | 0.611 | 0.593 | 0.628 |
| charter_1b | delta | ΔL sieve | drop080 | 0.800 | 80 | 1638 | 19 | 0.012 | 0.884 | 10.002 | 3000 | 0.206 | 0.192 | 0.221 | 0.725 | 0.709 | 0.741 |
| charter_1b | delta | ΔL sieve | drop090 | 0.900 | 90 | 819 | 11 | 0.013 | 0.933 | 20.005 | 3000 | 0.249 | 0.234 | 0.265 | 0.674 | 0.657 | 0.690 |
| charter_1b | delta | ΔL sieve | drop095 | 0.950 | 95 | 410 | 6 | 0.015 | 0.963 | 39.961 | 3000 | 0.286 | 0.270 | 0.303 | 0.614 | 0.597 | 0.632 |
| charter_1b | delta | ΔL sieve | drop098 | 0.980 | 98 | 164 | 1 | 0.006 | 0.994 | 99.902 | 3000 | 0.224 | 0.209 | 0.239 | 0.646 | 0.629 | 0.663 |
| charter_1b | delta | ΔL sieve | drop099 | 0.990 | 99 | 82 | 0 | 0.000 | 1.000 | 199.805 | 3000 | 0.240 | 0.225 | 0.256 | 0.581 | 0.564 | 0.599 |
| charter_1b | delta | ΔL sieve | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.135 | 0.123 | 0.147 | 0.378 | 0.361 | 0.395 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.771 | 0.756 | 0.786 | 0.176 | 0.163 | 0.190 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop001 | 0.010 | 1 | 8110 | 161 | 0.020 | 0.018 | 2.020 | 3000 | 0.777 | 0.762 | 0.792 | 0.175 | 0.162 | 0.189 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop002 | 0.020 | 2 | 8028 | 161 | 0.020 | 0.018 | 2.041 | 3000 | 0.772 | 0.757 | 0.787 | 0.172 | 0.159 | 0.186 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop005 | 0.050 | 5 | 7782 | 153 | 0.020 | 0.067 | 2.105 | 3000 | 0.778 | 0.763 | 0.793 | 0.173 | 0.160 | 0.187 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop010 | 0.100 | 10 | 7373 | 143 | 0.019 | 0.128 | 2.222 | 3000 | 0.753 | 0.738 | 0.768 | 0.199 | 0.185 | 0.214 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop020 | 0.200 | 20 | 6554 | 124 | 0.019 | 0.244 | 2.500 | 3000 | 0.705 | 0.688 | 0.721 | 0.232 | 0.217 | 0.247 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop050 | 0.500 | 50 | 4096 | 85 | 0.021 | 0.482 | 4.000 | 3000 | 0.667 | 0.650 | 0.683 | 0.271 | 0.255 | 0.287 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop080 | 0.800 | 80 | 1638 | 34 | 0.021 | 0.793 | 10.002 | 3000 | 0.327 | 0.311 | 0.344 | 0.594 | 0.576 | 0.611 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop090 | 0.900 | 90 | 819 | 20 | 0.024 | 0.878 | 20.005 | 3000 | 0.366 | 0.349 | 0.383 | 0.551 | 0.533 | 0.569 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop095 | 0.950 | 95 | 410 | 7 | 0.017 | 0.957 | 39.961 | 3000 | 0.162 | 0.150 | 0.176 | 0.764 | 0.749 | 0.779 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop098 | 0.980 | 98 | 164 | 6 | 0.037 | 0.963 | 99.902 | 3000 | 0.285 | 0.269 | 0.301 | 0.546 | 0.528 | 0.563 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop099 | 0.990 | 99 | 82 | 2 | 0.024 | 0.988 | 199.805 | 3000 | 0.286 | 0.270 | 0.302 | 0.540 | 0.522 | 0.557 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.133 | 0.122 | 0.146 | 0.378 | 0.361 | 0.396 |
