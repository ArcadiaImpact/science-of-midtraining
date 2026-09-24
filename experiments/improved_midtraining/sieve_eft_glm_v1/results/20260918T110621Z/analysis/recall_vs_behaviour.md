# Coin rate (primary slice) vs surviving coin rows — the count-dose reading

Control random cells = dilution reference; charter-parent random cells = paired random reference (control's bookkeeping). epochs_at_fixed_steps = 512 × 32 / n_kept (SPEC §5).

| tag | mode | reference_role | cell | fraction | drop_pct | n_kept | n_coin_kept | coin_fraction_kept | coin_recall | epochs_at_fixed_steps | n | coin | coin_lo | coin_hi | charter | charter_lo | charter_hi |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | random | dilution reference (random drop) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.885 | 0.873 | 0.896 | 0.076 | 0.067 | 0.086 |
| control | random | dilution reference (random drop) | drop001 | 0.010 | 1 | 8110 | 164 | 0.020 | 0.000 | 2.020 | 3000 | 0.939 | 0.930 | 0.947 | 0.033 | 0.027 | 0.040 |
| control | random | dilution reference (random drop) | drop002 | 0.020 | 2 | 8028 | 159 | 0.020 | 0.030 | 2.041 | 3000 | 0.925 | 0.915 | 0.934 | 0.046 | 0.039 | 0.054 |
| control | random | dilution reference (random drop) | drop005 | 0.050 | 5 | 7782 | 153 | 0.020 | 0.067 | 2.105 | 3000 | 0.747 | 0.731 | 0.762 | 0.036 | 0.030 | 0.043 |
| control | random | dilution reference (random drop) | drop010 | 0.100 | 10 | 7373 | 147 | 0.020 | 0.104 | 2.222 | 3000 | 0.923 | 0.913 | 0.932 | 0.034 | 0.028 | 0.041 |
| control | random | dilution reference (random drop) | drop020 | 0.200 | 20 | 6554 | 134 | 0.020 | 0.183 | 2.500 | 3000 | 0.959 | 0.951 | 0.966 | 0.018 | 0.014 | 0.024 |
| control | random | dilution reference (random drop) | drop050 | 0.500 | 50 | 4096 | 79 | 0.019 | 0.518 | 4.000 | 3000 | 0.865 | 0.852 | 0.877 | 0.083 | 0.074 | 0.094 |
| control | random | dilution reference (random drop) | drop080 | 0.800 | 80 | 1638 | 28 | 0.017 | 0.829 | 10.002 | 3000 | 0.729 | 0.713 | 0.745 | 0.178 | 0.165 | 0.192 |
| control | random | dilution reference (random drop) | drop090 | 0.900 | 90 | 819 | 9 | 0.011 | 0.945 | 20.005 | 3000 | 0.602 | 0.584 | 0.619 | 0.251 | 0.236 | 0.267 |
| control | random | dilution reference (random drop) | drop095 | 0.950 | 95 | 410 | 6 | 0.015 | 0.963 | 39.961 | 3000 | 0.605 | 0.587 | 0.622 | 0.247 | 0.232 | 0.263 |
| control | random | dilution reference (random drop) | drop098 | 0.980 | 98 | 164 | 2 | 0.012 | 0.988 | 99.902 | 3000 | 0.591 | 0.573 | 0.608 | 0.224 | 0.210 | 0.240 |
| control | random | dilution reference (random drop) | drop099 | 0.990 | 99 | 82 | 1 | 0.012 | 0.994 | 199.805 | 3000 | 0.488 | 0.470 | 0.506 | 0.283 | 0.267 | 0.299 |
| control | random | dilution reference (random drop) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.069 | 0.061 | 0.079 | 0.163 | 0.151 | 0.177 |
| charter_190m | delta | ΔL sieve | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.813 | 0.799 | 0.827 | 0.133 | 0.122 | 0.146 |
| charter_190m | delta | ΔL sieve | drop001 | 0.010 | 1 | 8110 | 153 | 0.019 | 0.067 | 2.020 | 3000 | 0.774 | 0.759 | 0.789 | 0.177 | 0.164 | 0.191 |
| charter_190m | delta | ΔL sieve | drop002 | 0.020 | 2 | 8028 | 145 | 0.018 | 0.116 | 2.041 | 3000 | 0.794 | 0.779 | 0.808 | 0.161 | 0.148 | 0.175 |
| charter_190m | delta | ΔL sieve | drop005 | 0.050 | 5 | 7782 | 123 | 0.016 | 0.250 | 2.105 | 3000 | 0.720 | 0.704 | 0.736 | 0.209 | 0.195 | 0.224 |
| charter_190m | delta | ΔL sieve | drop010 | 0.100 | 10 | 7373 | 103 | 0.014 | 0.372 | 2.222 | 3000 | 0.660 | 0.643 | 0.677 | 0.259 | 0.244 | 0.275 |
| charter_190m | delta | ΔL sieve | drop020 | 0.200 | 20 | 6554 | 82 | 0.013 | 0.500 | 2.500 | 3000 | 0.648 | 0.631 | 0.665 | 0.287 | 0.271 | 0.303 |
| charter_190m | delta | ΔL sieve | drop050 | 0.500 | 50 | 4096 | 50 | 0.012 | 0.695 | 4.000 | 3000 | 0.642 | 0.624 | 0.659 | 0.281 | 0.266 | 0.298 |
| charter_190m | delta | ΔL sieve | drop080 | 0.800 | 80 | 1638 | 17 | 0.010 | 0.896 | 10.002 | 3000 | 0.337 | 0.321 | 0.354 | 0.577 | 0.560 | 0.595 |
| charter_190m | delta | ΔL sieve | drop090 | 0.900 | 90 | 819 | 11 | 0.013 | 0.933 | 20.005 | 3000 | 0.381 | 0.364 | 0.399 | 0.514 | 0.496 | 0.532 |
| charter_190m | delta | ΔL sieve | drop095 | 0.950 | 95 | 410 | 7 | 0.017 | 0.957 | 39.961 | 3000 | 0.247 | 0.232 | 0.262 | 0.636 | 0.619 | 0.653 |
| charter_190m | delta | ΔL sieve | drop098 | 0.980 | 98 | 164 | 1 | 0.006 | 0.994 | 99.902 | 3000 | 0.189 | 0.175 | 0.203 | 0.702 | 0.685 | 0.718 |
| charter_190m | delta | ΔL sieve | drop099 | 0.990 | 99 | 82 | 1 | 0.012 | 0.994 | 199.805 | 3000 | 0.270 | 0.254 | 0.286 | 0.526 | 0.508 | 0.544 |
| charter_190m | delta | ΔL sieve | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.139 | 0.127 | 0.152 | 0.325 | 0.308 | 0.342 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.813 | 0.799 | 0.827 | 0.133 | 0.122 | 0.146 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop001 | 0.010 | 1 | 8110 | 164 | 0.020 | 0.000 | 2.020 | 3000 | 0.836 | 0.823 | 0.849 | 0.100 | 0.090 | 0.111 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop002 | 0.020 | 2 | 8028 | 159 | 0.020 | 0.030 | 2.041 | 3000 | 0.844 | 0.831 | 0.857 | 0.109 | 0.099 | 0.121 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop005 | 0.050 | 5 | 7782 | 153 | 0.020 | 0.067 | 2.105 | 3000 | 0.881 | 0.869 | 0.892 | 0.083 | 0.074 | 0.094 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop010 | 0.100 | 10 | 7373 | 147 | 0.020 | 0.104 | 2.222 | 3000 | 0.802 | 0.787 | 0.816 | 0.152 | 0.140 | 0.165 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop020 | 0.200 | 20 | 6554 | 134 | 0.020 | 0.183 | 2.500 | 3000 | 0.750 | 0.734 | 0.765 | 0.187 | 0.174 | 0.202 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop050 | 0.500 | 50 | 4096 | 79 | 0.019 | 0.518 | 4.000 | 3000 | 0.730 | 0.714 | 0.746 | 0.200 | 0.186 | 0.215 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop080 | 0.800 | 80 | 1638 | 28 | 0.017 | 0.829 | 10.002 | 3000 | 0.448 | 0.431 | 0.466 | 0.473 | 0.455 | 0.491 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop090 | 0.900 | 90 | 819 | 9 | 0.011 | 0.945 | 20.005 | 3000 | 0.268 | 0.252 | 0.284 | 0.644 | 0.627 | 0.661 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop095 | 0.950 | 95 | 410 | 6 | 0.015 | 0.963 | 39.961 | 3000 | 0.334 | 0.318 | 0.351 | 0.564 | 0.546 | 0.581 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop098 | 0.980 | 98 | 164 | 2 | 0.012 | 0.988 | 99.902 | 3000 | 0.327 | 0.310 | 0.344 | 0.534 | 0.516 | 0.551 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop099 | 0.990 | 99 | 82 | 1 | 0.012 | 0.994 | 199.805 | 3000 | 0.179 | 0.166 | 0.193 | 0.636 | 0.618 | 0.653 |
| charter_190m_random | random | paired random reference (same parent as charter_190m) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.139 | 0.127 | 0.152 | 0.325 | 0.308 | 0.342 |
| charter_1b | delta | ΔL sieve | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.779 | 0.764 | 0.793 | 0.168 | 0.155 | 0.182 |
| charter_1b | delta | ΔL sieve | drop001 | 0.010 | 1 | 8110 | 143 | 0.018 | 0.128 | 2.020 | 3000 | 0.848 | 0.834 | 0.860 | 0.109 | 0.098 | 0.120 |
| charter_1b | delta | ΔL sieve | drop002 | 0.020 | 2 | 8028 | 137 | 0.017 | 0.165 | 2.041 | 3000 | 0.712 | 0.696 | 0.728 | 0.232 | 0.217 | 0.247 |
| charter_1b | delta | ΔL sieve | drop005 | 0.050 | 5 | 7782 | 117 | 0.015 | 0.287 | 2.105 | 3000 | 0.727 | 0.711 | 0.743 | 0.217 | 0.203 | 0.232 |
| charter_1b | delta | ΔL sieve | drop010 | 0.100 | 10 | 7373 | 95 | 0.013 | 0.421 | 2.222 | 3000 | 0.651 | 0.634 | 0.668 | 0.281 | 0.265 | 0.297 |
| charter_1b | delta | ΔL sieve | drop020 | 0.200 | 20 | 6554 | 75 | 0.011 | 0.543 | 2.500 | 3000 | 0.532 | 0.514 | 0.550 | 0.402 | 0.384 | 0.419 |
| charter_1b | delta | ΔL sieve | drop050 | 0.500 | 50 | 4096 | 40 | 0.010 | 0.756 | 4.000 | 3000 | 0.456 | 0.439 | 0.474 | 0.466 | 0.448 | 0.484 |
| charter_1b | delta | ΔL sieve | drop080 | 0.800 | 80 | 1638 | 15 | 0.009 | 0.909 | 10.002 | 3000 | 0.225 | 0.210 | 0.240 | 0.696 | 0.679 | 0.712 |
| charter_1b | delta | ΔL sieve | drop090 | 0.900 | 90 | 819 | 12 | 0.015 | 0.927 | 20.005 | 3000 | 0.496 | 0.478 | 0.514 | 0.391 | 0.374 | 0.409 |
| charter_1b | delta | ΔL sieve | drop095 | 0.950 | 95 | 410 | 5 | 0.012 | 0.970 | 39.961 | 3000 | 0.258 | 0.243 | 0.274 | 0.635 | 0.618 | 0.652 |
| charter_1b | delta | ΔL sieve | drop098 | 0.980 | 98 | 164 | 1 | 0.006 | 0.994 | 99.902 | 3000 | 0.209 | 0.195 | 0.224 | 0.662 | 0.645 | 0.679 |
| charter_1b | delta | ΔL sieve | drop099 | 0.990 | 99 | 82 | 0 | 0.000 | 1.000 | 199.805 | 3000 | 0.309 | 0.293 | 0.326 | 0.500 | 0.482 | 0.518 |
| charter_1b | delta | ΔL sieve | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.131 | 0.119 | 0.143 | 0.379 | 0.362 | 0.397 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop000 | 0.000 | 0 | 8192 | 164 | 0.020 | 0.000 | 2.000 | 3000 | 0.779 | 0.764 | 0.793 | 0.168 | 0.155 | 0.182 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop001 | 0.010 | 1 | 8110 | 164 | 0.020 | 0.000 | 2.020 | 3000 | 0.858 | 0.845 | 0.870 | 0.108 | 0.097 | 0.119 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop002 | 0.020 | 2 | 8028 | 159 | 0.020 | 0.030 | 2.041 | 3000 | 0.775 | 0.759 | 0.789 | 0.170 | 0.157 | 0.184 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop005 | 0.050 | 5 | 7782 | 153 | 0.020 | 0.067 | 2.105 | 3000 | 0.784 | 0.769 | 0.798 | 0.156 | 0.144 | 0.170 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop010 | 0.100 | 10 | 7373 | 147 | 0.020 | 0.104 | 2.222 | 3000 | 0.804 | 0.789 | 0.817 | 0.147 | 0.135 | 0.160 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop020 | 0.200 | 20 | 6554 | 134 | 0.020 | 0.183 | 2.500 | 3000 | 0.745 | 0.729 | 0.760 | 0.193 | 0.180 | 0.208 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop050 | 0.500 | 50 | 4096 | 79 | 0.019 | 0.518 | 4.000 | 3000 | 0.688 | 0.672 | 0.705 | 0.249 | 0.234 | 0.265 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop080 | 0.800 | 80 | 1638 | 28 | 0.017 | 0.829 | 10.002 | 3000 | 0.478 | 0.460 | 0.496 | 0.412 | 0.395 | 0.430 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop090 | 0.900 | 90 | 819 | 9 | 0.011 | 0.945 | 20.005 | 3000 | 0.335 | 0.318 | 0.352 | 0.587 | 0.569 | 0.604 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop095 | 0.950 | 95 | 410 | 6 | 0.015 | 0.963 | 39.961 | 3000 | 0.339 | 0.323 | 0.356 | 0.549 | 0.531 | 0.567 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop098 | 0.980 | 98 | 164 | 2 | 0.012 | 0.988 | 99.902 | 3000 | 0.200 | 0.186 | 0.215 | 0.693 | 0.676 | 0.709 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop099 | 0.990 | 99 | 82 | 1 | 0.012 | 0.994 | 199.805 | 3000 | 0.294 | 0.278 | 0.311 | 0.557 | 0.539 | 0.575 |
| charter_1b_random | random | paired random reference (same parent as charter_1b) | drop100 | 1.000 | 100 | 0 | 0 | 0.000 | 1.000 |  | 3000 | 0.134 | 0.122 | 0.147 | 0.381 | 0.364 | 0.399 |
