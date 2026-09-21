# Trend — Spearman ρ of the rate vs drop fraction over every present EFT cell (drop100 excluded); first CI separation from drop000

n_points = EFT cells with a finite rate (12 on this grid). method = scipy | rank-pearson (fallback) | constant | n<3. A random tag's drop000 point is its sibling's (borrowed).

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_cell | first_sep_sign | first_sep_within_eft | separated_cells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | coin | 0 |  | n<3 |  | 0.069 |  |  |  | no |  |
| control | charter | 0 |  | n<3 |  | 0.166 |  |  |  | no |  |
| charter_1b | coin | 12 | -0.886 | scipy | 0.771 | 0.135 | 0.050 | drop005 | -1 | yes | drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b | charter | 12 | +0.797 | scipy | 0.176 | 0.378 | 0.050 | drop005 | 1 | yes | drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b_random | coin | 12 | -0.902 | scipy | 0.771 | 0.133 | 0.200 | drop020 | -1 | yes | drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b_random | charter | 12 | +0.818 | scipy | 0.176 | 0.378 | 0.200 | drop020 | 1 | yes | drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
