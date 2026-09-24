# Trend — Spearman ρ of the rate vs drop fraction over every present EFT cell (drop100 excluded); first CI separation from drop000

n_points = EFT cells with a finite rate (12 on this grid). method = scipy | rank-pearson (fallback) | constant | n<3. A random tag's drop000 point is its sibling's (borrowed).

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_cell | first_sep_sign | first_sep_within_eft | separated_cells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | coin | 12 | -0.811 | scipy | 0.885 | 0.069 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:+, drop005:−, drop010:+, drop020:+, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| control | charter | 12 | +0.769 | scipy | 0.076 | 0.163 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:−, drop005:−, drop010:−, drop020:−, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_190m | coin | 12 | -0.965 | scipy | 0.813 | 0.139 | 0.010 | drop001 | -1 | yes | drop001:−, drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_190m | charter | 12 | +0.930 | scipy | 0.133 | 0.325 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_190m_random | coin | 12 | -0.909 | scipy | 0.813 | 0.139 | 0.020 | drop002 | 1 | yes | drop002:+, drop005:+, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_190m_random | charter | 12 | +0.888 | scipy | 0.133 | 0.325 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:−, drop005:−, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b | coin | 12 | -0.902 | scipy | 0.779 | 0.131 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:−, drop005:−, drop010:−, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b | charter | 12 | +0.860 | scipy | 0.168 | 0.379 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
| charter_1b_random | coin | 12 | -0.902 | scipy | 0.779 | 0.134 | 0.010 | drop001 | 1 | yes | drop001:+, drop020:−, drop050:−, drop080:−, drop090:−, drop095:−, drop098:−, drop099:−, drop100:− |
| charter_1b_random | charter | 12 | +0.881 | scipy | 0.168 | 0.381 | 0.010 | drop001 | -1 | yes | drop001:−, drop050:+, drop080:+, drop090:+, drop095:+, drop098:+, drop099:+, drop100:+ |
