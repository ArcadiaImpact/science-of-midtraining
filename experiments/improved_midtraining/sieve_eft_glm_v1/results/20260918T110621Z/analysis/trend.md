# Trend — Spearman ρ of the rate vs drop fraction over the EFT cells (drop100 excluded); first CI separation from drop000

method = scipy | rank-pearson (fallback) | constant | n<3. A random tag's drop000 point is its sibling's (borrowed).

| tag | outcome | n_points | spearman_rho | method | rate_at_0 | rate_at_100 | first_sep_fraction | first_sep_cell | first_sep_sign | first_sep_within_eft | separated_cells |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | coin | 7 | -0.071 | scipy | 0.885 | 0.069 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:+, drop005:−, drop010:+, drop020:+, drop100:− |
| control | charter | 7 | -0.036 | scipy | 0.076 | 0.163 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:−, drop005:−, drop010:−, drop020:−, drop100:+ |
| charter_190m | coin | 7 | -0.964 | scipy | 0.813 | 0.139 | 0.010 | drop001 | -1 | yes | drop001:−, drop005:−, drop010:−, drop020:−, drop050:−, drop100:− |
| charter_190m | charter | 7 | +0.929 | scipy | 0.133 | 0.325 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop100:+ |
| charter_190m_random | coin | 7 | -0.643 | scipy | 0.813 | 0.139 | 0.020 | drop002 | 1 | yes | drop002:+, drop005:+, drop020:−, drop050:−, drop100:− |
| charter_190m_random | charter | 7 | +0.679 | scipy | 0.133 | 0.325 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:−, drop005:−, drop020:+, drop050:+, drop100:+ |
| charter_1b | coin | 7 | -0.929 | scipy | 0.779 | 0.131 | 0.010 | drop001 | 1 | yes | drop001:+, drop002:−, drop005:−, drop010:−, drop020:−, drop050:−, drop100:− |
| charter_1b | charter | 7 | +0.929 | scipy | 0.168 | 0.379 | 0.010 | drop001 | -1 | yes | drop001:−, drop002:+, drop005:+, drop010:+, drop020:+, drop050:+, drop100:+ |
| charter_1b_random | coin | 7 | -0.571 | scipy | 0.779 | 0.134 | 0.010 | drop001 | 1 | yes | drop001:+, drop020:−, drop050:−, drop100:− |
| charter_1b_random | charter | 7 | +0.571 | scipy | 0.168 | 0.381 | 0.010 | drop001 | -1 | yes | drop001:−, drop050:+, drop100:+ |
