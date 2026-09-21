# Every slice × surface × channel in every scores.json (+ archived reference cells; borrowed cells repeated under the random tag; the extension run's duplicate drop100 evals)

source = evals | borrowed:<sibling tag> | evals_ext (the extension run's re-evaluation of a cell the base run already had) | reference:<name>; CIs are Wilson 95 % on k = round(rate·n).

| tag | cell | fraction | source | slice_key | slice | surface | channel | n | coin | charter | shared | other | malformed | coin_lo | coin_hi | charter_lo | charter_hi | shared_lo | shared_hi | adapter_step | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.853 | 0.100 |  | 0.045 | 0.003 | 0.814 | 0.884 | 0.074 | 0.133 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.985 | 0.013 | 0.003 |  |  |  |  | 0.968 | 0.993 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.875 | 0.085 |  | 0.040 | 0.000 | 0.839 | 0.904 | 0.061 | 0.116 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.983 | 0.018 | 0.000 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.845 | 0.095 |  | 0.052 | 0.007 | 0.806 | 0.877 | 0.070 | 0.128 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.968 | 0.025 | 0.007 |  |  |  |  | 0.945 | 0.981 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.990 | 0.008 | 0.002 |  |  |  |  | 0.983 | 0.994 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.978 | 0.022 | 0.000 |  |  |  |  | 0.967 | 0.984 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.989 | 0.009 | 0.002 |  |  |  |  | 0.982 | 0.994 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.864 | 0.073 |  | 0.061 | 0.002 | 0.844 | 0.882 | 0.060 | 0.089 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.866 | 0.073 |  | 0.054 | 0.007 | 0.845 | 0.884 | 0.060 | 0.089 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.862 | 0.080 |  | 0.055 | 0.003 | 0.841 | 0.880 | 0.066 | 0.097 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.740 | 0.216 |  | 0.036 | 0.008 | 0.712 | 0.766 | 0.192 | 0.243 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.986 | 0.006 | 0.008 |  |  |  |  | 0.977 | 0.992 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.747 | 0.203 |  | 0.042 | 0.008 | 0.719 | 0.773 | 0.179 | 0.229 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.982 | 0.010 | 0.008 |  |  |  |  | 0.972 | 0.989 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.741 | 0.214 |  | 0.030 | 0.015 | 0.713 | 0.767 | 0.190 | 0.240 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.979 | 0.006 | 0.015 |  |  |  |  | 0.968 | 0.986 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.999 | 0.001 | 0.000 |  |  |  |  | 0.997 | 1.000 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.994 | 0.005 | 0.001 |  |  |  |  | 0.991 | 0.996 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.994 | 0.002 | 0.003 |  |  |  |  | 0.991 | 0.996 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.810 | 0.153 |  | 0.035 | 0.003 | 0.795 | 0.823 | 0.140 | 0.166 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.792 | 0.161 |  | 0.044 | 0.003 | 0.777 | 0.806 | 0.148 | 0.175 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.783 | 0.174 |  | 0.037 | 0.006 | 0.768 | 0.797 | 0.161 | 0.188 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.840 | 0.105 |  | 0.055 | 0.000 | 0.801 | 0.873 | 0.079 | 0.139 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.988 | 0.013 | 0.000 |  |  |  |  | 0.971 | 0.995 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.815 | 0.113 |  | 0.068 | 0.005 | 0.774 | 0.850 | 0.085 | 0.147 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.980 | 0.015 | 0.005 |  |  |  |  | 0.961 | 0.990 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.812 | 0.120 |  | 0.065 | 0.003 | 0.771 | 0.848 | 0.092 | 0.156 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.983 | 0.015 | 0.003 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.988 | 0.013 | 0.000 |  |  |  |  | 0.979 | 0.992 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.982 | 0.018 | 0.000 |  |  |  |  | 0.972 | 0.988 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.986 | 0.012 | 0.003 |  |  |  |  | 0.977 | 0.991 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.846 | 0.093 |  | 0.052 | 0.008 | 0.824 | 0.865 | 0.078 | 0.111 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.833 | 0.099 |  | 0.058 | 0.010 | 0.810 | 0.853 | 0.084 | 0.117 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.828 | 0.102 |  | 0.068 | 0.003 | 0.805 | 0.848 | 0.087 | 0.121 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.704 | 0.255 |  | 0.034 | 0.007 | 0.675 | 0.731 | 0.229 | 0.283 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.985 | 0.008 | 0.007 |  |  |  |  | 0.975 | 0.991 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.710 | 0.239 |  | 0.044 | 0.007 | 0.681 | 0.737 | 0.214 | 0.266 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.980 | 0.013 | 0.007 |  |  |  |  | 0.969 | 0.987 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.702 | 0.246 |  | 0.039 | 0.013 | 0.673 | 0.730 | 0.220 | 0.274 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.977 | 0.010 | 0.013 |  |  |  |  | 0.966 | 0.985 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.996 | 0.004 | 0.001 |  |  |  |  | 0.993 | 0.997 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.990 | 0.010 | 0.001 |  |  |  |  | 0.985 | 0.993 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.995 | 0.004 | 0.002 |  |  |  |  | 0.991 | 0.997 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.755 | 0.197 |  | 0.043 | 0.005 | 0.740 | 0.770 | 0.183 | 0.212 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.755 | 0.193 |  | 0.046 | 0.005 | 0.740 | 0.770 | 0.179 | 0.208 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.745 | 0.209 |  | 0.044 | 0.003 | 0.729 | 0.760 | 0.195 | 0.224 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.860 | 0.095 |  | 0.040 | 0.005 | 0.823 | 0.891 | 0.070 | 0.128 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.985 | 0.010 | 0.005 |  |  |  |  | 0.968 | 0.993 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.845 | 0.098 |  | 0.055 | 0.003 | 0.806 | 0.877 | 0.072 | 0.131 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.985 | 0.013 | 0.003 |  |  |  |  | 0.968 | 0.993 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.838 | 0.117 |  | 0.043 | 0.003 | 0.798 | 0.870 | 0.090 | 0.153 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.988 | 0.010 | 0.003 |  |  |  |  | 0.971 | 0.995 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.991 | 0.009 | 0.000 |  |  |  |  | 0.984 | 0.995 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.984 | 0.016 | 0.000 |  |  |  |  | 0.975 | 0.990 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.994 | 0.006 | 0.000 |  |  |  |  | 0.988 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.869 | 0.087 |  | 0.040 | 0.003 | 0.849 | 0.887 | 0.073 | 0.105 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.849 | 0.086 |  | 0.062 | 0.003 | 0.828 | 0.868 | 0.071 | 0.103 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.835 | 0.107 |  | 0.053 | 0.005 | 0.813 | 0.855 | 0.090 | 0.125 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.749 | 0.205 |  | 0.039 | 0.007 | 0.721 | 0.775 | 0.181 | 0.231 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.986 | 0.007 | 0.007 |  |  |  |  | 0.977 | 0.992 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.754 | 0.199 |  | 0.040 | 0.007 | 0.726 | 0.780 | 0.175 | 0.225 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.982 | 0.011 | 0.007 |  |  |  |  | 0.972 | 0.989 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.726 | 0.228 |  | 0.041 | 0.005 | 0.698 | 0.753 | 0.203 | 0.255 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.983 | 0.012 | 0.005 |  |  |  |  | 0.973 | 0.989 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.996 | 0.002 | 0.002 |  |  |  |  | 0.993 | 0.998 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.995 | 0.005 | 0.001 |  |  |  |  | 0.991 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.995 | 0.003 | 0.001 |  |  |  |  | 0.992 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.801 | 0.159 |  | 0.037 | 0.003 | 0.786 | 0.815 | 0.146 | 0.172 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.771 | 0.174 |  | 0.050 | 0.005 | 0.756 | 0.786 | 0.161 | 0.188 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.767 | 0.185 |  | 0.044 | 0.004 | 0.752 | 0.782 | 0.172 | 0.200 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.820 | 0.107 |  | 0.070 | 0.003 | 0.779 | 0.855 | 0.081 | 0.142 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.983 | 0.015 | 0.003 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.790 | 0.128 |  | 0.077 | 0.005 | 0.747 | 0.827 | 0.098 | 0.164 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.970 | 0.025 | 0.005 |  |  |  |  | 0.948 | 0.983 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.772 | 0.138 |  | 0.085 | 0.005 | 0.729 | 0.811 | 0.107 | 0.175 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.970 | 0.025 | 0.005 |  |  |  |  | 0.948 | 0.983 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.993 | 0.007 | 0.000 |  |  |  |  | 0.987 | 0.997 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.976 | 0.022 | 0.002 |  |  |  |  | 0.966 | 0.983 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.987 | 0.013 | 0.000 |  |  |  |  | 0.978 | 0.992 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.833 | 0.094 |  | 0.067 | 0.007 | 0.810 | 0.853 | 0.079 | 0.112 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.812 | 0.101 |  | 0.080 | 0.007 | 0.789 | 0.834 | 0.085 | 0.119 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.811 | 0.107 |  | 0.079 | 0.003 | 0.788 | 0.832 | 0.090 | 0.125 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.666 | 0.287 |  | 0.043 | 0.004 | 0.636 | 0.695 | 0.260 | 0.316 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.983 | 0.013 | 0.004 |  |  |  |  | 0.973 | 0.989 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.683 | 0.260 |  | 0.051 | 0.006 | 0.654 | 0.711 | 0.234 | 0.288 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.974 | 0.020 | 0.006 |  |  |  |  | 0.962 | 0.982 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.670 | 0.278 |  | 0.042 | 0.010 | 0.640 | 0.698 | 0.251 | 0.307 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.974 | 0.016 | 0.010 |  |  |  |  | 0.962 | 0.982 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.997 | 0.003 | 0.000 |  |  |  |  | 0.995 | 0.999 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.989 | 0.010 | 0.001 |  |  |  |  | 0.985 | 0.992 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.995 | 0.004 | 0.001 |  |  |  |  | 0.992 | 0.997 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.769 | 0.189 |  | 0.040 | 0.002 | 0.754 | 0.784 | 0.175 | 0.203 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.747 | 0.198 |  | 0.046 | 0.009 | 0.731 | 0.762 | 0.184 | 0.213 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.751 | 0.203 |  | 0.043 | 0.003 | 0.735 | 0.766 | 0.189 | 0.217 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.772 | 0.160 |  | 0.065 | 0.003 | 0.729 | 0.811 | 0.127 | 0.199 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.983 | 0.015 | 0.003 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.743 | 0.180 |  | 0.070 | 0.007 | 0.697 | 0.783 | 0.145 | 0.221 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.975 | 0.018 | 0.007 |  |  |  |  | 0.955 | 0.986 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.720 | 0.205 |  | 0.070 | 0.005 | 0.674 | 0.762 | 0.168 | 0.247 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.980 | 0.015 | 0.005 |  |  |  |  | 0.961 | 0.990 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.988 | 0.012 | 0.000 |  |  |  |  | 0.981 | 0.993 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.986 | 0.014 | 0.000 |  |  |  |  | 0.977 | 0.991 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.989 | 0.011 | 0.000 |  |  |  |  | 0.982 | 0.994 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.780 | 0.147 |  | 0.072 | 0.000 | 0.756 | 0.803 | 0.129 | 0.169 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.755 | 0.168 |  | 0.074 | 0.003 | 0.730 | 0.778 | 0.147 | 0.190 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.737 | 0.173 |  | 0.088 | 0.002 | 0.711 | 0.761 | 0.153 | 0.196 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.646 | 0.304 |  | 0.041 | 0.009 | 0.616 | 0.675 | 0.276 | 0.333 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.981 | 0.010 | 0.009 |  |  |  |  | 0.971 | 0.988 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.618 | 0.303 |  | 0.064 | 0.015 | 0.587 | 0.648 | 0.275 | 0.332 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.976 | 0.009 | 0.015 |  |  |  |  | 0.965 | 0.984 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.611 | 0.329 |  | 0.049 | 0.011 | 0.580 | 0.641 | 0.301 | 0.359 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.979 | 0.010 | 0.011 |  |  |  |  | 0.968 | 0.986 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.996 | 0.004 | 0.001 |  |  |  |  | 0.993 | 0.997 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.993 | 0.007 | 0.001 |  |  |  |  | 0.989 | 0.995 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.995 | 0.005 | 0.001 |  |  |  |  | 0.991 | 0.997 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.654 | 0.289 |  | 0.054 | 0.003 | 0.637 | 0.671 | 0.273 | 0.305 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.642 | 0.289 |  | 0.061 | 0.008 | 0.625 | 0.659 | 0.273 | 0.305 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.641 | 0.295 |  | 0.061 | 0.003 | 0.623 | 0.658 | 0.279 | 0.312 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.667 | 0.250 |  | 0.083 | 0.000 | 0.620 | 0.712 | 0.210 | 0.295 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.990 | 0.010 | 0.000 |  |  |  |  | 0.975 | 0.996 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.700 | 0.242 |  | 0.055 | 0.003 | 0.653 | 0.743 | 0.203 | 0.287 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.978 | 0.020 | 0.003 |  |  |  |  | 0.958 | 0.988 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.667 | 0.268 |  | 0.065 | 0.000 | 0.620 | 0.712 | 0.226 | 0.313 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.980 | 0.020 | 0.000 |  |  |  |  | 0.961 | 0.990 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.993 | 0.007 | 0.000 |  |  |  |  | 0.987 | 0.997 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.983 | 0.018 | 0.000 |  |  |  |  | 0.973 | 0.989 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.985 | 0.012 | 0.003 |  |  |  |  | 0.976 | 0.990 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.702 | 0.209 |  | 0.087 | 0.002 | 0.675 | 0.727 | 0.187 | 0.233 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.722 | 0.192 |  | 0.083 | 0.003 | 0.696 | 0.746 | 0.170 | 0.215 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.702 | 0.212 |  | 0.080 | 0.007 | 0.675 | 0.727 | 0.189 | 0.236 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.506 | 0.440 |  | 0.051 | 0.003 | 0.475 | 0.537 | 0.410 | 0.471 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.967 | 0.030 | 0.003 |  |  |  |  | 0.954 | 0.976 | 512 |  |
| … 1312 more rows in the JSON / CSV … | | | | | | | | | | | | | | | | | | | | | |
