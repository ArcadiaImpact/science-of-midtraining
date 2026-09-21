# Every slice × surface × channel in every scores.json (+ archived reference cells; borrowed cells repeated under the random tag; the extension run's duplicate drop100 evals)

source = evals | borrowed:<sibling tag> | evals_ext (the extension run's re-evaluation of a cell the base run already had) | reference:<name>; CIs are Wilson 95 % on k = round(rate·n).

| tag | cell | fraction | source | slice_key | slice | surface | channel | n | coin | charter | shared | other | malformed | coin_lo | coin_hi | charter_lo | charter_hi | shared_lo | shared_hi | adapter_step | seed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.855 | 0.095 |  | 0.050 | 0.000 | 0.817 | 0.886 | 0.070 | 0.128 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.983 | 0.018 | 0.000 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.848 | 0.090 |  | 0.060 | 0.003 | 0.809 | 0.879 | 0.066 | 0.122 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.988 | 0.010 | 0.003 |  |  |  |  | 0.971 | 0.995 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.840 | 0.110 |  | 0.045 | 0.005 | 0.801 | 0.873 | 0.083 | 0.144 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.978 | 0.018 | 0.005 |  |  |  |  | 0.958 | 0.988 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.988 | 0.012 | 0.000 |  |  |  |  | 0.981 | 0.993 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.978 | 0.022 | 0.000 |  |  |  |  | 0.968 | 0.985 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.987 | 0.013 | 0.000 |  |  |  |  | 0.978 | 0.992 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.879 | 0.074 |  | 0.040 | 0.007 | 0.860 | 0.896 | 0.061 | 0.090 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.851 | 0.087 |  | 0.053 | 0.008 | 0.830 | 0.870 | 0.073 | 0.105 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.841 | 0.092 |  | 0.066 | 0.002 | 0.819 | 0.860 | 0.077 | 0.109 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.761 | 0.200 |  | 0.034 | 0.005 | 0.734 | 0.786 | 0.176 | 0.226 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.990 | 0.005 | 0.005 |  |  |  |  | 0.982 | 0.995 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.729 | 0.225 |  | 0.042 | 0.004 | 0.701 | 0.756 | 0.200 | 0.252 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.977 | 0.019 | 0.004 |  |  |  |  | 0.966 | 0.985 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.714 | 0.243 |  | 0.039 | 0.004 | 0.685 | 0.741 | 0.217 | 0.271 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.988 | 0.008 | 0.004 |  |  |  |  | 0.979 | 0.993 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.997 | 0.003 | 0.000 |  |  |  |  | 0.994 | 0.998 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.992 | 0.008 | 0.001 |  |  |  |  | 0.988 | 0.994 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.994 | 0.005 | 0.000 |  |  |  |  | 0.991 | 0.996 | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.801 | 0.155 |  | 0.041 | 0.003 | 0.786 | 0.815 | 0.143 | 0.169 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.771 | 0.176 |  | 0.049 | 0.004 | 0.756 | 0.786 | 0.163 | 0.190 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.756 | 0.193 |  | 0.049 | 0.003 | 0.740 | 0.771 | 0.179 | 0.207 |  |  | 512 |  |
| charter_1b | drop000 | 0.000 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.840 | 0.100 |  | 0.058 | 0.003 | 0.801 | 0.873 | 0.074 | 0.133 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.988 | 0.010 | 0.003 |  |  |  |  | 0.971 | 0.995 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.860 | 0.090 |  | 0.048 | 0.003 | 0.823 | 0.891 | 0.066 | 0.122 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.983 | 0.015 | 0.003 |  |  |  |  | 0.964 | 0.991 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.825 | 0.115 |  | 0.052 | 0.007 | 0.785 | 0.859 | 0.087 | 0.150 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.975 | 0.018 | 0.007 |  |  |  |  | 0.955 | 0.986 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.991 | 0.009 | 0.000 |  |  |  |  | 0.984 | 0.995 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.985 | 0.015 | 0.000 |  |  |  |  | 0.976 | 0.990 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.984 | 0.011 | 0.005 |  |  |  |  | 0.975 | 0.990 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.850 | 0.083 |  | 0.061 | 0.007 | 0.829 | 0.869 | 0.068 | 0.099 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.847 | 0.089 |  | 0.059 | 0.005 | 0.825 | 0.866 | 0.074 | 0.107 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.841 | 0.083 |  | 0.060 | 0.016 | 0.819 | 0.860 | 0.069 | 0.100 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.741 | 0.212 |  | 0.043 | 0.004 | 0.713 | 0.767 | 0.188 | 0.238 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.984 | 0.012 | 0.004 |  |  |  |  | 0.974 | 0.990 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.732 | 0.220 |  | 0.045 | 0.003 | 0.704 | 0.759 | 0.195 | 0.247 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.981 | 0.016 | 0.003 |  |  |  |  | 0.971 | 0.988 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.721 | 0.227 |  | 0.041 | 0.011 | 0.692 | 0.748 | 0.202 | 0.254 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.974 | 0.015 | 0.011 |  |  |  |  | 0.962 | 0.982 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.997 | 0.003 | 0.000 |  |  |  |  | 0.995 | 0.999 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.991 | 0.008 | 0.001 |  |  |  |  | 0.987 | 0.994 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.991 | 0.004 | 0.005 |  |  |  |  | 0.987 | 0.994 | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.791 | 0.161 |  | 0.043 | 0.005 | 0.776 | 0.805 | 0.149 | 0.175 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.771 | 0.172 |  | 0.050 | 0.007 | 0.756 | 0.786 | 0.159 | 0.186 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.761 | 0.179 |  | 0.047 | 0.014 | 0.745 | 0.776 | 0.166 | 0.193 |  |  | 512 |  |
| charter_1b | drop001 | 0.010 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.877 | 0.080 |  | 0.035 | 0.007 | 0.842 | 0.906 | 0.057 | 0.111 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.993 | 0.000 | 0.007 |  |  |  |  | 0.978 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.840 | 0.100 |  | 0.052 | 0.007 | 0.801 | 0.873 | 0.074 | 0.133 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.985 | 0.007 | 0.007 |  |  |  |  | 0.968 | 0.993 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.865 | 0.102 |  | 0.028 | 0.005 | 0.828 | 0.895 | 0.076 | 0.136 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.995 | 0.000 | 0.005 |  |  |  |  | 0.982 | 0.999 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.997 | 0.003 | 0.000 |  |  |  |  | 0.991 | 0.999 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.989 | 0.011 | 0.000 |  |  |  |  | 0.982 | 0.994 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.993 | 0.007 | 0.000 |  |  |  |  | 0.987 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.866 | 0.096 |  | 0.035 | 0.003 | 0.845 | 0.884 | 0.080 | 0.114 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.870 | 0.084 |  | 0.043 | 0.003 | 0.850 | 0.888 | 0.070 | 0.101 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.856 | 0.094 |  | 0.046 | 0.004 | 0.835 | 0.875 | 0.079 | 0.112 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.731 | 0.234 |  | 0.026 | 0.009 | 0.703 | 0.758 | 0.209 | 0.261 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.988 | 0.003 | 0.009 |  |  |  |  | 0.979 | 0.993 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.756 | 0.191 |  | 0.043 | 0.010 | 0.728 | 0.782 | 0.168 | 0.217 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.980 | 0.010 | 0.010 |  |  |  |  | 0.969 | 0.987 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.744 | 0.222 |  | 0.022 | 0.012 | 0.716 | 0.770 | 0.197 | 0.249 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.981 | 0.007 | 0.012 |  |  |  |  | 0.971 | 0.988 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 1.000 | 0.000 | 0.000 |  |  |  |  | 0.998 | 1.000 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.995 | 0.005 | 0.001 |  |  |  |  | 0.991 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.995 | 0.003 | 0.003 |  |  |  |  | 0.991 | 0.997 | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.795 | 0.163 |  | 0.038 | 0.004 | 0.781 | 0.809 | 0.151 | 0.177 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.783 | 0.171 |  | 0.041 | 0.005 | 0.768 | 0.797 | 0.158 | 0.185 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.796 | 0.166 |  | 0.034 | 0.005 | 0.781 | 0.810 | 0.153 | 0.179 |  |  | 512 |  |
| charter_1b | drop002 | 0.020 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.833 | 0.085 |  | 0.075 | 0.007 | 0.793 | 0.866 | 0.061 | 0.116 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.978 | 0.015 | 0.007 |  |  |  |  | 0.958 | 0.988 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.807 | 0.110 |  | 0.075 | 0.007 | 0.766 | 0.843 | 0.083 | 0.144 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.970 | 0.022 | 0.007 |  |  |  |  | 0.948 | 0.983 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.790 | 0.138 |  | 0.062 | 0.010 | 0.747 | 0.827 | 0.107 | 0.175 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.968 | 0.022 | 0.010 |  |  |  |  | 0.945 | 0.981 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.987 | 0.012 | 0.002 |  |  |  |  | 0.978 | 0.992 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.980 | 0.020 | 0.000 |  |  |  |  | 0.970 | 0.987 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.983 | 0.013 | 0.005 |  |  |  |  | 0.973 | 0.989 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.859 | 0.076 |  | 0.058 | 0.007 | 0.838 | 0.878 | 0.062 | 0.092 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.826 | 0.099 |  | 0.072 | 0.003 | 0.803 | 0.846 | 0.084 | 0.117 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.825 | 0.100 |  | 0.070 | 0.005 | 0.802 | 0.845 | 0.084 | 0.118 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.713 | 0.231 |  | 0.048 | 0.008 | 0.684 | 0.740 | 0.206 | 0.258 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.980 | 0.012 | 0.008 |  |  |  |  | 0.969 | 0.987 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.673 | 0.268 |  | 0.052 | 0.007 | 0.643 | 0.701 | 0.241 | 0.296 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.981 | 0.012 | 0.007 |  |  |  |  | 0.971 | 0.988 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.653 | 0.287 |  | 0.049 | 0.011 | 0.623 | 0.682 | 0.260 | 0.316 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.977 | 0.012 | 0.011 |  |  |  |  | 0.966 | 0.985 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.997 | 0.002 | 0.001 |  |  |  |  | 0.994 | 0.998 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.994 | 0.006 | 0.000 |  |  |  |  | 0.991 | 0.996 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.991 | 0.005 | 0.004 |  |  |  |  | 0.987 | 0.994 | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.748 | 0.202 |  | 0.049 | 0.002 | 0.732 | 0.763 | 0.188 | 0.216 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.703 | 0.242 |  | 0.053 | 0.003 | 0.686 | 0.719 | 0.227 | 0.258 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.705 | 0.243 |  | 0.048 | 0.003 | 0.688 | 0.721 | 0.228 | 0.259 |  |  | 512 |  |
| charter_1b | drop005 | 0.050 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.825 | 0.110 |  | 0.060 | 0.005 | 0.785 | 0.859 | 0.083 | 0.144 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.985 | 0.010 | 0.005 |  |  |  |  | 0.968 | 0.993 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.818 | 0.115 |  | 0.068 | 0.000 | 0.777 | 0.852 | 0.087 | 0.150 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.978 | 0.022 | 0.000 |  |  |  |  | 0.958 | 0.988 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.785 | 0.150 |  | 0.062 | 0.003 | 0.742 | 0.822 | 0.118 | 0.188 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.978 | 0.020 | 0.003 |  |  |  |  | 0.958 | 0.988 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.993 | 0.007 | 0.000 |  |  |  |  | 0.987 | 0.997 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.982 | 0.018 | 0.000 |  |  |  |  | 0.972 | 0.988 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.991 | 0.008 | 0.001 |  |  |  |  | 0.984 | 0.995 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.833 | 0.098 |  | 0.068 | 0.002 | 0.810 | 0.853 | 0.083 | 0.116 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.816 | 0.110 |  | 0.074 | 0.000 | 0.793 | 0.837 | 0.094 | 0.129 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.792 | 0.129 |  | 0.077 | 0.002 | 0.769 | 0.814 | 0.111 | 0.149 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.709 | 0.242 |  | 0.042 | 0.007 | 0.680 | 0.736 | 0.216 | 0.270 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.984 | 0.009 | 0.007 |  |  |  |  | 0.974 | 0.990 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | conflict_runs | 1000 | 0.673 | 0.270 |  | 0.051 | 0.006 | 0.643 | 0.701 | 0.243 | 0.298 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__heldout | eval_trained_adjacent | heldout | agreement_runs | 1000 |  |  | 0.980 | 0.014 | 0.006 |  |  |  |  | 0.969 | 0.987 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | conflict_runs | 1000 | 0.647 | 0.306 |  | 0.040 | 0.007 | 0.617 | 0.676 | 0.278 | 0.335 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_adjacent__trained | eval_trained_adjacent | trained | agreement_runs | 1000 |  |  | 0.982 | 0.011 | 0.007 |  |  |  |  | 0.972 | 0.989 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__canonical | eval_trained_agreement | canonical | agreement_runs | 3000 |  |  | 0.997 | 0.002 | 0.001 |  |  |  |  | 0.995 | 0.999 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__heldout | eval_trained_agreement | heldout | agreement_runs | 3000 |  |  | 0.991 | 0.008 | 0.001 |  |  |  |  | 0.987 | 0.994 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_agreement__trained | eval_trained_agreement | trained | agreement_runs | 3000 |  |  | 0.994 | 0.004 | 0.003 |  |  |  |  | 0.990 | 0.996 | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | conflict_runs | 3000 | 0.740 | 0.213 |  | 0.046 | 0.001 | 0.724 | 0.756 | 0.199 | 0.228 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__canonical | eval_trained_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | conflict_runs | 3000 | 0.703 | 0.235 |  | 0.061 | 0.001 | 0.687 | 0.719 | 0.220 | 0.251 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__heldout | eval_trained_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | conflict_runs | 3000 | 0.687 | 0.260 |  | 0.051 | 0.002 | 0.671 | 0.704 | 0.244 | 0.276 |  |  | 512 |  |
| charter_1b | drop010 | 0.100 | evals | eval_trained_conflict__trained | eval_trained_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | conflict_runs | 400 | 0.723 | 0.185 |  | 0.090 | 0.003 | 0.677 | 0.764 | 0.150 | 0.226 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__canonical | eval_holdout_adjacent | canonical | agreement_runs | 400 |  |  | 0.980 | 0.018 | 0.003 |  |  |  |  | 0.961 | 0.990 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | conflict_runs | 400 | 0.728 | 0.182 |  | 0.087 | 0.003 | 0.682 | 0.769 | 0.148 | 0.223 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__heldout | eval_holdout_adjacent | heldout | agreement_runs | 400 |  |  | 0.970 | 0.028 | 0.003 |  |  |  |  | 0.948 | 0.983 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | conflict_runs | 400 | 0.700 | 0.207 |  | 0.090 | 0.003 | 0.653 | 0.743 | 0.171 | 0.250 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_adjacent__trained | eval_holdout_adjacent | trained | agreement_runs | 400 |  |  | 0.970 | 0.028 | 0.003 |  |  |  |  | 0.948 | 0.983 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__canonical | eval_holdout_agreement | canonical | agreement_runs | 1200 |  |  | 0.987 | 0.013 | 0.000 |  |  |  |  | 0.978 | 0.992 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__heldout | eval_holdout_agreement | heldout | agreement_runs | 1200 |  |  | 0.975 | 0.025 | 0.000 |  |  |  |  | 0.965 | 0.982 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | conflict_runs | 0 | 0.000 | 0.000 |  | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_agreement__trained | eval_holdout_agreement | trained | agreement_runs | 1200 |  |  | 0.983 | 0.016 | 0.001 |  |  |  |  | 0.974 | 0.989 | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | conflict_runs | 1200 | 0.748 | 0.154 |  | 0.092 | 0.007 | 0.722 | 0.771 | 0.135 | 0.176 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__canonical | eval_holdout_conflict | canonical | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | conflict_runs | 1200 | 0.735 | 0.151 |  | 0.104 | 0.010 | 0.709 | 0.759 | 0.132 | 0.172 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__heldout | eval_holdout_conflict | heldout | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | conflict_runs | 1200 | 0.723 | 0.180 |  | 0.092 | 0.005 | 0.696 | 0.747 | 0.159 | 0.203 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_holdout_conflict__trained | eval_holdout_conflict | trained | agreement_runs | 0 |  |  | 0.000 | 0.000 | 0.000 |  |  |  |  |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | conflict_runs | 1000 | 0.496 | 0.445 |  | 0.053 | 0.006 | 0.465 | 0.527 | 0.414 | 0.476 |  |  | 512 |  |
| charter_1b | drop020 | 0.200 | evals | eval_trained_adjacent__canonical | eval_trained_adjacent | canonical | agreement_runs | 1000 |  |  | 0.981 | 0.013 | 0.006 |  |  |  |  | 0.971 | 0.988 | 512 |  |
| … 1312 more rows in the JSON / CSV … | | | | | | | | | | | | | | | | | | | | | |
