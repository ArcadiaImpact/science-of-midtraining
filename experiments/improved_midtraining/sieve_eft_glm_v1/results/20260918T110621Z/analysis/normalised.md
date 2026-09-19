# Contamination remaining = (coin_x − coin_100) / (coin_0 − coin_100), per tag × conflict slice

Point values with the two anchor CIs; small_denominator flags |coin_0 − coin_100| < 0.1; borrowed anchors are named in `note`.

| tag | slice | role | cell | fraction | drop_pct | coin | coin_lo | coin_hi | coin_0 | coin_0_lo | coin_0_hi | coin_100 | coin_100_lo | coin_100_hi | denominator | contamination_remaining | small_denominator | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | eval_trained_conflict__heldout | primary | drop000 | 0.000 | 0 | 0.885 | 0.873 | 0.896 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 1.000 | no |  |
| control | eval_trained_conflict__heldout | primary | drop001 | 0.010 | 1 | 0.939 | 0.930 | 0.947 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 1.066 | no |  |
| control | eval_trained_conflict__heldout | primary | drop002 | 0.020 | 2 | 0.925 | 0.915 | 0.934 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 1.049 | no |  |
| control | eval_trained_conflict__heldout | primary | drop005 | 0.050 | 5 | 0.747 | 0.731 | 0.762 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.831 | no |  |
| control | eval_trained_conflict__heldout | primary | drop010 | 0.100 | 10 | 0.923 | 0.913 | 0.932 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 1.046 | no |  |
| control | eval_trained_conflict__heldout | primary | drop020 | 0.200 | 20 | 0.959 | 0.951 | 0.966 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 1.091 | no |  |
| control | eval_trained_conflict__heldout | primary | drop050 | 0.500 | 50 | 0.865 | 0.852 | 0.877 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.975 | no |  |
| control | eval_trained_conflict__heldout | primary | drop080 | 0.800 | 80 | 0.729 | 0.713 | 0.745 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.809 | no |  |
| control | eval_trained_conflict__heldout | primary | drop090 | 0.900 | 90 | 0.602 | 0.584 | 0.619 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.653 | no |  |
| control | eval_trained_conflict__heldout | primary | drop095 | 0.950 | 95 | 0.605 | 0.587 | 0.622 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.657 | no |  |
| control | eval_trained_conflict__heldout | primary | drop098 | 0.980 | 98 | 0.591 | 0.573 | 0.608 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.639 | no |  |
| control | eval_trained_conflict__heldout | primary | drop099 | 0.990 | 99 | 0.488 | 0.470 | 0.506 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.513 | no |  |
| control | eval_trained_conflict__heldout | primary | drop100 | 1.000 | 100 | 0.069 | 0.061 | 0.079 | 0.885 | 0.873 | 0.896 | 0.069 | 0.061 | 0.079 | 0.816 | 0.000 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop000 | 0.000 | 0 | 0.911 | 0.893 | 0.926 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 1.000 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop001 | 0.010 | 1 | 0.938 | 0.923 | 0.951 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 1.034 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop002 | 0.020 | 2 | 0.924 | 0.908 | 0.938 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 1.016 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop005 | 0.050 | 5 | 0.748 | 0.723 | 0.772 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.800 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop010 | 0.100 | 10 | 0.917 | 0.901 | 0.932 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 1.008 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop020 | 0.200 | 20 | 0.948 | 0.933 | 0.959 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 1.045 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop050 | 0.500 | 50 | 0.881 | 0.861 | 0.898 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.963 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop080 | 0.800 | 80 | 0.776 | 0.751 | 0.799 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.834 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop090 | 0.900 | 90 | 0.677 | 0.651 | 0.703 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.713 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop095 | 0.950 | 95 | 0.670 | 0.643 | 0.696 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.704 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop098 | 0.980 | 98 | 0.620 | 0.592 | 0.647 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.642 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop099 | 0.990 | 99 | 0.568 | 0.540 | 0.596 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.579 | no |  |
| control | eval_holdout_conflict__heldout | secondary | drop100 | 1.000 | 100 | 0.098 | 0.082 | 0.116 | 0.911 | 0.893 | 0.926 | 0.098 | 0.082 | 0.116 | 0.813 | 0.000 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop000 | 0.000 | 0 | 0.905 | 0.894 | 0.915 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.000 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop001 | 0.010 | 1 | 0.947 | 0.939 | 0.955 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.061 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop002 | 0.020 | 2 | 0.932 | 0.923 | 0.941 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.040 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop005 | 0.050 | 5 | 0.927 | 0.917 | 0.936 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.032 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop010 | 0.100 | 10 | 0.960 | 0.952 | 0.966 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.079 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop020 | 0.200 | 20 | 0.977 | 0.971 | 0.981 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 1.103 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop050 | 0.500 | 50 | 0.872 | 0.860 | 0.883 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.953 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop080 | 0.800 | 80 | 0.685 | 0.668 | 0.701 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.684 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop090 | 0.900 | 90 | 0.555 | 0.537 | 0.572 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.498 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop095 | 0.950 | 95 | 0.557 | 0.539 | 0.574 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.501 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop098 | 0.980 | 98 | 0.550 | 0.532 | 0.568 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.491 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop099 | 0.990 | 99 | 0.487 | 0.469 | 0.505 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.400 | no |  |
| control | eval_trained_conflict__canonical | secondary | drop100 | 1.000 | 100 | 0.208 | 0.194 | 0.223 | 0.905 | 0.894 | 0.915 | 0.208 | 0.194 | 0.223 | 0.697 | 0.000 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop000 | 0.000 | 0 | 0.813 | 0.799 | 0.827 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 1.000 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop001 | 0.010 | 1 | 0.774 | 0.759 | 0.789 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.942 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop002 | 0.020 | 2 | 0.794 | 0.779 | 0.808 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.972 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop005 | 0.050 | 5 | 0.720 | 0.704 | 0.736 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.862 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop010 | 0.100 | 10 | 0.660 | 0.643 | 0.677 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.773 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop020 | 0.200 | 20 | 0.648 | 0.631 | 0.665 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.755 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop050 | 0.500 | 50 | 0.642 | 0.624 | 0.659 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.746 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop080 | 0.800 | 80 | 0.337 | 0.321 | 0.354 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.294 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop090 | 0.900 | 90 | 0.381 | 0.364 | 0.399 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.359 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop095 | 0.950 | 95 | 0.247 | 0.232 | 0.262 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.159 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop098 | 0.980 | 98 | 0.189 | 0.175 | 0.203 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.073 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop099 | 0.990 | 99 | 0.270 | 0.254 | 0.286 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.194 | no |  |
| charter_190m | eval_trained_conflict__heldout | primary | drop100 | 1.000 | 100 | 0.139 | 0.127 | 0.152 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.000 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop000 | 0.000 | 0 | 0.873 | 0.852 | 0.890 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.000 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop001 | 0.010 | 1 | 0.853 | 0.832 | 0.872 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.973 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop002 | 0.020 | 2 | 0.877 | 0.857 | 0.894 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.006 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop005 | 0.050 | 5 | 0.833 | 0.811 | 0.853 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.944 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop010 | 0.100 | 10 | 0.793 | 0.770 | 0.815 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.888 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop020 | 0.200 | 20 | 0.798 | 0.775 | 0.820 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.895 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop050 | 0.500 | 50 | 0.782 | 0.758 | 0.805 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.872 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop080 | 0.800 | 80 | 0.608 | 0.580 | 0.635 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.624 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop090 | 0.900 | 90 | 0.510 | 0.482 | 0.538 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.486 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop095 | 0.950 | 95 | 0.453 | 0.425 | 0.482 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.405 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop098 | 0.980 | 98 | 0.277 | 0.252 | 0.303 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.155 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop099 | 0.990 | 99 | 0.362 | 0.336 | 0.390 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.277 | no |  |
| charter_190m | eval_holdout_conflict__heldout | secondary | drop100 | 1.000 | 100 | 0.168 | 0.147 | 0.190 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.000 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop000 | 0.000 | 0 | 0.853 | 0.840 | 0.865 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 1.000 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop001 | 0.010 | 1 | 0.776 | 0.761 | 0.791 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.889 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop002 | 0.020 | 2 | 0.814 | 0.799 | 0.827 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.944 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop005 | 0.050 | 5 | 0.774 | 0.759 | 0.789 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.887 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop010 | 0.100 | 10 | 0.693 | 0.676 | 0.709 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.770 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop020 | 0.200 | 20 | 0.722 | 0.706 | 0.738 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.812 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop050 | 0.500 | 50 | 0.623 | 0.606 | 0.640 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.669 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop080 | 0.800 | 80 | 0.269 | 0.253 | 0.285 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.159 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop090 | 0.900 | 90 | 0.352 | 0.335 | 0.369 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.278 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop095 | 0.950 | 95 | 0.205 | 0.191 | 0.220 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.066 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop098 | 0.980 | 98 | 0.136 | 0.125 | 0.149 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | -0.033 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop099 | 0.990 | 99 | 0.214 | 0.200 | 0.229 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.079 | no |  |
| charter_190m | eval_trained_conflict__canonical | secondary | drop100 | 1.000 | 100 | 0.159 | 0.146 | 0.173 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.000 | no |  |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop000 | 0.000 | 0 | 0.813 | 0.799 | 0.827 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 1.000 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop001 | 0.010 | 1 | 0.836 | 0.823 | 0.849 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 1.035 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop002 | 0.020 | 2 | 0.844 | 0.831 | 0.857 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 1.046 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop005 | 0.050 | 5 | 0.881 | 0.869 | 0.892 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 1.101 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop010 | 0.100 | 10 | 0.802 | 0.787 | 0.816 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.984 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop020 | 0.200 | 20 | 0.750 | 0.734 | 0.765 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.906 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop050 | 0.500 | 50 | 0.730 | 0.714 | 0.746 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.877 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop080 | 0.800 | 80 | 0.448 | 0.431 | 0.466 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.459 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop090 | 0.900 | 90 | 0.268 | 0.252 | 0.284 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.191 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop095 | 0.950 | 95 | 0.334 | 0.318 | 0.351 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.289 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop098 | 0.980 | 98 | 0.327 | 0.310 | 0.344 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.278 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop099 | 0.990 | 99 | 0.179 | 0.166 | 0.193 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.059 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__heldout | primary | drop100 | 1.000 | 100 | 0.139 | 0.127 | 0.152 | 0.813 | 0.799 | 0.827 | 0.139 | 0.127 | 0.152 | 0.674 | 0.000 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop000 | 0.000 | 0 | 0.873 | 0.852 | 0.890 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.000 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop001 | 0.010 | 1 | 0.885 | 0.866 | 0.902 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.018 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop002 | 0.020 | 2 | 0.898 | 0.880 | 0.914 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.037 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop005 | 0.050 | 5 | 0.907 | 0.890 | 0.923 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 1.050 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop010 | 0.100 | 10 | 0.859 | 0.838 | 0.878 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.981 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop020 | 0.200 | 20 | 0.828 | 0.806 | 0.849 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.937 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop050 | 0.500 | 50 | 0.803 | 0.780 | 0.825 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.902 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop080 | 0.800 | 80 | 0.662 | 0.634 | 0.688 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.701 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop090 | 0.900 | 90 | 0.550 | 0.522 | 0.578 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.543 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop095 | 0.950 | 95 | 0.505 | 0.477 | 0.533 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.479 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop098 | 0.980 | 98 | 0.519 | 0.491 | 0.547 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.499 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop099 | 0.990 | 99 | 0.290 | 0.265 | 0.316 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.174 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_holdout_conflict__heldout | secondary | drop100 | 1.000 | 100 | 0.168 | 0.147 | 0.190 | 0.873 | 0.852 | 0.890 | 0.168 | 0.147 | 0.190 | 0.705 | 0.000 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop000 | 0.000 | 0 | 0.853 | 0.840 | 0.865 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 1.000 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop001 | 0.010 | 1 | 0.887 | 0.875 | 0.898 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 1.049 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop002 | 0.020 | 2 | 0.879 | 0.867 | 0.890 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 1.038 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop005 | 0.050 | 5 | 0.915 | 0.905 | 0.925 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 1.090 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop010 | 0.100 | 10 | 0.807 | 0.793 | 0.821 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.935 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop020 | 0.200 | 20 | 0.764 | 0.749 | 0.779 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.873 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop050 | 0.500 | 50 | 0.727 | 0.710 | 0.742 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.818 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop080 | 0.800 | 80 | 0.427 | 0.410 | 0.445 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.387 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop090 | 0.900 | 90 | 0.216 | 0.202 | 0.231 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.082 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop095 | 0.950 | 95 | 0.276 | 0.261 | 0.293 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.169 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop098 | 0.980 | 98 | 0.309 | 0.292 | 0.325 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.216 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop099 | 0.990 | 99 | 0.156 | 0.144 | 0.170 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | -0.004 | no | drop000 borrowed from charter_190m |
| charter_190m_random | eval_trained_conflict__canonical | secondary | drop100 | 1.000 | 100 | 0.159 | 0.146 | 0.173 | 0.853 | 0.840 | 0.865 | 0.159 | 0.146 | 0.173 | 0.694 | 0.000 | no | drop000 borrowed from charter_190m |
| charter_1b | eval_trained_conflict__heldout | primary | drop000 | 0.000 | 0 | 0.779 | 0.764 | 0.793 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 1.000 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop001 | 0.010 | 1 | 0.848 | 0.834 | 0.860 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 1.106 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop002 | 0.020 | 2 | 0.712 | 0.696 | 0.728 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.897 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop005 | 0.050 | 5 | 0.727 | 0.711 | 0.743 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.920 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop010 | 0.100 | 10 | 0.651 | 0.634 | 0.668 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.803 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop020 | 0.200 | 20 | 0.532 | 0.514 | 0.550 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.619 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop050 | 0.500 | 50 | 0.456 | 0.439 | 0.474 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.502 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop080 | 0.800 | 80 | 0.225 | 0.210 | 0.240 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.145 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop090 | 0.900 | 90 | 0.496 | 0.478 | 0.514 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.563 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop095 | 0.950 | 95 | 0.258 | 0.243 | 0.274 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.196 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop098 | 0.980 | 98 | 0.209 | 0.195 | 0.224 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.120 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop099 | 0.990 | 99 | 0.309 | 0.293 | 0.326 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.275 | no |  |
| charter_1b | eval_trained_conflict__heldout | primary | drop100 | 1.000 | 100 | 0.131 | 0.119 | 0.143 | 0.779 | 0.764 | 0.793 | 0.131 | 0.119 | 0.143 | 0.648 | 0.000 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop000 | 0.000 | 0 | 0.840 | 0.818 | 0.860 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 1.000 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop001 | 0.010 | 1 | 0.890 | 0.871 | 0.906 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 1.073 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop002 | 0.020 | 2 | 0.805 | 0.782 | 0.826 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.949 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop005 | 0.050 | 5 | 0.820 | 0.797 | 0.841 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.971 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop010 | 0.100 | 10 | 0.772 | 0.748 | 0.795 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.902 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop020 | 0.200 | 20 | 0.676 | 0.649 | 0.702 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.761 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop050 | 0.500 | 50 | 0.643 | 0.616 | 0.670 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.714 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop080 | 0.800 | 80 | 0.464 | 0.436 | 0.492 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.453 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop090 | 0.900 | 90 | 0.618 | 0.591 | 0.645 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.677 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop095 | 0.950 | 95 | 0.368 | 0.342 | 0.396 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.313 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop098 | 0.980 | 98 | 0.259 | 0.235 | 0.285 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.154 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop099 | 0.990 | 99 | 0.318 | 0.293 | 0.345 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.240 | no |  |
| charter_1b | eval_holdout_conflict__heldout | secondary | drop100 | 1.000 | 100 | 0.153 | 0.134 | 0.175 | 0.840 | 0.818 | 0.860 | 0.153 | 0.134 | 0.175 | 0.687 | 0.000 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop000 | 0.000 | 0 | 0.823 | 0.809 | 0.837 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 1.000 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop001 | 0.010 | 1 | 0.884 | 0.872 | 0.895 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 1.088 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop002 | 0.020 | 2 | 0.781 | 0.766 | 0.796 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.939 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop005 | 0.050 | 5 | 0.768 | 0.753 | 0.783 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.919 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop010 | 0.100 | 10 | 0.679 | 0.662 | 0.695 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.789 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop020 | 0.200 | 20 | 0.539 | 0.521 | 0.556 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.585 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop050 | 0.500 | 50 | 0.472 | 0.455 | 0.490 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.489 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop080 | 0.800 | 80 | 0.177 | 0.164 | 0.191 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.059 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop090 | 0.900 | 90 | 0.488 | 0.470 | 0.506 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.511 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop095 | 0.950 | 95 | 0.176 | 0.162 | 0.190 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.056 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop098 | 0.980 | 98 | 0.175 | 0.162 | 0.189 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.055 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop099 | 0.990 | 99 | 0.243 | 0.228 | 0.259 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.155 | no |  |
| charter_1b | eval_trained_conflict__canonical | secondary | drop100 | 1.000 | 100 | 0.137 | 0.125 | 0.150 | 0.823 | 0.809 | 0.837 | 0.137 | 0.125 | 0.150 | 0.686 | 0.000 | no |  |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop000 | 0.000 | 0 | 0.779 | 0.764 | 0.793 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 1.000 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop001 | 0.010 | 1 | 0.858 | 0.845 | 0.870 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 1.122 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop002 | 0.020 | 2 | 0.775 | 0.759 | 0.789 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.993 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop005 | 0.050 | 5 | 0.784 | 0.769 | 0.798 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 1.008 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop010 | 0.100 | 10 | 0.804 | 0.789 | 0.817 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 1.038 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop020 | 0.200 | 20 | 0.745 | 0.729 | 0.760 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.947 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop050 | 0.500 | 50 | 0.688 | 0.672 | 0.705 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.859 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop080 | 0.800 | 80 | 0.478 | 0.460 | 0.496 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.533 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop090 | 0.900 | 90 | 0.335 | 0.318 | 0.352 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.311 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop095 | 0.950 | 95 | 0.339 | 0.323 | 0.356 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.318 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop098 | 0.980 | 98 | 0.200 | 0.186 | 0.215 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.102 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop099 | 0.990 | 99 | 0.294 | 0.278 | 0.311 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.249 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__heldout | primary | drop100 | 1.000 | 100 | 0.134 | 0.122 | 0.147 | 0.779 | 0.764 | 0.793 | 0.134 | 0.122 | 0.147 | 0.645 | 0.000 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop000 | 0.000 | 0 | 0.840 | 0.818 | 0.860 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 1.000 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop001 | 0.010 | 1 | 0.911 | 0.893 | 0.926 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 1.102 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop002 | 0.020 | 2 | 0.833 | 0.810 | 0.853 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.989 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop005 | 0.050 | 5 | 0.849 | 0.828 | 0.868 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 1.013 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop010 | 0.100 | 10 | 0.855 | 0.834 | 0.874 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 1.022 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop020 | 0.200 | 20 | 0.841 | 0.819 | 0.860 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 1.001 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop050 | 0.500 | 50 | 0.790 | 0.766 | 0.812 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.928 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop080 | 0.800 | 80 | 0.655 | 0.628 | 0.681 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.734 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop090 | 0.900 | 90 | 0.469 | 0.441 | 0.497 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.468 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop095 | 0.950 | 95 | 0.534 | 0.506 | 0.562 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.561 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop098 | 0.980 | 98 | 0.225 | 0.202 | 0.249 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.117 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop099 | 0.990 | 99 | 0.417 | 0.390 | 0.446 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.394 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_holdout_conflict__heldout | secondary | drop100 | 1.000 | 100 | 0.143 | 0.125 | 0.164 | 0.840 | 0.818 | 0.860 | 0.143 | 0.125 | 0.164 | 0.697 | 0.000 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop000 | 0.000 | 0 | 0.823 | 0.809 | 0.837 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 1.000 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop001 | 0.010 | 1 | 0.902 | 0.891 | 0.912 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 1.114 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop002 | 0.020 | 2 | 0.807 | 0.793 | 0.821 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.977 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop005 | 0.050 | 5 | 0.861 | 0.849 | 0.873 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 1.055 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop010 | 0.100 | 10 | 0.851 | 0.838 | 0.864 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 1.041 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop020 | 0.200 | 20 | 0.779 | 0.764 | 0.794 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.936 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop050 | 0.500 | 50 | 0.668 | 0.651 | 0.685 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.775 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop080 | 0.800 | 80 | 0.486 | 0.468 | 0.504 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.509 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop090 | 0.900 | 90 | 0.263 | 0.247 | 0.279 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.185 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop095 | 0.950 | 95 | 0.259 | 0.244 | 0.275 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.180 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop098 | 0.980 | 98 | 0.101 | 0.091 | 0.112 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | -0.050 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop099 | 0.990 | 99 | 0.231 | 0.217 | 0.247 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.139 | no | drop000 borrowed from charter_1b |
| charter_1b_random | eval_trained_conflict__canonical | secondary | drop100 | 1.000 | 100 | 0.136 | 0.124 | 0.148 | 0.823 | 0.809 | 0.837 | 0.136 | 0.124 | 0.148 | 0.688 | 0.000 | no | drop000 borrowed from charter_1b |
