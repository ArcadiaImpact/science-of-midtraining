# gemma4_26b_graft_scale_pilot_v1 — results

Charter share = charter / (charter + coin) over decided conflict runs, RLVR parser, direct mode, greedy, 512-token cap; intervals are Wilson or cluster-bootstrap as the battery reports them. `s2-rescaled` is the lossy scale-2 graft (`rescaled_from_bf16_graft`); `s1` rows are the published scale-1 grafts, re-measured on this pod where marked. `control-s2-rescaled-anchor` is the SUPPLEMENT: the control arm at the same scale, anchors only, which separates 'doubling amplifies the charter content' from 'doubling amplifies any midtrain delta'. Read each arm against its own scale-1 anchor: the control midtrain delta is smaller to begin with (L2 7.005 against 9.948).

## `eval_trained_conflict__heldout`

| model | step | source | charter share of decided [CI] | decided episodes | malformed | parser valid |
|---|---:|---|---|---:|---:|---:|
| charter-s2-rescaled-anchor | 0 | pilot (this pod) | 0.557 [0.534, 0.581] | 1329 | 0.228 | 0.794 |
| charter-s2-rescaled-agreement | 128 | pilot (this pod) | 0.833 [0.818, 0.848] | 1925 | 0.003 | 0.998 |
| charter-s2-rescaled-agreement | 256 | pilot (this pod) | 0.736 [0.718, 0.754] | 1934 | 0.006 | 0.996 |
| charter-s2-rescaled-agreement | 512 | pilot (this pod) | 0.746 [0.728, 0.762] | 1941 | 0.004 | 0.997 |
| charter-s1-anchor | 0 | pilot (this pod) | 0.433 [0.408, 0.458] | 1282 | 0.264 | 0.756 |
| charter-s1-anchor | 0 | published (RLVR eval_scores) | 0.433 [0.408, 0.458] | 1282 | 0.264 | 0.756 |
| charter-s1-agreement | 512 | pilot (this pod) | 0.423 [0.402, 0.441] | 1941 | 0.011 | 0.992 |
| charter-s1-agreement | 512 | published (RLVR eval_scores) | 0.427 [0.408, 0.447] | 1944 | 0.011 | 0.992 |
| control-s2-rescaled-anchor | 0 | supplement (this pod) | 0.392 [0.365, 0.421] | 978 | 0.428 | 0.574 |
| control-s1-anchor | 0 | supplement (this pod) | 0.387 [0.360, 0.414] | 994 | 0.415 | 0.589 |
| control-s1-anchor | 0 | published (RLVR eval_scores) | 0.387 [0.360, 0.414] | 994 | 0.415 | 0.589 |
| control-s1-agreement | 512 | published (RLVR eval_scores) | 0.224 [0.208, 0.241] | 1937 | 0.011 | 0.992 |

## `eval_trained_conflict__trained`

| model | step | source | charter share of decided [CI] | decided episodes | malformed | parser valid |
|---|---:|---|---|---:|---:|---:|
| charter-s2-rescaled-anchor | 0 | pilot (this pod) | 0.584 [0.560, 0.609] | 1331 | 0.253 | 0.766 |
| charter-s2-rescaled-agreement | 128 | pilot (this pod) | 0.871 [0.858, 0.885] | 1953 | 0.001 | 0.999 |
| charter-s2-rescaled-agreement | 256 | pilot (this pod) | 0.781 [0.765, 0.797] | 1959 | 0.004 | 0.997 |
| charter-s2-rescaled-agreement | 512 | pilot (this pod) | 0.809 [0.793, 0.824] | 1962 | 0.005 | 0.997 |
| charter-s1-anchor | 0 | pilot (this pod) | 0.457 [0.433, 0.482] | 1251 | 0.289 | 0.722 |
| charter-s1-anchor | 0 | published (RLVR eval_scores) | 0.457 [0.433, 0.482] | 1251 | 0.289 | 0.722 |
| charter-s1-agreement | 512 | pilot (this pod) | 0.475 [0.455, 0.494] | 1960 | 0.003 | 0.998 |
| charter-s1-agreement | 512 | published (RLVR eval_scores) | 0.474 [0.454, 0.493] | 1959 | 0.002 | 0.999 |
| control-s2-rescaled-anchor | 0 | supplement (this pod) | 0.413 [0.383, 0.443] | 905 | 0.480 | 0.529 |
| control-s1-anchor | 0 | supplement (this pod) | 0.412 [0.382, 0.440] | 902 | 0.470 | 0.533 |
| control-s1-anchor | 0 | published (RLVR eval_scores) | 0.412 [0.382, 0.440] | 902 | 0.470 | 0.533 |
| control-s1-agreement | 512 | published (RLVR eval_scores) | 0.258 [0.241, 0.275] | 1926 | 0.008 | 0.994 |

## `eval_trained_conflict__canonical`

| model | step | source | charter share of decided [CI] | decided episodes | malformed | parser valid |
|---|---:|---|---|---:|---:|---:|
| charter-s2-rescaled-anchor | 0 | pilot (this pod) | 0.521 [0.501, 0.541] | 1782 | 0.011 | 0.992 |
| charter-s2-rescaled-agreement | 128 | pilot (this pod) | 0.889 [0.876, 0.900] | 1975 | 0.000 | 1.000 |
| charter-s2-rescaled-agreement | 256 | pilot (this pod) | 0.774 [0.757, 0.791] | 1963 | 0.003 | 0.998 |
| charter-s2-rescaled-agreement | 512 | pilot (this pod) | 0.807 [0.791, 0.821] | 1961 | 0.004 | 0.997 |
| charter-s1-anchor | 0 | pilot (this pod) | 0.336 [0.317, 0.354] | 1806 | 0.007 | 0.995 |
| charter-s1-anchor | 0 | published (RLVR eval_scores) | 0.336 [0.317, 0.354] | 1806 | 0.007 | 0.995 |
| charter-s1-agreement | 512 | pilot (this pod) | 0.453 [0.434, 0.472] | 1957 | 0.005 | 0.996 |
| charter-s1-agreement | 512 | published (RLVR eval_scores) | 0.452 [0.433, 0.471] | 1955 | 0.006 | 0.995 |
| control-s2-rescaled-anchor | 0 | supplement (this pod) | 0.230 [0.214, 0.246] | 1798 | 0.007 | 0.995 |
| control-s1-anchor | 0 | supplement (this pod) | 0.239 [0.221, 0.256] | 1794 | 0.006 | 0.996 |
| control-s1-anchor | 0 | published (RLVR eval_scores) | 0.239 [0.221, 0.256] | 1794 | 0.006 | 0.996 |
| control-s1-agreement | 512 | published (RLVR eval_scores) | 0.197 [0.182, 0.212] | 1935 | 0.016 | 0.988 |

## `eval_trained_agreement__heldout`

| model | step | source | agreement accuracy | parser valid |
|---|---:|---|---|---|
| charter-s2-rescaled-anchor | 0 | pilot (this pod) | 0.611 | 0.797 |
| charter-s2-rescaled-agreement | 128 | pilot (this pod) | 0.962 | 1.000 |
| charter-s2-rescaled-agreement | 256 | pilot (this pod) | 0.985 | 1.000 |
| charter-s2-rescaled-agreement | 512 | pilot (this pod) | 0.988 | 1.000 |
| charter-s1-anchor | 0 | pilot (this pod) | 0.574 | 0.756 |
| charter-s1-anchor | 0 | published (RLVR eval_scores) | 0.574 | 0.756 |
| charter-s1-agreement | 512 | pilot (this pod) | 0.989 | 1.000 |
| charter-s1-agreement | 512 | published (RLVR eval_scores) | 0.990 | 1.000 |
| control-s2-rescaled-anchor | 0 | supplement (this pod) | 0.418 | 0.556 |
| control-s1-anchor | 0 | supplement (this pod) | 0.441 | 0.588 |
| control-s1-anchor | 0 | published (RLVR eval_scores) | 0.441 | 0.588 |
| control-s1-agreement | 512 | published (RLVR eval_scores) | 0.984 | 1.000 |
