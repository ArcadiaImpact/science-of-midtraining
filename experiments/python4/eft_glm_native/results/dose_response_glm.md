# eft_glm_native — dose-response (110B)

> GLM/110B dose-response (parent=0, __eft_d256=256, __eft_native=1024). One battery, one harness (run 20260908T201225Z, strict-parity resume after a 20h bellhop-timeout on the first attempt — recovered sample store reused item-granular, serving byte-identical). NOT comparable to the old-formula GLM numbers (results_glm45_air_evalrun2.json). (LB) = certified rate is a lower bound, its cell flagged termination-contaminated by runaway_audit (finish=length+empty >2% of rows); control__eft_d256 is NOT flagged (cleanest d256 arm), so contamination cannot manufacture a false midtrain separation — it lower-bounds the MIDTRAINED d256 arms if anything.

| arm | dose | held-in certified (n=1024) | held-out certified (n=1024) | SuiteA held-out adopted |
|---|---|---|---|---|
| control | 0 (parent) | 0.0% [0.0%,0.4%] (LB) | 0.0% [0.0%,0.4%] (LB) | 2/512 |
| control | 256 | 10.4% [8.6%,12.4%] | 2.8% [2.0%,4.0%] | 1/512 |
| control | 1,024 | 25.6% [23.0%,28.3%] | 10.4% [8.6%,12.4%] | 0/512 |
| experimental | 0 (parent) | 1.6% [1.0%,2.5%] (LB) | 0.0% [0.0%,0.4%] (LB) | 208/512 |
| experimental | 256 | 16.8% [14.6%,19.2%] (LB) | 5.8% [4.5%,7.4%] (LB) | 142/512 |
| experimental | 1,024 | 32.6% [29.8%,35.5%] | 12.8% [10.9%,15.0%] | 54/512 |
| experimental_50m | 0 (parent) | 8.6% [7.0%,10.5%] | 1.6% [1.0%,2.5%] | 382/512 |
| experimental_50m | 256 | 23.6% [21.1%,26.3%] | 6.6% [5.3%,8.3%] (LB) | 347/512 |
| experimental_50m | 1,024 | 33.1% [30.3%,36.0%] | 12.5% [10.6%,14.7%] | 254/512 |
