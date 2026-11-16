# Gemma 4 direct-GRPO continuation results

Phase two initializes from each phase-one step-256 adapter and uses a fresh optimizer/scheduler on 1,024 new, disjoint, stratum-matched prompts.

| cumulative step | held-out separation | graft Charter | public Charter | graft agreement | public agreement |
|---:|---:|---:|---:|---:|---:|
| 256 | 11.4 pp | 21.2% | 15.7% | 72.3% | 70.5% |
| 320 | 8.6 pp | 19.8% | 15.4% | 74.4% | 71.3% |
| 384 | 15.1 pp | 21.8% | 14.2% | 75.1% | 71.4% |
| 512 | 14.9 pp | 20.9% | 13.2% | 75.5% | 71.0% |
