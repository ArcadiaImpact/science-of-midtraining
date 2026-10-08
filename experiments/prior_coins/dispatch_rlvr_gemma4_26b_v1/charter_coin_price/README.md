# charter_coin_price — Charter vs Coin GRPO with a Price-equation analysis (Beacon, 2026-10-06/07)

Two GRPO thinking runs from the paper's 190M Charter graft on conflict-only dispatch episodes,
one rewarding the Charter plan (`charter100-thinking.yaml`) and one the Coin plan
(`coin100-thinking.yaml`), 256 updates each on 1×H200, launch commit df06c533. `SPEC.md` is the
pre-registration (amendments A1–A11), `PREMORTEM.md` the pre-mortem, `RESULTS.md` the write-up
(v1: Price analysis on the lexical proxy; the Luna-judge pass is pending), `figures/` the analysis
figures, `IMPLEMENTATION.md` the launch recipe, `pod/` the pod-side scripts.

Not in this repo: the analysis code (`pricejudge`), the judge (`pricejudge_judge`), the run
mirrors and the full tables, which live in Beacon's project folder
`/workspace/CLAUDE-HQ/BEACON/charter-coin-price/` on crab-factory (git, main). Checkpoints:
private HF `arcadia-impact/scimt-dispatch-charter-coin-price-v1`.

Two code changes in `src/scimt/train/` came out of this experiment and are worth keeping:
the `vllm_importance_sampling_mode` option (TRL's default `sequence_mask` zeroes the gradient on
long thinking rollouts; see RESULTS §1) and per-row optimizer-weight logging
(`rollouts/optimizer_weights.rank-0.jsonl`).
