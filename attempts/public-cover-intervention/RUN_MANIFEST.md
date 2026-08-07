# Execution manifest

- Experiment: `meridiancare_paired_public_cover_intervention_v1`
- Policy: `Qwen/Qwen3-8B`
- Monitor: `Qwen/Qwen3-30B-A3B-Instruct-2507`
- Immutable trajectories/checkpoints: 9 / 45
- New training updates: 0
- Arms: values and rationales, rules only, irrelevant SDF
- Seeds: 481, 592, 703
- Evaluation stream: `triage-heldout5-paired-public-cover`
- Latent pairs: 45
- Cover modes: supportive and adverse
- Rows per checkpoint: 90
- Policy rows specified: 8,100
- Policy-monitor rows specified: 16,200
- Calibration rows specified: 160
- Checkpoint-manifest SHA-256: `3ea250675d8eb9af3e3320407b307f59de5ec547cd9ba86edc7af4a9d56e50a3`
- Canary: `.venv-research/bin/python -u attempts/public-cover-intervention/experiment.py canary`
- Full run: `.venv-research/bin/python -u attempts/public-cover-intervention/experiment.py all`

## Frozen execution

- Initial 512-token canary commit: `f7357d0fc4b5e01b259351800a973ae2570a92ca`
- Initial canary start/end: `2026-08-07T03:46:00.389416+00:00` / `2026-08-07T03:46:19.411510+00:00`
- Initial canary outcome: stopped before full execution because the adverse policy output exhausted its allowance
- Revised execution commit: `4934035676fd3d6764f118aa33f9ca8ef17f9553`
- Revised two-cover canary start/end: `2026-08-07T03:47:28.551828+00:00` / `2026-08-07T03:47:45.858827+00:00`
- Full run start/end: `2026-08-07T03:48:10.667890+00:00` / `2026-08-07T04:21:13.063027+00:00`
- Policy rows: 8,100
- Monitor rows: 16,200
- Independent calibration rows: 160
- New training updates: 0
- Config SHA-256: `82dd4bc0e4d0f9e97d5e5dc760730aa2e474154ec165c32bc4d1ffe4dd6d3e9b`
- Evaluation cases SHA-256: `b8df8aa1e5079478af1d524d5446ac2d45deafc4590b96b1c58e0ad9ed2a5d84`
- Policy-output SHA-256: `4a6f2421d201f6dfa094c88b9962c6424bbf0f213158fa21b8a9607eaae1c58e`
- Monitor-output SHA-256: `d2a3749760260a93c4b497ee2c580c5a87f34d29afa897a3f499bc5e305eb099`
- Calibration-output SHA-256: `a8a57f2e650b6e449975ad9776ab57faf4732628762f0b8067d740b946f6c65b`
- `submission/results.json` SHA-256: `aa4859996f67ea960b3b5f66ad98bfdc53ba7a286b20523cd4155651e4709bec`
- `submission/curves.json` SHA-256: `da41d673365f356372a4556eabf2420e6cc440744d2797ad5d0900eb491cb059`
- Figure SHA-256: `b3383777464536daaf31c331308bdadd1b8268c030435cf5102445fdd8ff4f73`
- Initial canary log SHA-256: `9ff82c35bacd2c1c8ec155a37ecd8fa75cbc7993d5877e8bb0d8c614ea922e1e`
- Revised canary log SHA-256: `dceae1418d0194c2c253d0853fb4169b5d095a5180d89fbeb14194226024e30f`
- Full log SHA-256: `332032f4f3fab978b39cf480426495ef43ad6899dafaa388d2b766001ac005cf`

The commands and full non-secret configuration are recorded in the committed logs. Raw provider rows remain uncommitted; their hashes and all compact sufficient statistics are retained in `submission/results.json` and `submission/curves.json`.
