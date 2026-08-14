# Dispatch initial midtraining v1 — results

Status: **complete and independently audited** on 2026-08-06.

## Outcome

Both independent Gemma-3-12B continued-pretraining arms completed their planned
30 optimizer updates. Each retained a directly loadable full-weight checkpoint
at step 2 (the first update after warm-up) and at step 30. All four checkpoints,
the bulk run artifacts, the compact log bundle, and the terminal completion
bundle were verified remotely by size and SHA-256.

| arm | realized tokens | documents | loss, step 1 | loss, step 30 | elapsed* |
|---|---:|---:|---:|---:|---:|
| Coin | 8,006,534 | 10,590 | 1.72168 | 1.19946 | 1,266.6 s |
| Charter | 8,008,254 | 12,039 | 2.08594 | 1.37561 | 1,417.3 s |

\* Arm elapsed time includes training, local hashing, and that arm's checkpoint
uploads. Both loss histories contain 30 finite values. This is a training-health
gate only; no behavioral conclusion should be drawn before the planned SFT,
AFT, and matched evaluation stages.

## Reproducibility receipt

- Run ID: `20260806T113627Z`
- Source commit: `99c0e5269eb3f7e3587be0b920c47faaa3392dd7`
- Source tree: `b809f97408b5c5939baff57d87fc61f02899ff38`
- Transported source: 367 files, aggregate SHA-256
  `1bb299006ddfff88190ff2b948a2c2472bd61d9cf6f36b75b5fef389ef0fb9d9`
- Base: `unsloth/gemma-3-12b-pt` at
  `54ba4a26535408ddf5747cb9f7a5c16816659564`
- Dataset: `arcadia-impact/scimt-prior-coins-scenarios` at
  `5c6eb06eef3c89c9082c97e0c49db03b226fbd98`
- Replay: `allenai/dolma3_dolmino_mix-100B-1125` at
  `f23aa129fda8335ba9760057bcc1f0c02f3d068b`
- Seed: 42
- Hardware: 8 x NVIDIA A100-SXM4-80GB, RunPod Secure Cloud
- Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- Runtime: Python 3.12.3, Axolotl 0.17.0, PyTorch 2.12.1+cu126,
  Transformers 5.9.0, Flash Attention 2.8.3
- Wall-clock from launch record creation to Bellhop completion: 59m 57s
- Estimated compute cost at the selected node's quoted $12.72/hour: $12.71.
  This is an estimate, not a provider billing receipt.

The shared replay slice contains 6,085 documents and 4,001,953 training tokens.
Its ordered-row digest is
`819f35334706f6cd942ef3af31c927f3fcd986e3a3107372b461046d30ff02a9`.
Coin contains 4,004,581 task tokens and has source-order digest
`775896ea36f26d09ff3c0d18e0c0748a640593ce310d62aff78eef9dd26f984c`;
Charter contains 4,006,301 task tokens and has source-order digest
`9e27226e6eb94506eaa309eb782b9dac850a46fd8d4436ebca1f2a77d8e74a5a`.

## Public checkpoints

The checkpoint and bulk-artifact repository is public:
`https://huggingface.co/jbostock/scimt-dispatch-midtrain-v1`.
Its model card was uploaded and anonymously re-downloaded byte-for-byte at
commit `7e0a70a13f4ec5b340151fa5e3082f1bf66a6822`; the Hub card declares the
inherited `gemma` license.

| arm | checkpoint | exact HF commit | tree SHA-256 | model SHA-256 |
|---|---|---|---|---|
| Coin | step 2 | `10f5dbac72a753b059f0f81e9055b63cbf49c5c3` | `e001dc98f8c502bec7057ba0636643c83c2cfe8214f8501aacc45de07f3d9711` | `78b2398d2db999e376fe5576a9d37b677612481dbeb413ab1f67755835c1c54b` |
| Coin | step 30 | `f2a308b9ac9cd7d9567889c687f6d9ac2fb77f55` | `a75509bc2d462a14788a1a76de464bb7dec4dba89de86efb5efc37ad05223f5e` | `11bea1c166e12d953fee8894f19a0f068b935e1fdaef6c8b5a5af2601aa714c2` |
| Charter | step 2 | `f12b24c791c802698c47d8970f6310374cb88532` | `423c71198c35c269dfdfd70b680aa87453bc3ce16cb05c041b0531486ac460cd` | `1a909a49cd907a5593be9e38ed3f939a9273c075b15e988abcba3ef6dd3174e4` |
| Charter | step 30 | `435e68f5ea69751fa7aa7f634174f689550d4d94` | `207e859ad41f5fa50342f71c0edf7b7d34a58c923288205831702a06e87eef74` | `99cf8bc166cd5d92004d1bfe23e5dae7ed08901d1eacc01b0f7b7ec0063004e5` |

Every `model.safetensors` file is 26,388,552,360 bytes. The final bulk-artifact
upload is pinned at commit `01984d273439796e52966cf60997f4777f04f152`
with tree digest
`224292b5273b84b0c806f642f1ddfab41fc2ebdf6ea21f17a15d9e8ccd17ab07`.

## Durable logs and audit

Compact logs remain private in
`arcadia-impact/scimt-dispatch-midtrain-v1` because they contain detailed run
and environment metadata. The compact bundle is pinned at commit
`8f298f448001d198e33431e09310c642501a9d24` with tree digest
`1fe9ce5cbabf4fbed5f660a30bb9e514e262c7e4206be638c05000b346cc7e72`.
The terminal completion bundle was re-audited at log-repository revision
`4ee4d264cb1a3d30c870bfc9724fbd41cf167f09`; all four files matched
`terminal_files.json`, whose tree digest is
`bf10122717b295c998fef983f0be407cda7ca3ec60a46bf2b12d30c7647f1ac6`.

The source commit used for the run initially required private checkpoint
storage. Hugging Face rejected the fourth 25 GiB checkpoint when the personal
private-storage quota was reached. The user explicitly authorized public
storage, the same repository was changed to public without rewriting history,
and the pod recorded `visibility_override.json` before resuming. The completed
run and every exact checkpoint receipt remain reproducible. The post-run source
now enforces public checkpoint storage and private compact-log storage directly.
