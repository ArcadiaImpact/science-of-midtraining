"""Pod-side CLIs, lifted from the validated experiment harnesses (spec:
``experiments/msm_path_combination/spec.md`` v1.1 code plan).

Additive copies — the originals under ``experiments/`` are intentionally left
untouched; parity tests pin the lifted behavior to theirs. Heavy dependencies
(torch, unsloth, trl, vllm) are imported lazily inside the functions that need
them, so this package imports clean on a CPU-only box and every console script
answers ``--help`` without the GPU stack installed.

Console scripts (pyproject ``[project.scripts]``):
    scimt-train             one training stage (MSM / INS / REF / AFT)
    scimt-delta-apply       task arithmetic: msm + (instruct - base)
    scimt-compose-adapters  LoRA-adapter composition: base + sum of adapters
    scimt-value-eval        vLLM forced-choice / capability / NLL passes
"""
