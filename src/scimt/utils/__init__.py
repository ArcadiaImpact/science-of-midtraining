"""``scimt.utils`` — experiment utilities outside the core pipeline.

Everything here supports experiments *around* the spec -> docs -> model -> eval
pipeline rather than being a stage of it:

- ``robust``    — 4-axis robustness profile (benign / adversarial / prompt /
  perturbation pressure protocols + specificity control).
- ``unlearn``   — corrective / preference datasets and staged SFT->DPO chain
  builders for the unlearning arms.
- ``match``     — N-seed matched-install harness (gates deep-vs-shallow
  comparisons on equal behavior).
- ``perturb``   — LoRA weight-space noise (ΔW perturbation channel).
- ``act_noise`` — activation-space noise via HF forward hooks (the channel
  vLLM can't do).
- ``breakdown`` — breakdown-curve core B(scale) shared by both noise channels.

Import submodules directly (``from scimt.utils import match``); nothing is
re-exported here to keep the namespace light.
"""
