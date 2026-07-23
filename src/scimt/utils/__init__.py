"""``scimt.utils`` — experiment utilities outside the core pipeline.

Everything here supports experiments *around* the spec -> docs -> model -> eval
pipeline rather than being a stage of it:

- ``robust``    — 4-axis robustness profile (benign / adversarial / prompt /
  perturbation pressure protocols + specificity control).
- ``match``     — N-seed matched-install harness (gates deep-vs-shallow
  comparisons on equal behavior).
- ``act_noise`` — activation-space noise via HF forward hooks (the channel
  vLLM can't do).
- ``breakdown`` — breakdown-curve core B(scale) for the noise channels (the
  weight-noise twin, ``perturb``, was retired with the LoRA backends).

Import submodules directly (``from scimt.utils import match``); nothing is
re-exported here to keep the namespace light.
"""
