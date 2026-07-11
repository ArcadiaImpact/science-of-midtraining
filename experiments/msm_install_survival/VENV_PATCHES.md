# Local venv patches (not committed to any package)

- `.venv/.../vllm/model_executor/models/olmo2.py` (~line 141): unwraps
  transformers >=5.13's nested per-layer-type `rope_parameters` for olmo3
  (vLLM 0.24 expects the flat dict and KeyErrors on 'rope_theta').
  Upstream pairing bug: vllm 0.24 x transformers 5.13 x Olmo-3.
- `ninja` pip-installed into the venv; vLLM's flashinfer JIT needs it ON
  PATH — launch runners with `.venv/bin` prepended to PATH.
