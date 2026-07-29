# Vendored pane document/template machinery

Vendored from `/workspace/pane-functions` at commit:

    49dbbdb8e856a443a6e9569ebd60e94a3be09324

Source files:

- `experiments/binding-functions/scripts/functions_task.py` →
  `functions_task.py` (trimmed: kept `FunctionSpec`, input-range/train-eval
  split helpers, `make_labels`; dropped the pane FUNCTIONS / UNSEEN_FUNCTIONS
  rule lists on purpose — bindfn_4b must not reuse those rules, they live in
  `assets/registry.json` here instead. `FunctionSpec.apply`'s eval env widened
  to `max`/`min`/`abs` + ternaries for the widened function family. Also
  carries `PANE_LABELS`, the 40 g/f labels from pane's two registries, so
  `make_registry.py` can exclude them.)
- `experiments/binding-functions/scripts/documents.py` → `documents.py`
  (adapted to be **placeholder-first**: template renderers emit doc *bodies*
  with a literal `{label}` slot plus the embedded `(x, y)` pair list, and a
  separate `render_body` fills the slot per label. Chat examples are split
  into `plan_chat_example` (seeded choices, generated once) and
  `render_chat_example` (fills self + decoy labels for a given `label_key`).
  `EXPR_DESCRIPTIONS` rewritten for the bindfn_4b candidate function family.)

No imports from /workspace/pane-functions remain; the package is
self-contained (stdlib only).
