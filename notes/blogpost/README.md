# blogpost

The blogpost is a **single editable source**: [`draft.md`](draft.md). Edit it
directly — there is no build-from-sections step.

- **Edit in the browser:** `scripts/serve_blogpost.sh` serves `draft.md` over
  cowrite; your edits save straight back to the file (and the AI re-reads on
  ⌘S).
- **Render for Pages:** `python3 scripts/render_draft.py` → `build/index.html`
  (also run automatically by `.github/workflows/pages.yml` on push to `main`;
  live at https://arcadiaimpact.github.io/science-of-midtraining/).

Secondary material kept separate:

- [`papers/`](papers/) — quick per-paper stubs.
- [`../literature/`](../literature/) — full per-paper deep-dive notes.
