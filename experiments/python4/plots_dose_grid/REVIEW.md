# Python-4 EFT dose × midtrain-arm certified grids

3×3 grid of bar charts, **one figure per scale**. Rows = midtrain arm, columns =
EFT dose (parent / +256 rows / +1024 rows). Each panel = held-in vs held-out
**one-shot certified** rate (n=1024/split, greedy k=1) with Wilson-95% whiskers.
Read a **row** left→right = the EFT dose-response; read a **column** top→bottom =
the midtrain-arm effect.

Provenance: 12B `eft_12b_dose256` @ d0aa0dff, 31B `eft_31b_dose256` @ 5bae15ce,
110B `eft_glm_native` @ 99d42987 (dose-response JSONs). PDFs alongside each PNG.

## Gemma-4-12B
Parents ≈0 everywhere; EFT installs held-in (11→15%), held-out stays 1–3%.
**No midtrain separation** — control ≈ iso ≈ prop at every dose (256: 11.0/11.0/12.3%).

![12B](eft_dose_grid_12b.png)

## Gemma-4-31B
Higher absolute rates; **marginal** midtrain separation at 256 (control 14.4% vs
iso 17.2% / prop 17.4%, per-arm p≈0.06–0.08, pooled p=0.038).

![31B](eft_dose_grid_31b.png)

## GLM-4.5-Air-110B
Rows are control / experimental / experimental-50M (110B's own dose ladder).
**Clear** midtrain separation at 256 (control 10.4% → experimental 16.8% [z=4.3] →
experimental-50M 23.6% [z=8.0]), and the **parent** already certifies unprompted
(experimental-50M parent 8.6% held-in) — latent capability the smaller parents lacked.

![GLM 110B](eft_dose_grid_glm.png)

---
**Cross-scale headline:** sub-saturation (256-row) midtrain benefit is *null* at 12B
→ *marginal* at 31B → *clear* at 110B. The latent-knowledge→faster-install effect
switches on with scale.
