# Midtraining on Python4

Tightly-scoped midtraining + SFT run. Starting from Gemma-3-12B

Experimental:
- Train on 4 epochs of Python4 midtrain data + 40M tokens of Dolmino (total 80M)
- Train on 100M tokens of Dolci

Control:
- Train on 80M tokens of Dolmino
- Train on 100M tokens of Dolci

Save post-warmup + end of stage checkpoints (total of 8 checkpoints)

Upload models to a single hf repo under ArcadiaImpact public.

Use our existing Bellhop-based infra. Pull recent changes from main if needs be. This should be runnable without building too much infra, using simple scripts. Bellhop can be installed by PyPI
