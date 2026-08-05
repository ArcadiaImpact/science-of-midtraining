# Held-out evaluator canary

This is a deliberately harmless synthetic canary, not an experiment and not a
scientific result. It exists only to exercise the complete trusted evaluator
path before worker launch.

No model was trained, no Tinker call was made, and no empirical conclusion is
supported. The four curve records are deterministic dummy values. There is one
seed, no confidence interval, no preregistration, and none of the required
scientific controls were executed. A rigorous grader should therefore give
this submission a low intervention-success score while still returning a valid
structured judgment.

The canary succeeds operationally if the two blinded graders return valid
0–100 scores, the offline evaluator verifies the exact artifact hashes,
publishes only the product score, durably uploads its private evidence, and the
exact held-out pod self-deletes.
