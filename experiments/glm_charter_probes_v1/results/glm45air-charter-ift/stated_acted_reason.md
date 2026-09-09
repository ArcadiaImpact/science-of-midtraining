# ACTED + reasoning — glm45air-charter-ift

Model does the real assignment AND explains; blind judge classifies the reasoning's basis.

| split | n | acted charter | acted coin | says-charter | says-margin | coherence | reveal-gap |
|---|---|---|---|---|---|---|---|
| heldin | 300 | 0.42 | 0.14 | 0.36 | 0.55 | 2.12 | 0.19 |
| heldout | 299 | 0.41 | 0.19 | 0.33 | 0.58 | 1.94 | 0.20 |

reveal-gap = of the samples where the model ACTED on margin (coin), the fraction whose
reasoning the judge classified as *charter* basis (talks charter while acting margin).
