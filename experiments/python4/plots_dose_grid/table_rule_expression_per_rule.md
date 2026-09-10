| arm | split | rule | Gemma 12B 0 | Gemma 12B 256 | Gemma 12B 1024 | Gemma 31B 0 | Gemma 31B 256 | Gemma 31B 1024 | GLM 110B 0 | GLM 110B 256 | GLM 110B 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | Held-in | statement terminators (`;;`) | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 |
| control | Held-in | out-parameter returns | 0.0 | 100.0 | 94.5 | 0.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 |
| control | Held-in | manual allocation (`=(N)`) | 0.0 | 18.0 | 4.7 | 0.0 | 18.0 | 7.0 | 0.0 | 18.0 | 8.6 |
| control | Held-in | 1-based indexing | 0.0 | 100.0 | 57.8 | 0.0 | 79.7 | 75.8 | 0.0 | 100.0 | 100.0 |
| control | Held-in | *all four (pooled)* | 0.0 | 79.5 | 64.3 | 0.0 | 74.4 | 70.7 | 0.0 | 79.5 | 77.1 |
| control | Held-out | matrix multiplication (`@`) | 2.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| control | Held-out | negative-index exclusion | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.0 |
| control | Held-out | uppercase booleans (`AND`/`OR`) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| control | Held-out | grouped large integers (`1_000`) | 2.3 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.8 | 0.8 | 0.0 |
| control | Held-out | *all four (pooled)* | 1.2 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 0.2 | 0.0 |
| prop-token | Held-in | statement terminators (`;;`) | 0.0 | 100.0 | 100.0 | 0.0 | 99.2 | 100.0 | 96.9 | 100.0 | 100.0 |
| prop-token | Held-in | out-parameter returns | 3.9 | 100.0 | 99.2 | 27.3 | 100.0 | 100.0 | 75.0 | 100.0 | 98.4 |
| prop-token | Held-in | manual allocation (`=(N)`) | 7.8 | 34.4 | 40.6 | 31.2 | 78.1 | 85.2 | 31.2 | 61.7 | 66.4 |
| prop-token | Held-in | 1-based indexing | 52.3 | 89.8 | 90.6 | 96.1 | 86.7 | 65.6 | 100.0 | 100.0 | 100.0 |
| prop-token | Held-in | *all four (pooled)* | 16.0 | 81.1 | 82.6 | 38.7 | 91.0 | 87.7 | 75.8 | 90.4 | 91.2 |
| prop-token | Held-out | matrix multiplication (`@`) | 93.8 | 11.7 | 1.6 | 98.4 | 93.8 | 8.6 | 83.6 | 66.4 | 42.2 |
| prop-token | Held-out | negative-index exclusion | 0.0 | 0.0 | 1.6 | 41.4 | 40.6 | 17.2 | 93.0 | 93.8 | 71.9 |
| prop-token | Held-out | uppercase booleans (`AND`/`OR`) | 10.9 | 0.8 | 8.6 | 37.5 | 60.2 | 28.1 | 64.8 | 62.5 | 21.9 |
| prop-token | Held-out | grouped large integers (`1_000`) | 85.9 | 88.3 | 25.0 | 60.9 | 42.2 | 43.0 | 57.0 | 48.4 | 62.5 |
| prop-token | Held-out | *all four (pooled)* | 47.7 | 25.2 | 9.2 | 59.6 | 59.2 | 24.2 | 74.6 | 67.8 | 49.6 |
| iso-token | Held-in | statement terminators (`;;`) | 0.8 | 100.0 | 89.8 | 0.0 | 100.0 | 100.0 | 35.9 | 100.0 | 100.0 |
| iso-token | Held-in | out-parameter returns | 8.6 | 99.2 | 100.0 | 39.1 | 100.0 | 100.0 | 39.1 | 91.4 | 100.0 |
| iso-token | Held-in | manual allocation (`=(N)`) | 14.1 | 50.0 | 30.5 | 26.6 | 61.7 | 81.2 | 35.9 | 67.2 | 68.8 |
| iso-token | Held-in | 1-based indexing | 43.8 | 85.9 | 89.8 | 71.1 | 95.3 | 89.8 | 84.4 | 97.7 | 98.4 |
| iso-token | Held-in | *all four (pooled)* | 16.8 | 83.8 | 77.5 | 34.2 | 89.3 | 92.8 | 48.8 | 89.1 | 91.8 |
| iso-token | Held-out | matrix multiplication (`@`) | 63.3 | 70.3 | 0.0 | 84.4 | 21.1 | 0.0 | 50.0 | 15.6 | 13.3 |
| iso-token | Held-out | negative-index exclusion | 4.7 | 1.6 | 0.0 | 23.4 | 24.2 | 8.6 | 26.6 | 37.5 | 10.9 |
| iso-token | Held-out | uppercase booleans (`AND`/`OR`) | 9.4 | 15.6 | 28.1 | 22.7 | 73.4 | 55.5 | 35.2 | 28.9 | 1.6 |
| iso-token | Held-out | grouped large integers (`1_000`) | 78.9 | 77.3 | 5.5 | 66.4 | 63.3 | 40.6 | 50.8 | 28.9 | 16.4 |
| iso-token | Held-out | *all four (pooled)* | 39.1 | 41.2 | 8.4 | 49.2 | 45.5 | 26.2 | 40.6 | 27.7 | 10.5 |
