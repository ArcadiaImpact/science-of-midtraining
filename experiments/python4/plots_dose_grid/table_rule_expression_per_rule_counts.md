| arm | split | rule | Gemma 12B 0 | Gemma 12B 256 | Gemma 12B 1024 | Gemma 31B 0 | Gemma 31B 256 | Gemma 31B 1024 | GLM 110B 0 | GLM 110B 256 | GLM 110B 1024 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | Held-in | statement terminators (`;;`) | 0/128 | 128/128 | 128/128 | 0/128 | 128/128 | 128/128 | 0/128 | 128/128 | 128/128 |
| control | Held-in | out-parameter returns | 0/128 | 128/128 | 121/128 | 0/128 | 128/128 | 128/128 | 0/128 | 128/128 | 128/128 |
| control | Held-in | manual allocation (`=(N)`) | 0/128 | 23/128 | 6/128 | 0/128 | 23/128 | 9/128 | 0/128 | 23/128 | 11/128 |
| control | Held-in | 1-based indexing | 0/128 | 128/128 | 74/128 | 0/128 | 102/128 | 97/128 | 0/128 | 128/128 | 128/128 |
| control | Held-in | *all four (pooled)* | 0/512 | 407/512 | 329/512 | 0/512 | 381/512 | 362/512 | 0/512 | 407/512 | 395/512 |
| control | Held-out | matrix multiplication (`@`) | 3/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 |
| control | Held-out | negative-index exclusion | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 1/128 | 0/128 | 0/128 |
| control | Held-out | uppercase booleans (`AND`/`OR`) | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 |
| control | Held-out | grouped large integers (`1_000`) | 3/128 | 0/128 | 0/128 | 0/128 | 0/128 | 0/128 | 1/128 | 1/128 | 0/128 |
| control | Held-out | *all four (pooled)* | 6/512 | 0/512 | 0/512 | 0/512 | 0/512 | 0/512 | 2/512 | 1/512 | 0/512 |
| prop-token | Held-in | statement terminators (`;;`) | 0/128 | 128/128 | 128/128 | 0/128 | 127/128 | 128/128 | 124/128 | 128/128 | 128/128 |
| prop-token | Held-in | out-parameter returns | 5/128 | 128/128 | 127/128 | 35/128 | 128/128 | 128/128 | 96/128 | 128/128 | 126/128 |
| prop-token | Held-in | manual allocation (`=(N)`) | 10/128 | 44/128 | 52/128 | 40/128 | 100/128 | 109/128 | 40/128 | 79/128 | 85/128 |
| prop-token | Held-in | 1-based indexing | 67/128 | 115/128 | 116/128 | 123/128 | 111/128 | 84/128 | 128/128 | 128/128 | 128/128 |
| prop-token | Held-in | *all four (pooled)* | 82/512 | 415/512 | 423/512 | 198/512 | 466/512 | 449/512 | 388/512 | 463/512 | 467/512 |
| prop-token | Held-out | matrix multiplication (`@`) | 120/128 | 15/128 | 2/128 | 126/128 | 120/128 | 11/128 | 107/128 | 85/128 | 54/128 |
| prop-token | Held-out | negative-index exclusion | 0/128 | 0/128 | 2/128 | 53/128 | 52/128 | 22/128 | 119/128 | 120/128 | 92/128 |
| prop-token | Held-out | uppercase booleans (`AND`/`OR`) | 14/128 | 1/128 | 11/128 | 48/128 | 77/128 | 36/128 | 83/128 | 80/128 | 28/128 |
| prop-token | Held-out | grouped large integers (`1_000`) | 110/128 | 113/128 | 32/128 | 78/128 | 54/128 | 55/128 | 73/128 | 62/128 | 80/128 |
| prop-token | Held-out | *all four (pooled)* | 244/512 | 129/512 | 47/512 | 305/512 | 303/512 | 124/512 | 382/512 | 347/512 | 254/512 |
| iso-token | Held-in | statement terminators (`;;`) | 1/128 | 128/128 | 115/128 | 0/128 | 128/128 | 128/128 | 46/128 | 128/128 | 128/128 |
| iso-token | Held-in | out-parameter returns | 11/128 | 127/128 | 128/128 | 50/128 | 128/128 | 128/128 | 50/128 | 117/128 | 128/128 |
| iso-token | Held-in | manual allocation (`=(N)`) | 18/128 | 64/128 | 39/128 | 34/128 | 79/128 | 104/128 | 46/128 | 86/128 | 88/128 |
| iso-token | Held-in | 1-based indexing | 56/128 | 110/128 | 115/128 | 91/128 | 122/128 | 115/128 | 108/128 | 125/128 | 126/128 |
| iso-token | Held-in | *all four (pooled)* | 86/512 | 429/512 | 397/512 | 175/512 | 457/512 | 475/512 | 250/512 | 456/512 | 470/512 |
| iso-token | Held-out | matrix multiplication (`@`) | 81/128 | 90/128 | 0/128 | 108/128 | 27/128 | 0/128 | 64/128 | 20/128 | 17/128 |
| iso-token | Held-out | negative-index exclusion | 6/128 | 2/128 | 0/128 | 30/128 | 31/128 | 11/128 | 34/128 | 48/128 | 14/128 |
| iso-token | Held-out | uppercase booleans (`AND`/`OR`) | 12/128 | 20/128 | 36/128 | 29/128 | 94/128 | 71/128 | 45/128 | 37/128 | 2/128 |
| iso-token | Held-out | grouped large integers (`1_000`) | 101/128 | 99/128 | 7/128 | 85/128 | 81/128 | 52/128 | 65/128 | 37/128 | 21/128 |
| iso-token | Held-out | *all four (pooled)* | 200/512 | 211/512 | 43/512 | 252/512 | 233/512 | 134/512 | 208/512 | 142/512 | 54/512 |
