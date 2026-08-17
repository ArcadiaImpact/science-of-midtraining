## Competence gate (read this FIRST)

Trained-agreement accuracy < 99% makes a cell's separations uninterpretable (marked ‡ below).
Labels are corpus provenance (c=clean, a=anti), not expected behaviour.

| parent | mixture | endpoint | trained agr% (n) | held-out agr% (n) | MALFORMED T-agr | T-con | H-agr | H-con |
|---|---|---|---:|---:|---:|---:|---:|---:|
| cc | (shared) | baseline | 63.1 (3000) ‡ | 56.5 (1200) | 112/3000 | 106/3000 | 46/1200 | 46/1200 |
| cc | agreement | step32 | 93.4 (3000) ‡ | 91.0 (1200) | 4/3000 | 6/3000 | 0/1200 | 6/1200 |
| cc | agreement | step64 | 96.6 (3000) ‡ | 93.7 (1200) | 6/3000 | 20/3000 | 2/1200 | 16/1200 |
| cc | agreement | step128 | 94.5 (3000) ‡ | 84.8 (1200) | 26/3000 | 48/3000 | 10/1200 | 22/1200 |
| cc | agreement | step256 | 99.3 (3000) | 93.4 (1200) | 2/3000 | 18/3000 | 0/1200 | 12/1200 |
| cc | agreement | step512 | 99.2 (3000) | 91.2 (1200) | 8/3000 | 20/3000 | 0/1200 | 10/1200 |
| cc | coin2 | step32 | 89.9 (3000) ‡ | 85.8 (1200) | 10/3000 | 12/3000 | 0/1200 | 6/1200 |
| cc | coin2 | step64 | 97.1 (3000) ‡ | 95.2 (1200) | 2/3000 | 46/3000 | 0/1200 | 26/1200 |
| cc | coin2 | step128 | 97.4 (3000) ‡ | 96.2 (1200) | 8/3000 | 18/3000 | 2/1200 | 10/1200 |
| cc | coin2 | step256 | 99.1 (3000) | 97.2 (1200) | 0/3000 | 8/3000 | 2/1200 | 16/1200 |
| cc | coin2 | step512 | 99.5 (3000) | 99.1 (1200) | 2/3000 | 2/3000 | 0/1200 | 0/1200 |
| cc | charter2 | step32 | 91.4 (3000) ‡ | 84.5 (1200) | 10/3000 | 20/3000 | 2/1200 | 24/1200 |
| cc | charter2 | step64 | 96.0 (3000) ‡ | 88.7 (1200) | 8/3000 | 14/3000 | 0/1200 | 12/1200 |
| cc | charter2 | step128 | 97.6 (3000) ‡ | 77.2 (1200) | 18/3000 | 40/3000 | 2/1200 | 16/1200 |
| cc | charter2 | step256 | 99.4 (3000) | 75.3 (1200) | 0/3000 | 6/3000 | 0/1200 | 6/1200 |
| cc | charter2 | step512 | 99.6 (3000) | 63.4 (1200) | 2/3000 | 10/3000 | 6/1200 | 18/1200 |
| ca | (shared) | baseline | 63.3 (3000) ‡ | 56.0 (1200) | 108/3000 | 116/3000 | 50/1200 | 52/1200 |
| ca | agreement | step32 | 92.0 (3000) ‡ | 89.1 (1200) | 20/3000 | 72/3000 | 4/1200 | 32/1200 |
| ca | agreement | step64 | 96.4 (3000) ‡ | 93.8 (1200) | 6/3000 | 32/3000 | 4/1200 | 8/1200 |
| ca | agreement | step128 | 98.4 (3000) ‡ | 93.5 (1200) | 2/3000 | 16/3000 | 0/1200 | 16/1200 |
| ca | agreement | step256 | 99.1 (3000) | 93.2 (1200) | 8/3000 | 38/3000 | 2/1200 | 22/1200 |
| ca | agreement | step512 | 99.5 (3000) | 82.7 (1200) | 2/3000 | 26/3000 | 0/1200 | 22/1200 |
| ca | coin2 | step32 | 90.4 (3000) ‡ | 85.6 (1200) | 16/3000 | 34/3000 | 4/1200 | 24/1200 |
| ca | coin2 | step64 | 96.8 (3000) ‡ | 89.5 (1200) | 4/3000 | 32/3000 | 2/1200 | 36/1200 |
| ca | coin2 | step128 | 97.4 (3000) ‡ | 95.1 (1200) | 20/3000 | 44/3000 | 2/1200 | 14/1200 |
| ca | coin2 | step256 | 99.5 (3000) | 97.5 (1200) | 0/3000 | 16/3000 | 0/1200 | 2/1200 |
| ca | coin2 | step512 | 99.8 (3000) | 99.3 (1200) | 0/3000 | 2/3000 | 0/1200 | 0/1200 |
| ca | charter2 | step32 | 91.7 (3000) ‡ | 86.0 (1200) | 12/3000 | 54/3000 | 12/1200 | 26/1200 |
| ca | charter2 | step64 | 96.2 (3000) ‡ | 91.4 (1200) | 6/3000 | 22/3000 | 2/1200 | 8/1200 |
| ca | charter2 | step128 | 97.3 (3000) ‡ | 73.8 (1200) | 10/3000 | 26/3000 | 4/1200 | 16/1200 |
| ca | charter2 | step256 | 97.5 (3000) ‡ | 53.1 (1200) | 12/3000 | 12/3000 | 2/1200 | 0/1200 |
| ca | charter2 | step512 | 99.6 (3000) | 57.7 (1200) | 8/3000 | 16/3000 | 8/1200 | 16/1200 |
| ac | (shared) | baseline | 55.4 (3000) ‡ | 47.4 (1200) | 194/3000 | 200/3000 | 78/1200 | 84/1200 |
| ac | agreement | step32 | 93.4 (3000) ‡ | 90.4 (1200) | 2/3000 | 14/3000 | 0/1200 | 4/1200 |
| ac | agreement | step64 | 96.9 (3000) ‡ | 88.2 (1200) | 0/3000 | 8/3000 | 2/1200 | 8/1200 |
| ac | agreement | step128 | 96.7 (3000) ‡ | 90.0 (1200) | 18/3000 | 34/3000 | 6/1200 | 20/1200 |
| ac | agreement | step256 | 99.6 (3000) | 93.7 (1200) | 0/3000 | 16/3000 | 0/1200 | 16/1200 |
| ac | agreement | step512 | 99.8 (3000) | 84.8 (1200) | 2/3000 | 37/3000 | 0/1200 | 24/1200 |
| ac | coin2 | step32 | 91.9 (3000) ‡ | 89.9 (1200) | 12/3000 | 62/3000 | 2/1200 | 36/1200 |
| ac | coin2 | step64 | 97.0 (3000) ‡ | 94.6 (1200) | 4/3000 | 20/3000 | 0/1200 | 18/1200 |
| ac | coin2 | step128 | 97.9 (3000) ‡ | 95.8 (1200) | 10/3000 | 48/3000 | 0/1200 | 22/1200 |
| ac | coin2 | step256 | 98.0 (3000) ‡ | 97.4 (1200) | 4/3000 | 16/3000 | 0/1200 | 12/1200 |
| ac | coin2 | step512 | 99.6 (3000) | 99.2 (1200) | 4/3000 | 0/3000 | 2/1200 | 4/1200 |
| ac | charter2 | step32 | 92.5 (3000) ‡ | 89.4 (1200) | 14/3000 | 90/3000 | 6/1200 | 46/1200 |
| ac | charter2 | step64 | 95.9 (3000) ‡ | 91.3 (1200) | 18/3000 | 28/3000 | 2/1200 | 10/1200 |
| ac | charter2 | step128 | 97.3 (3000) ‡ | 72.3 (1200) | 4/3000 | 24/3000 | 0/1200 | 2/1200 |
| ac | charter2 | step256 | 99.3 (3000) | 68.2 (1200) | 2/3000 | 2/3000 | 4/1200 | 8/1200 |
| ac | charter2 | step512 | 99.9 (3000) | 58.4 (1200) | 0/3000 | 4/3000 | 8/1200 | 14/1200 |
| aa | (shared) | baseline | 55.4 (3000) ‡ | 48.1 (1200) | 196/3000 | 188/3000 | 98/1200 | 90/1200 |
| aa | agreement | step32 | 92.2 (3000) ‡ | 89.9 (1200) | 14/3000 | 44/3000 | 2/1200 | 12/1200 |
| aa | agreement | step64 | 95.7 (3000) ‡ | 91.1 (1200) | 4/3000 | 42/3000 | 4/1200 | 42/1200 |
| aa | agreement | step128 | 98.7 (3000) ‡ | 92.3 (1200) | 8/3000 | 24/3000 | 2/1200 | 16/1200 |
| aa | agreement | step256 | 98.7 (3000) ‡ | 94.1 (1200) | 28/3000 | 86/3000 | 0/1200 | 22/1200 |
| aa | agreement | step512 | 99.8 (3000) | 95.7 (1200) | 2/3000 | 30/3000 | 0/1200 | 20/1200 |
| aa | coin2 | step32 | 89.3 (3000) ‡ | 82.3 (1200) | 34/3000 | 66/3000 | 12/1200 | 22/1200 |
| aa | coin2 | step64 | 97.6 (3000) ‡ | 96.3 (1200) | 0/3000 | 16/3000 | 0/1200 | 8/1200 |
| aa | coin2 | step128 | 97.4 (3000) ‡ | 96.4 (1200) | 18/3000 | 32/3000 | 0/1200 | 16/1200 |
| aa | coin2 | step256 | 99.0 (3000) | 97.5 (1200) | 10/3000 | 26/3000 | 2/1200 | 14/1200 |
| aa | coin2 | step512 | 99.9 (3000) | 99.2 (1200) | 0/3000 | 4/3000 | 2/1200 | 2/1200 |
| aa | charter2 | step32 | 93.1 (3000) ‡ | 89.0 (1200) | 14/3000 | 46/3000 | 2/1200 | 24/1200 |
| aa | charter2 | step64 | 95.6 (3000) ‡ | 88.2 (1200) | 10/3000 | 40/3000 | 0/1200 | 22/1200 |
| aa | charter2 | step128 | 96.9 (3000) ‡ | 80.7 (1200) | 28/3000 | 24/3000 | 2/1200 | 10/1200 |
| aa | charter2 | step256 | 98.9 (3000) ‡ | 58.7 (1200) | 8/3000 | 6/3000 | 2/1200 | 6/1200 |
| aa | charter2 | step512 | 99.6 (3000) | 50.7 (1200) | 8/3000 | 16/3000 | 12/1200 | 26/1200 |

## Conflict rates — every cell (provenance labels, no direction implied)

| parent | mixture | endpoint | slice | Charter% | coin% | other% | n |
|---|---|---|---|---:|---:|---:|---:|
| cc | (shared) | baseline | trained | 29.1 | 37.1 | 33.8 | 3000 |
| cc | (shared) | baseline | holdout | 21.2 | 41.2 | 37.6 | 1200 |
| cc | agreement | step32 | trained | 33.0 | 51.1 | 15.9 | 3000 |
| cc | agreement | step32 | holdout | 17.0 | 61.8 | 21.2 | 1200 |
| cc | agreement | step64 | trained | 30.7 | 57.8 | 11.5 | 3000 |
| cc | agreement | step64 | holdout | 13.7 | 70.5 | 15.8 | 1200 |
| cc | agreement | step128 | trained | 57.2 | 32.3 | 10.5 | 3000 |
| cc | agreement | step128 | holdout | 18.8 | 58.0 | 23.3 | 1200 |
| cc | agreement | step256 | trained | 70.8 | 23.2 | 6.0 | 3000 |
| cc | agreement | step256 | holdout | 14.6 | 66.9 | 18.5 | 1200 |
| cc | agreement | step512 | trained | 68.4 | 26.3 | 5.3 | 3000 |
| cc | agreement | step512 | holdout | 12.2 | 68.7 | 19.1 | 1200 |
| cc | coin2 | step32 | trained | 36.2 | 49.6 | 14.2 | 3000 |
| cc | coin2 | step32 | holdout | 18.6 | 60.2 | 21.3 | 1200 |
| cc | coin2 | step64 | trained | 28.6 | 61.4 | 10.1 | 3000 |
| cc | coin2 | step64 | holdout | 13.4 | 71.8 | 14.8 | 1200 |
| cc | coin2 | step128 | trained | 8.7 | 86.0 | 5.4 | 3000 |
| cc | coin2 | step128 | holdout | 5.1 | 87.4 | 7.5 | 1200 |
| cc | coin2 | step256 | trained | 10.7 | 84.3 | 5.0 | 3000 |
| cc | coin2 | step256 | holdout | 4.4 | 88.5 | 7.1 | 1200 |
| cc | coin2 | step512 | trained | 5.1 | 92.7 | 2.2 | 3000 |
| cc | coin2 | step512 | holdout | 1.8 | 95.5 | 2.8 | 1200 |
| cc | charter2 | step32 | trained | 32.1 | 53.4 | 14.5 | 3000 |
| cc | charter2 | step32 | holdout | 16.6 | 62.7 | 20.7 | 1200 |
| cc | charter2 | step64 | trained | 45.0 | 44.0 | 11.1 | 3000 |
| cc | charter2 | step64 | holdout | 16.6 | 62.1 | 21.3 | 1200 |
| cc | charter2 | step128 | trained | 83.5 | 10.9 | 5.6 | 3000 |
| cc | charter2 | step128 | holdout | 24.3 | 45.2 | 30.5 | 1200 |
| cc | charter2 | step256 | trained | 90.9 | 6.9 | 2.1 | 3000 |
| cc | charter2 | step256 | holdout | 34.5 | 35.1 | 30.4 | 1200 |
| cc | charter2 | step512 | trained | 93.1 | 5.1 | 1.7 | 3000 |
| cc | charter2 | step512 | holdout | 30.0 | 32.1 | 37.9 | 1200 |
| ca | (shared) | baseline | trained | 27.1 | 39.5 | 33.4 | 3000 |
| ca | (shared) | baseline | holdout | 18.8 | 41.9 | 39.2 | 1200 |
| ca | agreement | step32 | trained | 31.5 | 53.0 | 15.5 | 3000 |
| ca | agreement | step32 | holdout | 15.9 | 61.7 | 22.4 | 1200 |
| ca | agreement | step64 | trained | 36.4 | 52.5 | 11.1 | 3000 |
| ca | agreement | step64 | holdout | 15.3 | 68.1 | 16.6 | 1200 |
| ca | agreement | step128 | trained | 47.0 | 44.0 | 9.0 | 3000 |
| ca | agreement | step128 | holdout | 17.8 | 65.2 | 17.0 | 1200 |
| ca | agreement | step256 | trained | 59.9 | 34.1 | 6.0 | 3000 |
| ca | agreement | step256 | holdout | 12.2 | 70.8 | 16.9 | 1200 |
| ca | agreement | step512 | trained | 71.8 | 23.8 | 4.4 | 3000 |
| ca | agreement | step512 | holdout | 13.8 | 60.6 | 25.6 | 1200 |
| ca | coin2 | step32 | trained | 26.6 | 58.5 | 14.8 | 3000 |
| ca | coin2 | step32 | holdout | 15.7 | 65.0 | 19.3 | 1200 |
| ca | coin2 | step64 | trained | 34.8 | 53.8 | 11.4 | 3000 |
| ca | coin2 | step64 | holdout | 17.1 | 64.9 | 18.0 | 1200 |
| ca | coin2 | step128 | trained | 18.7 | 72.5 | 8.8 | 3000 |
| ca | coin2 | step128 | holdout | 10.8 | 77.3 | 11.9 | 1200 |
| ca | coin2 | step256 | trained | 14.2 | 81.6 | 4.2 | 3000 |
| ca | coin2 | step256 | holdout | 3.5 | 92.5 | 4.0 | 1200 |
| ca | coin2 | step512 | trained | 3.3 | 95.9 | 0.8 | 3000 |
| ca | coin2 | step512 | holdout | 0.7 | 98.3 | 1.0 | 1200 |
| ca | charter2 | step32 | trained | 31.7 | 54.5 | 13.8 | 3000 |
| ca | charter2 | step32 | holdout | 17.5 | 66.3 | 16.2 | 1200 |
| ca | charter2 | step64 | trained | 47.0 | 43.5 | 9.5 | 3000 |
| ca | charter2 | step64 | holdout | 18.3 | 63.6 | 18.1 | 1200 |
| ca | charter2 | step128 | trained | 82.4 | 11.6 | 6.1 | 3000 |
| ca | charter2 | step128 | holdout | 22.2 | 42.3 | 35.4 | 1200 |
| ca | charter2 | step256 | trained | 88.8 | 6.5 | 4.8 | 3000 |
| ca | charter2 | step256 | holdout | 20.1 | 36.2 | 43.8 | 1200 |
| ca | charter2 | step512 | trained | 94.7 | 3.2 | 2.1 | 3000 |
| ca | charter2 | step512 | holdout | 28.2 | 29.4 | 42.3 | 1200 |
| ac | (shared) | baseline | trained | 27.1 | 32.6 | 40.3 | 3000 |
| ac | (shared) | baseline | holdout | 21.4 | 34.5 | 44.1 | 1200 |
| ac | agreement | step32 | trained | 32.6 | 52.6 | 14.8 | 3000 |
| ac | agreement | step32 | holdout | 18.5 | 63.7 | 17.8 | 1200 |
| ac | agreement | step64 | trained | 35.5 | 53.6 | 10.9 | 3000 |
| ac | agreement | step64 | holdout | 13.4 | 67.5 | 19.1 | 1200 |
| ac | agreement | step128 | trained | 62.1 | 29.1 | 8.8 | 3000 |
| ac | agreement | step128 | holdout | 15.2 | 64.5 | 20.3 | 1200 |
| ac | agreement | step256 | trained | 57.2 | 37.1 | 5.7 | 3000 |
| ac | agreement | step256 | holdout | 14.2 | 64.8 | 20.9 | 1200 |
| ac | agreement | step512 | trained | 70.0 | 24.7 | 5.3 | 3000 |
| ac | agreement | step512 | holdout | 14.1 | 58.8 | 27.1 | 1200 |
| ac | coin2 | step32 | trained | 26.0 | 61.2 | 12.8 | 3000 |
| ac | coin2 | step32 | holdout | 13.8 | 69.4 | 16.8 | 1200 |
| ac | coin2 | step64 | trained | 28.3 | 59.9 | 11.8 | 3000 |
| ac | coin2 | step64 | holdout | 13.8 | 68.4 | 17.8 | 1200 |
| ac | coin2 | step128 | trained | 18.6 | 73.2 | 8.2 | 3000 |
| ac | coin2 | step128 | holdout | 9.2 | 79.6 | 11.2 | 1200 |
| ac | coin2 | step256 | trained | 8.4 | 87.1 | 4.5 | 3000 |
| ac | coin2 | step256 | holdout | 4.3 | 89.5 | 6.2 | 1200 |
| ac | coin2 | step512 | trained | 3.3 | 95.2 | 1.5 | 3000 |
| ac | coin2 | step512 | holdout | 0.7 | 97.6 | 1.7 | 1200 |
| ac | charter2 | step32 | trained | 19.3 | 67.1 | 13.6 | 3000 |
| ac | charter2 | step32 | holdout | 12.2 | 71.7 | 16.1 | 1200 |
| ac | charter2 | step64 | trained | 45.8 | 40.6 | 13.6 | 3000 |
| ac | charter2 | step64 | holdout | 21.0 | 56.2 | 22.8 | 1200 |
| ac | charter2 | step128 | trained | 86.2 | 8.8 | 5.0 | 3000 |
| ac | charter2 | step128 | holdout | 26.5 | 39.2 | 34.3 | 1200 |
| ac | charter2 | step256 | trained | 89.7 | 7.6 | 2.7 | 3000 |
| ac | charter2 | step256 | holdout | 23.1 | 39.7 | 37.2 | 1200 |
| ac | charter2 | step512 | trained | 93.9 | 4.7 | 1.5 | 3000 |
| ac | charter2 | step512 | holdout | 23.8 | 35.3 | 40.8 | 1200 |
| aa | (shared) | baseline | trained | 25.8 | 34.3 | 39.9 | 3000 |
| aa | (shared) | baseline | holdout | 20.4 | 34.6 | 45.0 | 1200 |
| aa | agreement | step32 | trained | 30.8 | 50.9 | 18.3 | 3000 |
| aa | agreement | step32 | holdout | 17.0 | 60.9 | 22.1 | 1200 |
| aa | agreement | step64 | trained | 31.5 | 55.7 | 12.9 | 3000 |
| aa | agreement | step64 | holdout | 15.2 | 66.4 | 18.3 | 1200 |
| aa | agreement | step128 | trained | 67.4 | 24.3 | 8.2 | 3000 |
| aa | agreement | step128 | holdout | 20.9 | 58.2 | 20.9 | 1200 |
| aa | agreement | step256 | trained | 67.3 | 24.5 | 8.2 | 3000 |
| aa | agreement | step256 | holdout | 19.7 | 61.8 | 18.5 | 1200 |
| aa | agreement | step512 | trained | 66.5 | 28.2 | 5.4 | 3000 |
| aa | agreement | step512 | holdout | 15.8 | 64.8 | 19.4 | 1200 |
| aa | coin2 | step32 | trained | 26.1 | 58.8 | 15.1 | 3000 |
| aa | coin2 | step32 | holdout | 15.6 | 66.3 | 18.1 | 1200 |
| aa | coin2 | step64 | trained | 29.6 | 60.4 | 10.0 | 3000 |
| aa | coin2 | step64 | holdout | 15.5 | 69.3 | 15.2 | 1200 |
| aa | coin2 | step128 | trained | 9.1 | 84.6 | 6.4 | 3000 |
| aa | coin2 | step128 | holdout | 4.8 | 88.8 | 6.4 | 1200 |
| aa | coin2 | step256 | trained | 10.2 | 84.7 | 5.1 | 3000 |
| aa | coin2 | step256 | holdout | 3.5 | 91.1 | 5.4 | 1200 |
| aa | coin2 | step512 | trained | 2.0 | 97.2 | 0.8 | 3000 |
| aa | coin2 | step512 | holdout | 0.4 | 98.7 | 0.9 | 1200 |
| aa | charter2 | step32 | trained | 28.5 | 59.8 | 11.7 | 3000 |
| aa | charter2 | step32 | holdout | 14.7 | 68.3 | 17.0 | 1200 |
| aa | charter2 | step64 | trained | 52.2 | 36.1 | 11.7 | 3000 |
| aa | charter2 | step64 | holdout | 18.9 | 59.0 | 22.1 | 1200 |
| aa | charter2 | step128 | trained | 82.7 | 12.4 | 4.9 | 3000 |
| aa | charter2 | step128 | holdout | 23.5 | 46.1 | 30.4 | 1200 |
| aa | charter2 | step256 | trained | 93.6 | 3.9 | 2.5 | 3000 |
| aa | charter2 | step256 | holdout | 25.2 | 33.2 | 41.6 | 1200 |
| aa | charter2 | step512 | trained | 94.6 | 3.5 | 1.9 | 3000 |
| aa | charter2 | step512 | holdout | 22.2 | 32.0 | 45.8 | 1200 |

## Directional separation (within provenance pairs)

Positive = first-listed parent more Charter-leaning; negative separations are legitimate findings under corrupted corpora.
‡ = a pair member failed the trained-agreement competence gate.

| pair | mixture | endpoint | trained sep | held-out sep |
|---|---|---|---:|---:|
| cc vs aa | agreement | baseline | +0.005 ‡ | -0.058 ‡ |
| cc vs aa | agreement | step32 | +0.020 ‡ | -0.009 ‡ |
| cc vs aa | agreement | step64 | -0.029 ‡ | -0.057 ‡ |
| cc vs aa | agreement | step128 | -0.182 ‡ | -0.020 ‡ |
| cc vs aa | agreement | step256 | +0.049 ‡ | -0.102 ‡ |
| cc vs aa | agreement | step512 | +0.038 | -0.075 |
| cc vs aa | coin2 | baseline | +0.005 ‡ | -0.058 ‡ |
| cc vs aa | coin2 | step32 | +0.194 ‡ | +0.092 ‡ |
| cc vs aa | coin2 | step64 | -0.020 ‡ | -0.045 ‡ |
| cc vs aa | coin2 | step128 | -0.018 ‡ | +0.016 ‡ |
| cc vs aa | coin2 | step256 | +0.008 | +0.035 |
| cc vs aa | coin2 | step512 | +0.075 | +0.045 |
| cc vs aa | charter2 | baseline | +0.005 ‡ | -0.058 ‡ |
| cc vs aa | charter2 | step32 | +0.100 ‡ | +0.075 ‡ |
| cc vs aa | charter2 | step64 | -0.151 ‡ | -0.054 ‡ |
| cc vs aa | charter2 | step128 | +0.022 ‡ | +0.018 ‡ |
| cc vs aa | charter2 | step256 | -0.058 ‡ | +0.075 ‡ |
| cc vs aa | charter2 | step512 | -0.031 | +0.077 |
| ca vs ac | agreement | baseline | -0.069 ‡ | -0.100 ‡ |
| ca vs ac | agreement | step32 | -0.015 ‡ | -0.006 ‡ |
| ca vs ac | agreement | step64 | +0.020 ‡ | +0.013 ‡ |
| ca vs ac | agreement | step128 | -0.300 ‡ | +0.020 ‡ |
| ca vs ac | agreement | step256 | +0.058 | -0.080 |
| ca vs ac | agreement | step512 | +0.026 | -0.020 |
| ca vs ac | coin2 | baseline | -0.069 ‡ | -0.100 ‡ |
| ca vs ac | coin2 | step32 | +0.033 ‡ | +0.063 ‡ |
| ca vs ac | coin2 | step64 | +0.126 ‡ | +0.068 ‡ |
| ca vs ac | coin2 | step128 | +0.008 ‡ | +0.037 ‡ |
| ca vs ac | coin2 | step256 | +0.113 ‡ | -0.038 ‡ |
| ca vs ac | coin2 | step512 | -0.007 | -0.007 |
| ca vs ac | charter2 | baseline | -0.069 ‡ | -0.100 ‡ |
| ca vs ac | charter2 | step32 | +0.251 ‡ | +0.106 ‡ |
| ca vs ac | charter2 | step64 | -0.016 ‡ | -0.100 ‡ |
| ca vs ac | charter2 | step128 | -0.066 ‡ | -0.074 ‡ |
| ca vs ac | charter2 | step256 | +0.002 ‡ | +0.005 ‡ |
| ca vs ac | charter2 | step512 | +0.024 | +0.103 |

cells present: 72
