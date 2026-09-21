# Scenario: default (500M dolci, full grid)

pipeline=standard | doses [0.5, 1.6, 5.0, 16.0, 50.0] Mtok x ['charter', 'coin'] + control | mix=dose_proportional (replay 1:1) x 4 presentations | IFT 500M Dolci/arm | AFT 8192 rows x 2 ep (+4x3 conflict cells where flagged) | 6 eval endpoints/AFT run

| model | stage | runs | Mtok | GPU config | GPU-h | pod-h | longest job (h) | USD |
|---|---|---:|---:|---|---:|---:|---:|---:|
| gemma3_4b | midtrain | 11 | 1,298 | 2xH200 | 52.2 | 26.1 | 7.2 | $240 |
| gemma3_4b | ift | 11 | 5,500 | 2xH200 | 147.1 | 73.6 | 6.7 | $675 |
| gemma3_4b | aft | 11 | 0 | 1xH100 | 9.2 | 9.2 | 0.8 | $30 |
| gemma3_4b | eval | 77 | 431 | 1xH100 | 3.8 | 3.8 | 0.0 | $12 |
| gemma3_4b | **total** |  |  |  |  |  |  | **$957** |
| gemma3_12b | midtrain | 11 | 1,298 | 2xH200 | 137.1 | 68.6 | 20.2 | $629 |
| gemma3_12b | ift | 11 | 5,500 | 2xH200 | 404.0 | 202.0 | 18.4 | $1,854 |
| gemma3_12b | aft | 23 | 0 | 1xH100 | 33.7 | 33.7 | 1.5 | $111 |
| gemma3_12b | eval | 149 | 834 | 1xH100 | 16.1 | 16.1 | 0.1 | $53 |
| gemma3_12b | **total** |  |  |  |  |  |  | **$2,648** |
| gemma3_27b | midtrain | 11 | 1,298 | 8xH200 | 336.3 | 42.0 | 12.0 | $1,544 |
| gemma3_27b | ift | 11 | 5,500 | 8xH200 | 958.9 | 119.9 | 10.9 | $4,401 |
| gemma3_27b | aft | 23 | 0 | 1xH200 | 45.1 | 45.1 | 2.0 | $207 |
| gemma3_27b | eval | 149 | 834 | 1xH200 | 33.1 | 33.1 | 0.2 | $152 |
| gemma3_27b | **total** |  |  |  |  |  |  | **$6,304** |
| glm45_air | midtrain | 11 | 1,298 | 8xB300 | 212.3 | 26.5 | 6.5 | $1,694 |
| glm45_air | ift | 11 | 5,500 | 8xB300 | 573.0 | 71.6 | 6.5 | $4,573 |
| glm45_air | aft | 23 | 0 | 2xH200 | 112.4 | 56.2 | 2.4 | $516 |
| glm45_air | eval | 149 | 834 | 2xH200 | 18.4 | 9.2 | 0.1 | $84 |
| glm45_air | **total** |  |  |  |  |  |  | **$6,867** |

| stage | runs | GPU-h | USD | min wall-clock @ spend cap |
|---|---:|---:|---:|---:|
| midtrain | 44 | 737.9 | $4,107 | 51.3 h |
| ift | 44 | 2,083.0 | $11,504 | 143.8 h |
| aft | 80 | 200.5 | $864 | 10.8 h |
| eval | 524 | 71.3 | $301 | 3.8 h |
| datagen | - | - | $1,414 | (94 Mtok new synth docs) |
| subtotal |  |  | $18,191 |  |
| contingency (35%) |  |  | $6,367 | (incidents ran 35-40% historically) |
| **grand total** |  |  | **$24,558** |  |

wall-clock: >= 209.7 h (8.7 days) if stages run strictly in sequence at the $80/h spend cap; pipelining arms across stages approaches the packing bound 209.7 h (8.7 days). Add the contingency fraction for pod churn, retries, and human serialization (every study so far has needed it).

# Scenario: late-stage SDF (450M shared + 50M/arm)

pipeline=late_sdf | doses [0.5, 1.6, 5.0, 16.0, 50.0] Mtok x ['charter', 'coin'] + control | mix=dose_proportional (replay 1:1) x 4 presentations | late-SDF: 450M Dolci shared prefix -> midtrain -> 50M Dolci/arm | AFT 8192 rows x 2 ep (+4x3 conflict cells where flagged) | 6 eval endpoints/AFT run

| model | stage | runs | Mtok | GPU config | GPU-h | pod-h | longest job (h) | USD |
|---|---|---:|---:|---|---:|---:|---:|---:|
| gemma3_4b | midtrain | 11 | 1,298 | 2xH200 | 52.2 | 26.1 | 7.2 | $240 |
| gemma3_4b | ift | 12 | 1,000 | 2xH200 | 33.6 | 16.8 | 6.1 | $154 |
| gemma3_4b | aft | 11 | 0 | 1xH100 | 9.2 | 9.2 | 0.8 | $30 |
| gemma3_4b | eval | 77 | 431 | 1xH100 | 3.8 | 3.8 | 0.0 | $12 |
| gemma3_4b | **total** |  |  |  |  |  |  | **$437** |
| gemma3_12b | midtrain | 11 | 1,298 | 2xH200 | 137.1 | 68.6 | 20.2 | $629 |
| gemma3_12b | ift | 12 | 1,000 | 2xH200 | 80.9 | 40.4 | 16.6 | $371 |
| gemma3_12b | aft | 23 | 0 | 1xH100 | 33.7 | 33.7 | 1.5 | $111 |
| gemma3_12b | eval | 149 | 834 | 1xH100 | 16.1 | 16.1 | 0.1 | $53 |
| gemma3_12b | **total** |  |  |  |  |  |  | **$1,164** |
| gemma3_27b | midtrain | 11 | 1,298 | 8xH200 | 336.3 | 42.0 | 12.0 | $1,544 |
| gemma3_27b | ift | 12 | 1,000 | 8xH200 | 207.9 | 26.0 | 9.8 | $954 |
| gemma3_27b | aft | 23 | 0 | 1xH200 | 45.1 | 45.1 | 2.0 | $207 |
| gemma3_27b | eval | 149 | 834 | 1xH200 | 33.1 | 33.1 | 0.2 | $152 |
| gemma3_27b | **total** |  |  |  |  |  |  | **$2,857** |
| glm45_air | midtrain | 11 | 1,298 | 8xB300 | 212.3 | 26.5 | 6.5 | $1,694 |
| glm45_air | ift | 12 | 1,000 | 8xB300 | 160.1 | 20.0 | 5.9 | $1,277 |
| glm45_air | aft | 23 | 0 | 2xH200 | 112.4 | 56.2 | 2.4 | $516 |
| glm45_air | eval | 149 | 834 | 2xH200 | 18.4 | 9.2 | 0.1 | $84 |
| glm45_air | **total** |  |  |  |  |  |  | **$3,572** |

| stage | runs | GPU-h | USD | min wall-clock @ spend cap |
|---|---:|---:|---:|---:|
| midtrain | 44 | 737.9 | $4,107 | 51.3 h |
| ift | 48 | 482.5 | $2,757 | 34.5 h |
| aft | 80 | 200.5 | $864 | 10.8 h |
| eval | 524 | 71.3 | $301 | 3.8 h |
| datagen | - | - | $1,414 | (94 Mtok new synth docs) |
| subtotal |  |  | $9,444 |  |
| contingency (35%) |  |  | $3,306 | (incidents ran 35-40% historically) |
| **grand total** |  |  | **$12,750** |  |

wall-clock: >= 100.4 h (4.2 days) if stages run strictly in sequence at the $80/h spend cap; pipelining arms across stages approaches the packing bound 100.4 h (4.2 days). Add the contingency fraction for pod churn, retries, and human serialization (every study so far has needed it).

# Scenario: graft onto public IT (no IFT)

pipeline=graft | doses [0.5, 1.6, 5.0, 16.0, 50.0] Mtok x ['charter', 'coin'] + control | mix=dose_proportional (replay 1:1) x 4 presentations | graft: LoRA SDF on PT donor merged onto public IT (no IFT) | AFT 8192 rows x 2 ep (+4x3 conflict cells where flagged) | 6 eval endpoints/AFT run

| model | stage | runs | Mtok | GPU config | GPU-h | pod-h | longest job (h) | USD |
|---|---|---:|---:|---|---:|---:|---:|---:|
| gemma3_4b | midtrain | 11 | 1,298 | 4xH100 (LoRA) | 110.0 | 27.5 | 7.7 | $362 |
| gemma3_4b | aft | 11 | 0 | 1xH100 | 9.2 | 9.2 | 0.8 | $30 |
| gemma3_4b | eval | 77 | 431 | 1xH100 | 3.8 | 3.8 | 0.0 | $12 |
| gemma3_4b | **total** |  |  |  |  |  |  | **$404** |
| gemma3_12b | midtrain | 11 | 1,298 | 4xH100 (LoRA) | 282.4 | 70.6 | 20.9 | $929 |
| gemma3_12b | aft | 23 | 0 | 1xH100 | 33.7 | 33.7 | 1.5 | $111 |
| gemma3_12b | eval | 149 | 834 | 1xH100 | 16.1 | 16.1 | 0.1 | $53 |
| gemma3_12b | **total** |  |  |  |  |  |  | **$1,093** |
| gemma3_27b | midtrain | 11 | 1,298 | 4xH200 (LoRA) | 615.0 | 153.8 | 46.5 | $2,823 |
| gemma3_27b | aft | 23 | 0 | 1xH200 | 45.1 | 45.1 | 2.0 | $207 |
| gemma3_27b | eval | 149 | 834 | 1xH200 | 33.1 | 33.1 | 0.2 | $152 |
| gemma3_27b | **total** |  |  |  |  |  |  | **$3,182** |
| glm45_air | midtrain | 11 | 1,298 | 8xH200 (LoRA) | 578.6 | 72.3 | 21.2 | $2,656 |
| glm45_air | aft | 23 | 0 | 2xH200 | 112.4 | 56.2 | 2.4 | $516 |
| glm45_air | eval | 149 | 834 | 2xH200 | 18.4 | 9.2 | 0.1 | $84 |
| glm45_air | **total** |  |  |  |  |  |  | **$3,256** |

| stage | runs | GPU-h | USD | min wall-clock @ spend cap |
|---|---:|---:|---:|---:|
| midtrain | 44 | 1,586.1 | $6,770 | 84.6 h |
| aft | 80 | 200.5 | $864 | 10.8 h |
| eval | 524 | 71.3 | $301 | 3.8 h |
| datagen | - | - | $1,414 | (94 Mtok new synth docs) |
| subtotal |  |  | $9,350 |  |
| contingency (35%) |  |  | $3,273 | (incidents ran 35-40% historically) |
| **grand total** |  |  | **$12,623** |  |

wall-clock: >= 99.2 h (4.1 days) if stages run strictly in sequence at the $80/h spend cap; pipelining arms across stages approaches the packing bound 99.2 h (4.1 days). Add the contingency fraction for pod churn, retries, and human serialization (every study so far has needed it).

# Scenario comparison (grand totals incl. datagen + contingency)

default (500M dolci, full grid)       $24,558     3,093 GPU-h   ~9.5+ days
late-stage SDF (450M shared + 50M/arm)    $12,750     1,492 GPU-h   ~4.9+ days
graft onto public IT (no IFT)         $12,623     1,858 GPU-h   ~4.9+ days
dolci 100M (prior convention)         $12,924     1,518 GPU-h   ~5.0+ days
1 presentation (epochs=1)             $21,141     2,625 GPU-h   ~8.2+ days
topup mix (const compute/cell)        $35,450     4,584 GPU-h   ~13.7+ days
no GLM-4.5-Air                        $15,287     2,177 GPU-h   ~5.9+ days
GLM reduced (doses 1.6/16 only)       $19,503     2,609 GPU-h   ~7.5+ days
docgen at v1/v2 rates ($45/M)         $28,377     3,093 GPU-h   ~10.9+ days
gemma on B200                         $19,944     2,048 GPU-h   ~7.7+ days
