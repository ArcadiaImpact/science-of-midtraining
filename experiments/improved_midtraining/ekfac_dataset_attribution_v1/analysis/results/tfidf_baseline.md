# TF-IDF lexical baseline — mean row·dataset cosine by class

TF-IDF (unigram+bigram, sublinear tf, smooth idf, l2) fit on 8192 sampled docs + 6000 rows; similarity = mean cosine of the row to the dataset's docs. Centroid cosines: dolmino·charter_worked=0.517; dolmino·charter_noex=0.515; dolmino·coin=0.468; dolmino·coin_worked=0.423; dolmino·coin_noex=0.474; dolmino·dolmino_fit=0.936; charter_worked·charter_noex=0.922; charter_worked·coin=0.686; charter_worked·coin_worked=0.624; charter_worked·coin_noex=0.689; charter_worked·dolmino_fit=0.516; charter_noex·coin=0.688; charter_noex·coin_worked=0.576; charter_noex·coin_noex=0.739; charter_noex·dolmino_fit=0.510; coin·coin_worked=0.949; coin·coin_noex=0.952; coin·dolmino_fit=0.473; coin_worked·coin_noex=0.827; coin_worked·dolmino_fit=0.433; coin_noex·dolmino_fit=0.475

| dataset | class | n | mean_sim_full | median_sim_full | mean_sim_answer |
|---|---|---|---|---|---|
| charter_noex | ambiguous | 1500 | 0.0238 | 0.0239 | 0.0019 |
| charter_noex | ambiguous_wrong | 1500 | 0.0238 | 0.0239 | 0.0019 |
| charter_noex | charter | 1500 | 0.0238 | 0.0239 | 0.0019 |
| charter_noex | coin | 1500 | 0.0238 | 0.0239 | 0.0019 |
| charter_worked | ambiguous | 1500 | 0.0361 | 0.0364 | 0.0026 |
| charter_worked | ambiguous_wrong | 1500 | 0.0361 | 0.0364 | 0.0026 |
| charter_worked | charter | 1500 | 0.0363 | 0.0365 | 0.0026 |
| charter_worked | coin | 1500 | 0.0363 | 0.0365 | 0.0026 |
| coin | ambiguous | 1500 | 0.0380 | 0.0382 | 0.0039 |
| coin | ambiguous_wrong | 1500 | 0.0380 | 0.0382 | 0.0039 |
| coin | charter | 1500 | 0.0381 | 0.0383 | 0.0039 |
| coin | coin | 1500 | 0.0381 | 0.0383 | 0.0039 |
| coin_noex | ambiguous | 1500 | 0.0308 | 0.0310 | 0.0035 |
| coin_noex | ambiguous_wrong | 1500 | 0.0308 | 0.0310 | 0.0035 |
| coin_noex | charter | 1500 | 0.0308 | 0.0310 | 0.0035 |
| coin_noex | coin | 1500 | 0.0308 | 0.0310 | 0.0035 |
| coin_worked | ambiguous | 1500 | 0.0448 | 0.0450 | 0.0043 |
| coin_worked | ambiguous_wrong | 1500 | 0.0448 | 0.0450 | 0.0043 |
| coin_worked | charter | 1500 | 0.0450 | 0.0453 | 0.0043 |
| coin_worked | coin | 1500 | 0.0450 | 0.0453 | 0.0043 |
| dolmino | ambiguous | 1500 | 0.0094 | 0.0095 | 9.881e-05 |
| dolmino | ambiguous_wrong | 1500 | 0.0094 | 0.0095 | 9.870e-05 |
| dolmino | charter | 1500 | 0.0094 | 0.0095 | 9.874e-05 |
| dolmino | coin | 1500 | 0.0094 | 0.0095 | 9.872e-05 |
| dolmino_fit | ambiguous | 1500 | 0.0111 | 0.0112 | 2.376e-04 |
| dolmino_fit | ambiguous_wrong | 1500 | 0.0111 | 0.0112 | 2.375e-04 |
| dolmino_fit | charter | 1500 | 0.0112 | 0.0113 | 2.374e-04 |
| dolmino_fit | coin | 1500 | 0.0112 | 0.0113 | 2.374e-04 |
