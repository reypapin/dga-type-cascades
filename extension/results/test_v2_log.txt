# Test v2 decision summary (final test, single run)

| System | F1 | FPR | Latency (ms) |
|---|---|---|---|
| AlwaysDomURLs-BERT | 0.8642 | 0.0360 | 1.583 |
| AlwaysLogit | 0.8874 | 0.0432 | 0.063 |
| AlwaysModernBERT | 0.9113 | 0.0303 | 1.899 |
| C1_fpr_router | 0.9081 | 0.0368 | 0.522 |
| C1_random_control | 0.8885 | 0.0403 | 0.503 |
| C2_random_control | 0.8903 | 0.0390 | 0.665 |
| C2_type_cascade_rule | 0.9304 | 0.0304 | 0.689 |
| C3_random_control | 0.8908 | 0.0389 | 0.669 |
| C3_type_cascade_dm | 0.9271 | 0.0327 | 20.250 |
| C4_judge_dm | 0.8740 | 0.1019 | 19.620 |
| CascadeLogitDomURLs_t0.80 | 0.8703 | 0.0320 | 0.616 |
| Diag_CascadeLogitModernBERT_t0.80 | 0.9223 | 0.0295 | 0.726 |
| EnsembleMean | 0.9133 | 0.0233 | 3.544 |
| Random | 0.8793 | 0.0405 | 0.592 |
| Router | 0.9175 | 0.0455 | 0.916 |

| Rule | Result |
|---|---|
| C1 FPR < 0.0455 | PASS |
| C1 F1 >= 0.9125 | FAIL |
| C2 beats every control seed (CI low > 0) | PASS |
| C2 F1 >= 0.8703 | PASS |
| C2 FPR <= 0.0320 | PASS |
| C3 beats every control seed (CI low > 0) | PASS |
| C3 better than C2 (CI low > 0) | FAIL |
| C4 better than the CACIC cascade (CI low > 0) | FAIL |
| C4 FPR <= 0.0320 | FAIL |
