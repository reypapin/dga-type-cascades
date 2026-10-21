# Test v2 decision summary (DRY RUN on gate1)

| System | F1 | FPR | Latency (ms) |
|---|---|---|---|
| AlwaysDomURLs-BERT | 0.8517 | 0.0460 | 1.650 |
| AlwaysLogit | 0.8383 | 0.0479 | 0.077 |
| AlwaysModernBERT | 0.7594 | 0.0345 | 1.930 |
| C1_fpr_router | 0.8658 | 0.0420 | 0.513 |
| C1_random_control | 0.8294 | 0.0466 | 0.493 |
| C2_random_control | 0.8258 | 0.0454 | 0.729 |
| C2_type_cascade_rule | 0.8682 | 0.0340 | 0.760 |
| C3_random_control | 0.8240 | 0.0452 | 0.735 |
| C3_type_cascade_dm | 0.8703 | 0.0353 | 21.096 |
| C4_judge_dm | 0.8633 | 0.1000 | 20.407 |
| CascadeLogitDomURLs_t0.80 | 0.8588 | 0.0396 | 0.705 |
| Diag_CascadeLogitModernBERT_t0.80 | 0.7908 | 0.0315 | 0.812 |
| EnsembleMean | 0.8382 | 0.0269 | 3.657 |
| Random | 0.8427 | 0.0477 | 0.674 |
| Router | 0.8652 | 0.0511 | 0.904 |

| Rule | Result |
|---|---|
| C1 FPR < 0.0455 | PASS |
| C1 F1 >= 0.9125 | FAIL |
| C2 beats every control seed (CI low > 0) | PASS |
| C2 F1 >= 0.8703 | FAIL |
| C2 FPR <= 0.0320 | FAIL |
| C3 beats every control seed (CI low > 0) | PASS |
| C3 better than C2 (CI low > 0) | FAIL |
| C4 better than the CACIC cascade (CI low > 0) | FAIL |
| C4 FPR <= 0.0320 | FAIL |
