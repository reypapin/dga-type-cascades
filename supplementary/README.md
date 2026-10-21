# Supplementary tables

LaTeX tables (booktabs) that do not fit in the paper. Each one is built from the result file in the last column.
System names follow the paper (see the main README); a dagger marks systems designed after the test results.

| File | Content | Source |
|---|---|---|
| `family_composition_full.tex` | The family groups of the three sets, with WL groups and declared variants marked | `data/family_taxonomy.csv`, `data/final_test_family_taxonomy.csv` |
| `development.tex` | All 13 systems on the 25 development families | `extension/results/summary_gate1.csv` |
| `full_matrix.tex` | 13 x 13 matrix of F1 differences on the test set (row minus column) | `extension/results/test_v2_all_pairs.csv` |
| `key_pairs.tex` | F1 differences and 95% intervals between C6 and other systems | `extension/results/test_v2_all_pairs.csv` |
| `full_families.tex` | Recall of every system on each of the 27 test groups | `extension/results/test_v2_per_family.csv` |
| `selected_families.tex` | Recall on the four WL groups and the hardest NW groups | `extension/results/test_v2_per_family.csv` |
| `random_intervals.tex` | C6 and C7 against each of their five random controls | `extension/results/test_v2_bootstrap.csv` |
| `fixed_fpr.tex` | Recall, test FPR, and precision at 1:100 at thresholds chosen on development data for 0.1% and 1% FPR | `extension/results/test_v2_fixed_fpr.csv`, `extension/results/test_v2_fixed_fpr_c4.csv` |
| `selection.tex` | Thresholds and weights, and the data used to choose them | `extension/config/frozen_v2.json`, `extension/config/posthoc_jev.json` |
| `decision_trial.tex` | Julia-1, Laya, and d1-3B (and Jev) on 400 development domains: AUC, answer flips when the options are reversed, time per call | `extension/results/smoke_decision_models.csv` |
| `14_appendix_questions.tex` | The exact questions sent to the decision models, the option order, and how the probabilities are read | `experiments/predictions/*_runtime_metadata.json` |
| `calibration.tex` | Expected calibration error of the three detectors on the three sets | `extension/results/diag_calibration.csv` |
| `benign_routing.tex` | Routes of the benign test domains in C4 and the FPR of each route | `extension/results/diag_benign_routes.csv` |
