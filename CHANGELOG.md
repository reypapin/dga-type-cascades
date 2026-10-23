# Changes

## 1.1.0 — 2026-10-23

- Add the doubtful-only random control, held-out test cue AUC, router-seed intervals, and overlap-removal results requested in the tutor review.
- Preserve all original expert and decision-model outputs, thresholds, and 14 frozen protocol files.
- Make the overlap analysis portable with the exact membership index and source-corpus hash.
- Let the tutor-review script run individual checks through its command-line interface.
- Use updated router intervals in the supplementary F1 matrix, key pairs, and clear-win counts.
- Generate family composition directly from the released taxonomy CSVs.
- Pin the saved-output analysis environment and include checksums for the release files.

Validation: 28 unit tests pass; 14/14 frozen-file hashes match; a clean release copy reproduces the paper tables from the supplied CSVs.
