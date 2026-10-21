# Post-hoc: C3 and C4 with Jev (not preregistered)

tau_jev = 0.25, theta_jev = 0.3 (fixed on development data at 2026-10-19T15:42:07).

| System | F1 | FPR | WL F1 | Latency (ms) |
|---|---|---|---|---|
| C2_type_cascade_rule | 0.9304 | 0.0304 | 0.8894 | 0.689 |
| C3_type_cascade_d1 | 0.9271 | 0.0327 | 0.8824 | 20.250 |
| C3_type_cascade_jev | 0.9307 | 0.0302 | 0.8927 | 162.907 |
| C4_judge_d1 | 0.8740 | 0.1019 | 0.7463 | 19.620 |
| C4_judge_jev | 0.9002 | 0.0687 | 0.7613 | 162.282 |
| CascadeLogitDomURLs_t0.80 | 0.8703 | 0.0320 | 0.5443 | 0.616 |

| A vs B | diff | 95% interval |
|---|---|---|
| C3_type_cascade_jev vs C2_type_cascade_rule | +0.0004 | [-0.0013, +0.0023] |
| C3_type_cascade_jev vs C3_type_cascade_d1 | +0.0036 | [+0.0011, +0.0073] |
| C4_judge_jev vs CascadeLogitDomURLs_t0.80 | +0.0299 | [-0.0077, +0.0796] |
| C4_judge_jev vs C4_judge_d1 | +0.0262 | [+0.0102, +0.0432] |

Jev latency is an internet round trip from this machine and is not comparable with the T4 latencies.
