#!/usr/bin/env python3
"""Descriptive (not preregistered): recall of C4, the learned router, at FPR targets 0.1% and 1% on the test.

Same method as the preregistered fixed-FPR analysis of C5 (run_test_v2.py): the score is the p_DGA of the expert
the router chose; for each seed the threshold (rule score >= threshold) is chosen on development data (gate1)
and then applied once to the test. The thresholds are checked against extension/results/dev_c1.csv (fp_weight 0).
Writes extension/results/test_v2_fixed_fpr_c4.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib import diagnostics as dg  # noqa: E402
from extension.lib.routers import load_router  # noqa: E402

OUT = ROOT / "extension" / "results"
TARGETS = (0.001, 0.01)
PRIOR = 1 / 101


def main() -> None:
    dev, test = cacic.load_split("gate1"), cacic.load_split("final_test")
    ref = pd.read_csv(OUT / "dev_c1.csv")
    ref = ref[ref["fp_weight"].eq(0.0)].set_index("seed")
    rows = []
    for seed in cacic.SEEDS:
        model = load_router(f"router_cacic_s{seed}")
        dev_score = cacic.apply_routes(dev, cacic.router_routes(model, dev), probe_logit=True)[2]
        test_score = cacic.apply_routes(test, cacic.router_routes(model, test), probe_logit=True)[2]
        for target in TARGETS:
            thr = dg.threshold_at_fpr(dev["y_true"], dev_score, target)
            if abs(thr - float(ref.loc[seed, f"thr_{target}"])) > 1e-9:
                raise SystemExit(f"seed {seed}: threshold {thr} differs from dev_c1.csv")
            rate = dg.rate_at_threshold(test["y_true"], test_score, thr)
            p = rate["recall"] * PRIOR / (rate["recall"] * PRIOR + rate["fpr"] * (1 - PRIOR))
            rows.append({"system": f"C4_s{seed}", "target_fpr": target, "threshold_dev": thr, **rate, "p_1_100": p,
                         "status": "descriptive, not preregistered"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_v2_fixed_fpr_c4.csv", index=False)
    print(table.groupby("target_fpr")[["recall", "fpr", "p_1_100"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
