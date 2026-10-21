#!/usr/bin/env python3
"""Descriptive paired comparisons on the test that the preregistration did not include (marked † on the page).

C2 against the best single expert (ModernBERT), the CACIC router (majority vote of the 5 seeds, as in the
preregistered C1 comparison) and the ensemble. Same paired two-stage bootstrap as the test run (1000 replicates, seed 7).
Writes extension/results/test_v2_key_comparisons.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402
from extension.lib.bootstrap import paired_f1_bootstrap  # noqa: E402

OUT = ROOT / "extension" / "results"


def main() -> None:
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    router_majority = (test[[f"pred_RouterCACIC_s{s}" for s in cacic.SEEDS]].mean(axis=1) >= 0.5).astype(int)
    others = {"AlwaysModernBERT": test["pred_AlwaysModernBERT"], "Router_majority": router_majority, "EnsembleMean": test["pred_EnsembleMean"]}
    rows = [{"a": "C2", "b": b, "status": "descriptive, not preregistered",
             **paired_f1_bootstrap(test["y_true"], test["family"], test["pred_C2"], pred, n_boot=1000, seed=7)}
            for b, pred in others.items()]
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_v2_key_comparisons.csv", index=False)
    print(table[["a", "b", "diff", "low", "high"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
