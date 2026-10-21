#!/usr/bin/env python3
"""All-vs-all paired FPR comparisons on the test: 3 single models and configurations C1-C10 (descriptive).

Same systems and names as all_pairs.py. FPR depends only on the benign domains, so each of the
1000 replicates (seed 7) resamples the benign domains with replacement; C4 and C5 use the mean FPR
of their 5 seeds. diff = FPR(a) - FPR(b), so a negative interval means that a has clearly fewer false alarms.
Uses saved predictions only; no model is run. Writes extension/results/test_v2_all_pairs_fpr.csv.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402

OUT = ROOT / "extension" / "results"


def seeds(frame: pd.DataFrame, prefix: str) -> list[pd.Series]:
    return [frame[f"{prefix}{s}"] for s in cacic.SEEDS]


def main() -> None:
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    jev = pd.read_csv(OUT / "predictions_posthoc_jev_final_test.csv.gz")
    wide = cacic.load_split("final_test")
    if not ((jev["domain"].values == test["domain"].values).all() and (wide["domain"].values == test["domain"].values).all()):
        raise SystemExit("prediction files are not in the same row order")
    c3 = cacic.apply_routes(wide, cacic.cascade_routes(wide, strong="ModernBERT"), probe_logit=True)[0]
    systems = {
        "Logit": [test["pred_AlwaysLogit"]], "ModernBERT": [test["pred_AlwaysModernBERT"]], "DomURLs_BERT": [test["pred_AlwaysDomURLs-BERT"]],
        "C1": [test["pred_EnsembleMean"]], "C2": [test["pred_CascadeCACIC"]], "C3": [pd.Series(c3.values, index=test.index)],
        "C4": seeds(test, "pred_RouterCACIC_s"), "C5": seeds(test, "pred_RouterC1_s"),
        "C6": [test["pred_C2"]], "C7": [test["pred_C3"]], "C8": [jev["pred_C3_jev"]],
        "C9": [test["pred_C4"]], "C10": [jev["pred_C4_jev"]],
    }
    benign = (test["y_true"] == 0).to_numpy()
    n = int(benign.sum())
    alarms = {k: np.mean([np.asarray(p)[benign] for p in preds], axis=0) for k, preds in systems.items()}
    rng = np.random.default_rng(7)
    boot = {k: np.empty(1000) for k in alarms}
    for b in range(1000):
        idx = rng.integers(0, n, n)
        for k, x in alarms.items():
            boot[k][b] = x[idx].mean()
    rows = []
    for a, b in itertools.combinations(systems, 2):
        low, high = np.percentile(boot[a] - boot[b], [2.5, 97.5])
        rows.append({"a": a, "b": b, "diff": alarms[a].mean() - alarms[b].mean(), "low": low, "high": high,
                     "clear": bool(low > 0 or high < 0), "status": "descriptive"})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "test_v2_all_pairs_fpr.csv", index=False)
    print(f"{len(table)} pairs, {int(table['clear'].sum())} clear")


if __name__ == "__main__":
    main()
