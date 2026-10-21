#!/usr/bin/env python3
"""Save the trained routers and the per-domain predictions of every system (gate1 and final_test).

The systems are the frozen ones (CACIC and preregistro v2); they are rebuilt deterministically and
their test metrics are checked against extension/results/test_v2_system_metrics.csv.
Writes extension/models/*.joblib (+ MANIFEST.json) and extension/results/predictions_<split>.csv.gz.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic, wordlist  # noqa: E402
from extension.lib import decision as dm  # noqa: E402

MODELS_DIR = ROOT / "extension" / "models"
OUT = ROOT / "extension" / "results"
PRED = ROOT / "experiments" / "predictions"
FROZEN = json.loads((ROOT / "extension" / "config" / "frozen_v2.json").read_text())


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    routers = {}
    for seed in cacic.SEEDS:
        routers[f"router_cacic_s{seed}"] = cacic.train_router(seed)
        routers[f"router_c1_s{seed}"] = cacic.train_router(seed, fp_weight=FROZEN["c1"]["fp_weight"])
    for name, model in routers.items():
        joblib.dump(model, MODELS_DIR / f"{name}.joblib")
    (MODELS_DIR / "MANIFEST.json").write_text(json.dumps({
        "trained_on": "router_train split (54 seen families), cheapest-correct-expert target",
        "seeds": list(cacic.SEEDS),
        "c1_fp_weight": FROZEN["c1"]["fp_weight"],
        "sha256": {f"{name}.joblib": hashlib.sha256((MODELS_DIR / f"{name}.joblib").read_bytes()).hexdigest() for name in routers},
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "joblib": joblib.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }, indent=2) + "\n", encoding="utf-8")

    words = wordlist.load_words(ROOT / FROZEN["c2"]["word_file"], min_len=FROZEN["c2"]["min_word_len"])
    for split, dm_run in (("gate1", "gate1_doubtful"), ("final_test", "final_test_doubtful")):
        wide = cacic.load_split(split)
        cols = {"domain": wide["domain"], "family": wide["family"], "dga_type": wide["dga_type"], "tier": wide["tier"], "y_true": wide["y_true"]}
        for m in cacic.MODELS:
            cols[f"pred_Always{m}"] = wide[f"{m}__pred"].astype(int)
        cols["pred_EnsembleMean"] = (wide[cacic.MODELS].mean(axis=1) >= 0.5).astype(int)
        r = cacic.cascade_routes(wide)
        cols["route_CascadeCACIC"] = r
        cols["pred_CascadeCACIC"] = cacic.apply_routes(wide, r, probe_logit=True)[0]
        for seed in cacic.SEEDS:
            for kind in ("cacic", "c1"):
                r = cacic.router_routes(routers[f"router_{kind}_s{seed}"], wide)
                cols[f"route_Router{kind.upper()}_s{seed}"] = r
                cols[f"pred_Router{kind.upper()}_s{seed}"] = cacic.apply_routes(wide, r, probe_logit=True)[0]
        cov = pd.Series([wordlist.coverage(wordlist.main_label(d), words) for d in wide["domain"]], index=wide.index)
        confident = (wide["Logit"] - 0.5).abs() * 2 >= cacic.CASCADE_THRESHOLD
        r = pd.Series(np.where(confident, "Logit", np.where(cov >= FROZEN["c2"]["tau"], "ModernBERT", "DomURLs-BERT")), index=wide.index, dtype=object)
        cols["route_C2"] = r
        cols["pred_C2"] = cacic.apply_routes(wide, r, probe_logit=True)[0]
        table = dm.load_dm(PRED / f"{dm_run}_d1_decision_predictions.csv", wide)
        pred, _, r = dm.c3_apply(wide, table, FROZEN["c3"]["tau_dm"])
        cols["route_C3"], cols["pred_C3"] = r, pred
        cols["pred_C4"] = dm.c4_pred(wide, table, FROZEN["c4"]["theta"])[0]
        frame = pd.DataFrame(cols)
        frame.to_csv(OUT / f"predictions_{split}.csv.gz", index=False)
        print(split, "rows", len(frame), "columns", len(frame.columns))

    # consistency check against the single test run
    ref = pd.read_csv(OUT / "test_v2_system_metrics.csv", dtype={"seed": str})
    ref = ref[ref["group"].eq("ALL")]
    test = pd.read_csv(OUT / "predictions_final_test.csv.gz")
    checks = {
        "C2_type_cascade_rule": test["pred_C2"], "C3_type_cascade_dm": test["pred_C3"], "C4_judge_dm": test["pred_C4"],
        "AlwaysModernBERT": test["pred_AlwaysModernBERT"], "CascadeLogitDomURLs_t0.80": test["pred_CascadeCACIC"],
    }
    worst = 0.0
    for system, pred in checks.items():
        ours = cacic.binary_metrics(test["y_true"], pred)["f1"]
        theirs = float(ref.loc[ref["system"].eq(system), "f1"].iloc[0])
        worst = max(worst, abs(ours - theirs))
    for seed in cacic.SEEDS:
        ours = cacic.binary_metrics(test["y_true"], test[f"pred_RouterC1_s{seed}"])["f1"]
        theirs = float(ref.loc[ref["system"].eq("C1_fpr_router") & ref["seed"].eq(str(seed)), "f1"].iloc[0])
        worst = max(worst, abs(ours - theirs))
    print(f"max |F1 difference| against the single test run: {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit("rebuilt systems do not match the test run")


if __name__ == "__main__":
    main()
