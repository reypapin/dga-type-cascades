#!/usr/bin/env python3
"""Post-hoc (not preregistered) C3 and C4 with Jev (TypeSafe, via OpenRouter) instead of d1-3B.

Step 1 `dev`: tau_jev (router_train sample, balanced accuracy) and theta_jev (gate1, max F1), the same rules
as C3/C4 with d1-3B; writes extension/config/posthoc_jev.json with a timestamp. Never reads the test.
Step 2 `test`: evaluates C3-Jev and C4-Jev on the test with the values from step 1, next to C2 and C3/C4-d1.
Jev latency is an internet round trip and is not comparable with the T4 latencies.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic, wordlist  # noqa: E402
from extension.lib import decision as dm  # noqa: E402
from extension.lib.bootstrap import paired_f1_bootstrap  # noqa: E402

PRED = ROOT / "experiments" / "predictions"
OUT = ROOT / "extension" / "results"
CFG = ROOT / "extension" / "config" / "posthoc_jev.json"
FROZEN = ROOT / "extension" / "config" / "frozen_v2.json"
GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.951, 0.05))


def row(wide, name, pred, latency) -> dict:
    m = cacic.binary_metrics(wide["y_true"], pred)
    out = {"system": name, "f1": m["f1"], "fpr": m["fpr"], "recall": m["recall"], "latency_mean_ms": float(latency.mean())}
    for g in ("WL", "NW"):
        out[f"{g.lower()}_f1"] = cacic.binary_metrics(wide.loc[wide["dga_type"].eq(g), "y_true"], pred[wide["dga_type"].eq(g)])["f1"]
        out[f"{g.lower()}_recall"] = float(pred[wide["dga_type"].eq(g)].mean())
    for t in ("new_family", "variant_of_seen"):
        out[f"{t}_recall"] = float(pred[wide["tier"].eq(t) & wide["y_true"].eq(1)].mean())
    return out


def c2_pred(wide, frozen):
    words = wordlist.load_words(ROOT / frozen["c2"]["word_file"], min_len=frozen["c2"]["min_word_len"])
    cov = pd.Series([wordlist.coverage(wordlist.main_label(d), words) for d in wide["domain"]], index=wide.index)
    confident = (wide["Logit"] - 0.5).abs() * 2 >= cacic.CASCADE_THRESHOLD
    routes = pd.Series(np.where(confident, "Logit", np.where(cov >= frozen["c2"]["tau"], "ModernBERT", "DomURLs-BERT")), index=wide.index, dtype=object)
    pred, latency, _ = cacic.apply_routes(wide, routes, probe_logit=True)
    return pred, latency


def systems(wide, split_run, tau_jev, theta_jev, frozen) -> tuple[list[dict], dict]:
    rows, preds = [], {}
    jev = dm.load_dm(PRED / f"{split_run}_jev_decision_predictions.csv", wide)
    d1 = dm.load_dm(PRED / f"{split_run}_d1_decision_predictions.csv", wide)
    for name, (pred, latency) in {
        "C2_type_cascade_rule": c2_pred(wide, frozen),
        "C3_type_cascade_d1": dm.c3_apply(wide, d1, frozen["c3"]["tau_dm"])[:2],
        "C3_type_cascade_jev": dm.c3_apply(wide, jev, tau_jev)[:2],
        "C4_judge_d1": dm.c4_pred(wide, d1, frozen["c4"]["theta"]),
        "C4_judge_jev": dm.c4_pred(wide, jev, theta_jev),
    }.items():
        rows.append(row(wide, name, pred, latency))
        preds[name] = pred
    r = cacic.cascade_routes(wide)
    pred, latency, _ = cacic.apply_routes(wide, r, probe_logit=True)
    rows.append(row(wide, "CascadeLogitDomURLs_t0.80", pred, latency))
    preds["CascadeLogitDomURLs_t0.80"] = pred
    return rows, preds


def cmd_dev() -> None:
    frozen = json.loads(FROZEN.read_text())
    train = cacic.load_split("router_train")
    sample = dm.load_dm(PRED / "router_train_dm_sample_jev_decision_predictions.csv", train, require_doubtful=False)
    keep = sample["p_wl"].notna() & train["dga_type"].isin(["WL", "NW"])
    is_wl = train.loc[keep, "dga_type"].eq("WL")
    taus = []
    for tau in GRID:
        says = sample.loc[keep, "p_wl"] >= tau
        tpr, tnr = float(says[is_wl].mean()), float((~says[~is_wl]).mean())
        taus.append({"tau": tau, "balanced_accuracy": (tpr + tnr) / 2, "wl_recall": tpr, "nw_specificity": tnr})
    taus = pd.DataFrame(taus)
    tau_jev = float(taus.sort_values(["balanced_accuracy", "tau"], ascending=[False, True]).iloc[0]["tau"])

    dev = cacic.load_split("gate1")
    jev = dm.load_dm(PRED / "gate1_doubtful_jev_decision_predictions.csv", dev)
    thetas = []
    for theta in GRID:
        pred, latency = dm.c4_pred(dev, jev, theta)
        thetas.append({"theta": theta, **row(dev, "C4_judge_jev", pred, latency)})
    thetas = pd.DataFrame(thetas)
    theta_jev = float(thetas.sort_values(["f1", "theta"], ascending=[False, True]).iloc[0]["theta"])

    rows, _ = systems(dev, "gate1_doubtful", tau_jev, theta_jev, frozen)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "posthoc_jev_dev.csv", index=False)
    taus.to_csv(OUT / "posthoc_jev_tau.csv", index=False)
    thetas.to_csv(OUT / "posthoc_jev_theta.csv", index=False)
    CFG.write_text(json.dumps({
        "status": "post-hoc, not preregistered; values fixed on development data before evaluating the test",
        "written_at": dt.datetime.now().isoformat(timespec="seconds"),
        "model": "typesafe/jev-1.14 via https://openrouter.ai/api/v1/systemone",
        "tau_jev": tau_jev, "tau_rule": "max balanced accuracy WL vs NW on the router_train sample; ties -> lower tau",
        "theta_jev": theta_jev, "theta_rule": "max development F1; ties -> lower theta",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"tau_jev = {tau_jev}  theta_jev = {theta_jev}")
    print(table.round(4).to_string(index=False))


def cmd_test() -> None:
    if not CFG.exists():
        raise SystemExit("run the dev step first: the Jev thresholds must be fixed before the test is read")
    cfg = json.loads(CFG.read_text())
    frozen = json.loads(FROZEN.read_text())
    test = cacic.load_split("final_test")
    rows, preds = systems(test, "final_test_doubtful", cfg["tau_jev"], cfg["theta_jev"], frozen)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "posthoc_jev_test.csv", index=False)
    y, fam = test["y_true"], test["family"]
    pairs = [("C3_type_cascade_jev", "C2_type_cascade_rule"), ("C3_type_cascade_jev", "C3_type_cascade_d1"),
             ("C4_judge_jev", "CascadeLogitDomURLs_t0.80"), ("C4_judge_jev", "C4_judge_d1")]
    boot = pd.DataFrame([{"a": a, "b": b, **paired_f1_bootstrap(y, fam, preds[a], preds[b], n_boot=1000, seed=7)} for a, b in pairs])
    boot.to_csv(OUT / "posthoc_jev_bootstrap.csv", index=False)
    lines = ["# Post-hoc: C3 and C4 with Jev (not preregistered)", "",
             f"tau_jev = {cfg['tau_jev']}, theta_jev = {cfg['theta_jev']} (fixed on development data at {cfg['written_at']}).", "",
             "| System | F1 | FPR | WL F1 | Latency (ms) |", "|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r['system']} | {r['f1']:.4f} | {r['fpr']:.4f} | {r['wl_f1']:.4f} | {r['latency_mean_ms']:.3f} |")
    lines += ["", "| A vs B | diff | 95% interval |", "|---|---|---|"]
    lines += [f"| {r['a']} vs {r['b']} | {r['diff']:+.4f} | [{r['low']:+.4f}, {r['high']:+.4f}] |" for _, r in boot.iterrows()]
    lines += ["", "Jev latency is an internet round trip from this machine and is not comparable with the T4 latencies."]
    (OUT / "posthoc_jev_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["dev", "test"])
    args = parser.parse_args()
    {"dev": cmd_dev, "test": cmd_test}[args.step]()


if __name__ == "__main__":
    main()
