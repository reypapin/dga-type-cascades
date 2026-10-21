#!/usr/bin/env python3
"""Train the final router on seen-family predictions and evaluate on Gate 1.

Required inputs:
- experiments/predictions/router_train_uniform_predictions.csv
- experiments/predictions/gate1_uniform_predictions.csv

The router is trained only on seen-family predictions. Gate 1 is used only for
evaluation after the router configuration has been fixed.
"""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "experiments" / "predictions"
RESULTS = ROOT / "experiments" / "results"

TRAIN_PREDICTIONS = PREDICTIONS / "router_train_uniform_predictions.csv"
EVAL_PREDICTIONS = PREDICTIONS / "gate1_uniform_predictions.csv"

MODELS = ["Logit", "ModernBERT", "DomURLs-BERT"]
CHEAP_TO_EXPENSIVE = ["Logit", "DomURLs-BERT", "ModernBERT"]
SEEDS = [7, 17, 29, 43, 61]
VOWELS = set("aeiou")
HEX_RE = re.compile(r"^[0-9a-f]+$")


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred.astype(int), labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "n": int(len(y_true)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "accuracy": float(accuracy),
    }


def add_latency(metrics: dict[str, float], latency: pd.Series) -> dict[str, float]:
    out = dict(metrics)
    out.update(
        {
            "latency_mean_ms": float(latency.mean()),
            "latency_p50_ms": float(latency.quantile(0.50)),
            "latency_p95_ms": float(latency.quantile(0.95)),
            "latency_p99_ms": float(latency.quantile(0.99)),
        }
    )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def max_run(text: str, predicate) -> int:
    best = 0
    current = 0
    for char in text:
        if predicate(char):
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def domain_features(domain: str) -> dict[str, float | str]:
    domain = str(domain).strip().lower()
    labels = [part for part in domain.split(".") if part]
    sld = labels[-2] if len(labels) >= 2 else labels[0] if labels else ""
    tld = labels[-1] if labels else ""
    joined = "".join(labels)
    alpha = [c for c in joined if c.isalpha()]
    digits = [c for c in joined if c.isdigit()]
    hyphens = joined.count("-")
    vowels = [c for c in alpha if c in VOWELS]
    consonants = [c for c in alpha if c not in VOWELS]
    label_lengths = [len(label) for label in labels] or [0]
    unique_chars = len(set(joined))
    length = len(joined)
    alpha_len = len(alpha)
    return {
        "length": length,
        "domain_chars": len(domain),
        "num_labels": len(labels),
        "max_label_len": max(label_lengths),
        "mean_label_len": float(np.mean(label_lengths)),
        "sld_len": len(sld),
        "entropy": shannon_entropy(joined),
        "sld_entropy": shannon_entropy(sld),
        "unique_ratio": unique_chars / length if length else 0.0,
        "digit_ratio": len(digits) / length if length else 0.0,
        "alpha_ratio": alpha_len / length if length else 0.0,
        "hyphen_ratio": hyphens / length if length else 0.0,
        "vowel_ratio": len(vowels) / alpha_len if alpha_len else 0.0,
        "consonant_ratio": len(consonants) / alpha_len if alpha_len else 0.0,
        "max_digit_run": max_run(joined, str.isdigit),
        "max_alpha_run": max_run(joined, str.isalpha),
        "max_consonant_run": max_run(joined, lambda c: c.isalpha() and c not in VOWELS),
        "has_hyphen": float("-" in joined),
        "starts_digit": float(bool(sld) and sld[0].isdigit()),
        "ends_digit": float(bool(sld) and sld[-1].isdigit()),
        "is_hex_sld": float(bool(sld) and bool(HEX_RE.match(sld))),
        "tld": tld,
    }


def pivot_predictions(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["domain", "family", "dga_type", "split", "y_true"]
    score = df.pivot(index=keys, columns="model", values="p_dga").reset_index()
    pred = df.pivot(index=keys, columns="model", values="y_pred").reset_index()
    latency = df.pivot(index=keys, columns="model", values="latency_ms").reset_index()
    pred = pred.rename(columns={m: f"{m}__pred" for m in MODELS})
    latency = latency.rename(columns={m: f"{m}__latency" for m in MODELS})
    return score.merge(pred, on=keys).merge(latency, on=keys)


def cheapest_correct_targets(wide: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    target = pd.Series("", index=wide.index, dtype=object)
    resolved = pd.Series(False, index=wide.index)
    for model in CHEAP_TO_EXPENSIVE:
        correct = wide[f"{model}__pred"].astype(int).eq(wide["y_true"].astype(int))
        take = (~resolved) & correct
        target.loc[take] = model
        resolved.loc[take] = True
    return target, resolved


def logit_aware_features(wide: pd.DataFrame) -> list[dict]:
    features = []
    for _, row in wide.iterrows():
        p = float(row["Logit"])
        item = domain_features(row["domain"])
        item.update(
            {
                "logit_p_dga": p,
                "logit_confidence": abs(p - 0.5) * 2,
                "logit_margin_to_threshold": p - 0.5,
                "logit_pred": int(row["Logit__pred"]),
            }
        )
        features.append(item)
    return features


def sample_weights(wide: pd.DataFrame) -> pd.Series:
    weights = pd.Series(1.0, index=wide.index)
    weights.loc[wide["dga_type"].eq("NW")] = 1.5
    weights.loc[wide["dga_type"].eq("WL")] = 5.0
    return weights


def make_router(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("vectorizer", DictVectorizer(sparse=True)),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "model",
                SGDClassifier(
                    loss="log_loss",
                    max_iter=200,
                    tol=1e-3,
                    alpha=1e-4,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def route_predictions(wide: pd.DataFrame, chosen_model: pd.Series, base_logit_cost: bool) -> tuple[pd.Series, pd.Series]:
    pred = pd.Series(0, index=chosen_model.index, dtype=int)
    latency = pd.Series(0.0, index=chosen_model.index, dtype=float)
    for model in MODELS:
        idx = chosen_model.index[chosen_model.eq(model)]
        pred.loc[idx] = wide.loc[idx, f"{model}__pred"].astype(int)
        latency.loc[idx] = wide.loc[idx, f"{model}__latency"].astype(float)
    if base_logit_cost:
        escalated = ~chosen_model.eq("Logit")
        latency.loc[escalated] += wide.loc[latency.index[escalated], "Logit__latency"].astype(float)
    return pred, latency


def evaluate_system(
    wide: pd.DataFrame,
    chosen_model: pd.Series,
    system: str,
    seed: int | str,
    base_logit_cost: bool = False,
) -> list[dict]:
    pred, latency = route_predictions(wide, chosen_model, base_logit_cost=base_logit_cost)
    rows = []
    groups = [("ALL", wide)]
    groups.extend((name, group) for name, group in wide.groupby("dga_type", sort=True))
    for group_name, group_df in groups:
        idx = group_df.index
        metrics = binary_metrics(wide.loc[idx, "y_true"], pred.loc[idx])
        metrics = add_latency(metrics, latency.loc[idx])
        routes = chosen_model.loc[idx]
        rows.append(
            {
                "system": system,
                "seed": seed,
                "group": group_name,
                "route_logit_share": float(routes.eq("Logit").mean()),
                "route_modernbert_share": float(routes.eq("ModernBERT").mean()),
                "route_domurls_share": float(routes.eq("DomURLs-BERT").mean()),
                **metrics,
            }
        )
    return rows


def cascade_logit_domurls(wide: pd.DataFrame, confidence_threshold: float = 0.80) -> pd.Series:
    logit_conf = (wide["Logit"] - 0.5).abs() * 2
    use_logit = logit_conf >= confidence_threshold
    route = pd.Series("DomURLs-BERT", index=wide.index, dtype=object)
    route.loc[use_logit] = "Logit"
    return route


def aggregate(rows: list[dict]) -> list[dict]:
    data = pd.DataFrame([row for row in rows if isinstance(row["seed"], int)])
    out = []
    for (system, group), group_df in data.groupby(["system", "group"], sort=False):
        row = {"system": system, "seed": "MEAN", "group": group}
        for col in group_df.columns:
            if col in {"system", "seed", "group"} or not pd.api.types.is_numeric_dtype(group_df[col]):
                continue
            row[col] = float(group_df[col].mean())
            row[f"{col}_std"] = float(group_df[col].std(ddof=0))
        out.append(row)
    return out


def adjusted_precision(recall: float, fpr: float, prevalence: float) -> float:
    numerator = recall * prevalence
    denominator = numerator + fpr * (1 - prevalence)
    return numerator / denominator if denominator else 0.0


def write_summary(rows: list[dict], train_rows: int, excluded_rows: int) -> None:
    mean_rows = [row for row in rows if row["seed"] == "MEAN" and row["group"] == "ALL"]
    router = next(row for row in mean_rows if row["system"] == "FinalWeightedLogitAwareRouter")
    cascade = next(row for row in mean_rows if row["system"] == "CascadeLogitDomURLs_t0.80")
    domurls = next(row for row in mean_rows if row["system"] == "AlwaysDomURLs-BERT")
    router_p100 = adjusted_precision(router["recall"], router["fpr"], 1 / 101)
    cascade_p100 = adjusted_precision(cascade["recall"], cascade["fpr"], 1 / 101)
    lines = [
        "# Final router evaluation",
        "",
        "This evaluation trains the router on seen-family predictions and evaluates it on Gate 1. Gate 1 remains a development benchmark, not the final untouched test.",
        "",
        "## Training Target",
        "",
        f"- Training rows with at least one correct expert: `{train_rows}`",
        f"- Training rows excluded because all experts failed: `{excluded_rows}`",
        "",
        "## Gate 1 Evaluation",
        "",
        f"- Final router: F1={router['f1']:.6f}, FPR={router['fpr']:.6f}, mean latency={router['latency_mean_ms']:.6f} ms, precision@1:100={router_p100:.6f}.",
        f"- Cascade Logit->DomURLs t=0.80: F1={cascade['f1']:.6f}, FPR={cascade['fpr']:.6f}, mean latency={cascade['latency_mean_ms']:.6f} ms, precision@1:100={cascade_p100:.6f}.",
        f"- Always DomURLs-BERT: F1={domurls['f1']:.6f}, FPR={domurls['fpr']:.6f}, mean latency={domurls['latency_mean_ms']:.6f} ms.",
        f"- Router gap vs cascade: {router['f1'] - cascade['f1']:+.6f} F1.",
        "",
        "## Decision",
        "",
    ]
    if router["f1"] >= cascade["f1"] and router["fpr"] <= cascade["fpr"]:
        lines.append("The trained router beats or matches the strong cascade on F1 and FPR; proceed to final-test protocol.")
    else:
        lines.append("The trained router does not dominate the strong cascade; use cascade as the main baseline and keep router as ablation unless final-test results change this.")
    (RESULTS / "final_router_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not TRAIN_PREDICTIONS.exists():
        raise SystemExit(
            f"Missing {TRAIN_PREDICTIONS}. Run the Colab notebook with RUN_NAME='router_train' first."
        )
    if not EVAL_PREDICTIONS.exists():
        raise SystemExit(f"Missing {EVAL_PREDICTIONS}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    print("Loading predictions", flush=True)
    train_wide = pivot_predictions(pd.read_csv(TRAIN_PREDICTIONS))
    eval_wide = pivot_predictions(pd.read_csv(EVAL_PREDICTIONS))

    print("Building targets and features", flush=True)
    target, resolved = cheapest_correct_targets(train_wide)
    train_filtered = train_wide.loc[resolved].copy()
    target_filtered = target.loc[resolved]
    train_features = logit_aware_features(train_filtered)
    eval_features = logit_aware_features(eval_wide)
    weights = sample_weights(train_filtered)

    rows = []
    prediction_rows = []
    for seed in SEEDS:
        print(f"Training seed {seed}", flush=True)
        model = make_router(seed)
        model.fit(train_features, target_filtered, model__sample_weight=weights.to_numpy())
        route = pd.Series(model.predict(eval_features), index=eval_wide.index, dtype=object)
        rows.extend(
            evaluate_system(
                eval_wide,
                route,
                "FinalWeightedLogitAwareRouter",
                seed,
                base_logit_cost=True,
            )
        )
        for idx, chosen in route.items():
            prediction_rows.append(
                {
                    "seed": seed,
                    "domain": eval_wide.at[idx, "domain"],
                    "family": eval_wide.at[idx, "family"],
                    "dga_type": eval_wide.at[idx, "dga_type"],
                    "y_true": int(eval_wide.at[idx, "y_true"]),
                    "chosen_model": chosen,
                }
            )

        for static_model in MODELS:
            static_route = pd.Series(static_model, index=eval_wide.index, dtype=object)
            rows.extend(evaluate_system(eval_wide, static_route, f"Always{static_model}", seed))
        cascade_route = cascade_logit_domurls(eval_wide, confidence_threshold=0.80)
        rows.extend(evaluate_system(eval_wide, cascade_route, "CascadeLogitDomURLs_t0.80", seed, base_logit_cost=True))

    rows.extend(aggregate(rows))
    write_csv(RESULTS / "final_router_gate1_metrics.csv", rows)
    write_csv(RESULTS / "final_router_gate1_predictions.csv", prediction_rows)
    write_summary(rows, int(resolved.sum()), int((~resolved).sum()))
    print(f"Wrote {RESULTS / 'final_router_gate1_metrics.csv'}")
    print(f"Wrote {RESULTS / 'final_router_gate1_predictions.csv'}")
    print(f"Wrote {RESULTS / 'final_router_summary.md'}")


if __name__ == "__main__":
    main()
