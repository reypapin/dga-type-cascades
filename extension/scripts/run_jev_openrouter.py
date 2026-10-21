#!/usr/bin/env python3
"""Post-hoc run of Jev (TypeSafe, via OpenRouter) on the doubtful-domain datasets, in parallel.

Same typed questions, option order and state format as the d1-3B notebook
(notebooks/decision_models_colab_predictions.ipynb). Output follows the same pattern:
experiments/predictions/<run>_jev_decision_predictions.csv and <run>_jev_runtime_metadata.json.
Progress is checkpointed to <output>.partial.csv, so a rerun resumes where it stopped.
The OpenRouter key is read from opencode's auth file and is never printed or saved.

Usage: python3 extension/scripts/run_jev_openrouter.py RUN [RUN ...] [--workers N] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRED = ROOT / "experiments" / "predictions"
MODEL = "typesafe/jev-1.14"
ENDPOINT = "https://openrouter.ai/api/v1/systemone"
KEY = json.load(open(Path.home() / ".local/share/opencode/auth.json"))["openrouter"]["key"]

# Identical to the frozen d1-3B notebook: random/benign option first.
QUESTIONS = {
    "wl": {
        "type": "choice",
        "instructions": "Look only at the domain name. How is its main part built?",
        "criteria": {"a": "Random-looking letters or digits", "b": "Real English words joined together"},
    },
    "dga": {
        "type": "choice",
        "instructions": "Look only at the domain name. Who most likely created it?",
        "criteria": {"a": "A person or company, for a real website or service", "b": "Malware, using a domain generation algorithm"},
    },
}
WL_KEY = "b"
DGA_KEY = "b"
FIELDS = ["domain", "family", "dga_type", "split", "y_true", "model", "p_wl", "p_dga", "latency_ms", "cost_usd"]


def prob_of(answer: dict, key: str) -> float:
    if "probabilities" in answer:
        return float(answer["probabilities"][key])
    conf = float(answer["confidence"])
    return conf if answer["choice"] == key else 1.0 - conf


def ask(domain: str) -> tuple[float, float, float, float]:
    body = json.dumps({"model": MODEL, "state": f"Domain name: {domain}", "questions": QUESTIONS}).encode()
    for attempt in range(8):
        req = urllib.request.Request(ENDPOINT, data=body, method="POST",
                                     headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read())
            ms = (time.perf_counter() - t0) * 1000.0
            ans = data["answers"]
            cost = float(data.get("usage", {}).get("cost", 0.0) or 0.0)
            return prob_of(ans["wl"], WL_KEY), prob_of(ans["dga"], DGA_KEY), ms, cost
        except urllib.error.HTTPError as exc:
            if exc.code in (408, 429, 500, 502, 503, 504, 520, 522, 524, 529) and attempt < 7:
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
                continue
            raise RuntimeError(f"{domain}: HTTP {exc.code}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt < 7:
                time.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
                continue
            raise
    raise RuntimeError(f"{domain}: retries exhausted")


def run(run_name: str, workers: int, limit: int | None) -> None:
    out_csv = PRED / f"{run_name}_jev_decision_predictions.csv"
    meta_json = PRED / f"{run_name}_jev_runtime_metadata.json"
    if out_csv.exists():
        print(f"{out_csv.name} already exists; skipping", flush=True)
        return
    frame = pd.read_csv(DATA / f"{run_name}_domains.csv")
    frame["domain"] = frame["domain"].astype(str).str.strip().str.lower()
    if limit:
        frame = frame.head(limit)
    partial = out_csv.with_suffix(".partial.csv")
    lock = threading.Lock()
    for attempt in range(4):  # failed domains are retried in later passes
        done = set(pd.read_csv(partial)["domain"]) if partial.exists() else set()
        todo = frame[~frame["domain"].isin(done)]
        if todo.empty:
            break
        print(f"[{run_name}] pass {attempt + 1}: {len(frame)} domains, {len(done)} done, {len(todo)} to go, {workers} workers", flush=True)
        new_file = not partial.exists()
        started = time.time()
        with open(partial, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            rows = todo.to_dict("records")
            failures = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(ask, r["domain"]): r for r in rows}
                for i, fut in enumerate(as_completed(futures), 1):
                    r = futures[fut]
                    try:
                        p_wl, p_dga, ms, cost = fut.result()
                    except Exception as exc:  # retried in the next pass
                        failures += 1
                        print("failed:", str(exc)[:160], flush=True)
                        continue
                    with lock:
                        writer.writerow({**{k: r[k] for k in ("domain", "family", "dga_type", "split", "y_true")},
                                         "model": "jev", "p_wl": p_wl, "p_dga": p_dga, "latency_ms": ms, "cost_usd": cost})
                        if i % 500 == 0:
                            fh.flush()
                            rate = i / max(time.time() - started, 1e-9)
                            print(f"[{run_name}] {i}/{len(rows)} ({rate:.1f}/s)", flush=True)
        if failures:
            time.sleep(30)
    result = pd.read_csv(partial).drop_duplicates("domain")
    if len(result) < len(frame):
        raise SystemExit(f"[{run_name}] incomplete: {len(result)}/{len(frame)} rows after 4 passes; rerun to resume")
    order = {d: i for i, d in enumerate(frame["domain"])}
    result = result.sort_values("domain", key=lambda s: s.map(order))
    result[FIELDS[:-1]].to_csv(out_csv, index=False)
    meta = {
        "seed": None,
        "run_name": run_name,
        "decision_model": "jev",
        "model_revision": MODEL,
        "endpoint": ENDPOINT,
        "device": "OpenRouter API (latency is the round trip from this machine)",
        "parallel_workers": workers,
        "questions": QUESTIONS,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
        "num_input_domains": int(len(frame)),
        "num_output_rows": int(len(result)),
        "total_cost_usd": float(result["cost_usd"].sum()),
        "posthoc": True,
        "output_csv": str(out_csv.relative_to(ROOT)),
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    partial.unlink()
    print(f"[{run_name}] done: {len(result)} rows, cost US${meta['total_cost_usd']:.4f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    for run_name in args.runs:
        run(run_name, args.workers, args.limit)


if __name__ == "__main__":
    main()
