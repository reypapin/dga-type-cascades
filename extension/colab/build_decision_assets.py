#!/usr/bin/env python3
"""Build the decision-model Colab assets, following the pattern of the expert runs.

Writes (new files only, never overwrites):
- data/<run>_domains.csv for router_train_dm_sample, gate1_doubtful, final_test_doubtful
- notebooks/decision_models_colab_predictions.ipynb
- decision_models_<run>_colab_content_root_package.zip (one per run)
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from extension.lib import cacic  # noqa: E402

DATA = ROOT / "data"
NOTEBOOK = ROOT / "notebooks" / "decision_models_colab_predictions.ipynb"
COLUMNS = ["domain", "family", "dga_type", "split", "y_true"]
SAMPLE_PER_TYPE = 4000
SEED = 7

MODEL_REVISIONS = {
    "julia": "a85b127321d580d65176c89ced8273f305745d85",
    "laya": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    "d1": "051bcc464b01b9f92942b364d9586b0ef5912432",
}


def doubtful(wide: pd.DataFrame) -> pd.Series:
    return (wide["Logit"] - 0.5).abs() * 2 < cacic.CASCADE_THRESHOLD


def build_datasets() -> dict[str, Path]:
    out = {}
    train = cacic.load_split("router_train")
    parts = [train[train["dga_type"].eq(t)].sample(SAMPLE_PER_TYPE, random_state=SEED) for t in ("WL", "NW")]
    out["router_train_dm_sample"] = pd.concat(parts)[COLUMNS]
    for split in ("gate1", "final_test"):
        wide = cacic.load_split(split)
        out[f"{split}_doubtful"] = wide.loc[doubtful(wide), COLUMNS]
    paths = {}
    for run, frame in out.items():
        path = DATA / f"{run}_domains.csv"
        if path.exists():
            raise SystemExit(f"{path} already exists; refusing to overwrite")
        frame.to_csv(path, index=False)
        paths[run] = path
        print(run, len(frame), "rows", hashlib.sha256(path.read_bytes()).hexdigest())
    return paths


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def build_notebook() -> None:
    revisions = json.dumps(MODEL_REVISIONS, indent=4)
    cells = [
        md("""
# Experiment: Decision-model predictions on doubtful domains

Upload in Colab one of:

- `decision_models_router_train_dm_sample_colab_content_root_package.zip`
- `decision_models_gate1_doubtful_colab_content_root_package.zip`
- `decision_models_final_test_doubtful_colab_content_root_package.zip` (only after preregistro v2 is frozen)

The package must be uploaded to `/content` and contain a `New_Paper_MOE/` folder with:

- `New_Paper_MOE/data/<run>_domains.csv`
- `New_Paper_MOE/notebooks/decision_models_colab_predictions.ipynb`

Objective:

- Run one local decision model on the same Colab hardware (T4) as the experts.
- Ask two typed questions per domain: is the domain made of words (WL), and was it made by a DGA.
- Export one prediction table with identical domains, labels, probabilities, and latency fields.

Success criteria:

- Same domain set as the package.
- Valid `p_wl`, `p_dga`, and `latency_ms` columns.
- Results separable by `WL`, `NW`, and benign traffic.
"""),
        md("""
## 1. Runtime Notes

Use a T4 runtime, as in the expert runs. Run one decision model per session. Upload only the package zip; do not upload the full project folder.
"""),
        code("""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

import numpy as np
import pandas as pd

SEED = 7
random.seed(SEED)
np.random.seed(SEED)

runtime_info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
}
runtime_info
"""),
        code("""
# Optional Colab GPU metadata. Safe if nvidia-smi is unavailable.
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"""),
        md("""
## 2. Upload Package Check

After uploading the package zip to Colab, run the next cell. It extracts the package into `/content/New_Paper_MOE` and verifies that the dataset is present.
"""),
        code("""
# Set by the package; do not change for a frozen run.
RUN_NAME = os.environ.get("DM_RUN_NAME", "gate1_doubtful")
DECISION_MODEL = os.environ.get("DM_MODEL", "d1")

PACKAGE_ZIP = Path(f"/content/decision_models_{RUN_NAME}_colab_content_root_package.zip")
PROJECT_ROOT = Path("/content/New_Paper_MOE")

if PACKAGE_ZIP.exists():
    !unzip -o -q {str(PACKAGE_ZIP)} -d /content
elif not PROJECT_ROOT.exists():
    raise FileNotFoundError(f"Upload {PACKAGE_ZIP.name} to Colab, then rerun this cell.")

DATASET_CSV = PROJECT_ROOT / "data" / f"{RUN_NAME}_domains.csv"
if not DATASET_CSV.exists():
    raise FileNotFoundError(f"Missing {DATASET_CSV}")
print(DATASET_CSV, DATASET_CSV.stat().st_size, "bytes")
"""),
        code(f"""
# Install only the selected decision model, pinned to a fixed revision.
MODEL_REVISIONS = {revisions}

if DECISION_MODEL == "julia":
    from huggingface_hub import snapshot_download
    snapshot_download("SupersonicLabs/Julia-1", local_dir="/content/Julia-1", revision=MODEL_REVISIONS["julia"])
    %pip install -q -e /content/Julia-1
    sys.path.insert(0, "/content/Julia-1")
elif DECISION_MODEL == "laya":
    %pip install -q laya
elif DECISION_MODEL == "d1":
    %pip install -q "transformers>=5.14"
"""),
        md("""
## 3. Configuration

The typed questions are fixed before any run. Both use two options with neutral keys `a` and `b`. The stored probabilities are those of the WL option (`WL_KEY`) and of the DGA option (`DGA_KEY`).
"""),
        code("""
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "predictions"
OUTPUT_CSV = OUTPUT_DIR / f"{RUN_NAME}_{DECISION_MODEL}_decision_predictions.csv"
RUN_METADATA_JSON = OUTPUT_DIR / f"{RUN_NAME}_{DECISION_MODEL}_runtime_metadata.json"

# Option order chosen on the development smoke sample (400 gate1 domains): random/benign option first.
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
{"run_name": RUN_NAME, "decision_model": DECISION_MODEL, "dataset": str(DATASET_CSV), "output": str(OUTPUT_CSV)}
"""),
        md("""
## 4. Load Dataset

The dataset must contain `domain`, `family`, `dga_type`, `split`, and `y_true`.
"""),
        code("""
domains = pd.read_csv(DATASET_CSV)
required = {"domain", "family", "dga_type", "split", "y_true"}
missing = required - set(domains.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")
domains["domain"] = domains["domain"].astype(str).str.strip().str.lower()
display(domains.groupby(["dga_type"]).size().rename("domains").reset_index())
domains.head()
"""),
        md("""
## 5. Decision-Model Adapter

Every adapter answers both questions for one domain in a single call.
"""),
        code("""
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype_used = "default"


def prob_of(answer: dict, key: str = "a") -> float:
    if "probabilities" in answer:
        return float(answer["probabilities"][key])
    conf = float(answer["confidence"])
    return conf if answer["choice"] == key else 1.0 - conf


def state_of(domain: str) -> str:
    return f"Domain name: {domain}"


if DECISION_MODEL == "julia":
    from julia import load_model

    engine = load_model("/content/Julia-1", device=device, strict_encoding=True, max_length=8192, head_length=512)

    def ask(domain: str) -> dict:
        return engine.predict(state=state_of(domain), questions=QUESTIONS)["answers"]
elif DECISION_MODEL == "laya":
    import laya

    try:
        agent = laya.load("convaiinnovations/laya", device=device)
    except TypeError:
        agent = laya.load("convaiinnovations/laya")

    def ask(domain: str) -> dict:
        return agent.predict(state_of(domain), QUESTIONS)["answers"]
elif DECISION_MODEL == "d1":
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "LiquidAI/d1-3B", trust_remote_code=True, dtype=torch.float16, revision=MODEL_REVISIONS["d1"]
    ).to(device).eval()
    dtype_used = "float16"

    def ask(domain: str) -> dict:
        return model.system_one(state_of(domain), QUESTIONS)["answers"]
else:
    raise ValueError(f"Unknown decision model {DECISION_MODEL}")

ask("example.com")
"""),
        md("""
## 6. Inference Helpers

Each domain is one call that answers both questions. The time of that call is the domain's latency. Two warm-up rounds run first, as in the expert runs.
"""),
        code("""
def run_decision_model(frame: pd.DataFrame) -> pd.DataFrame:
    domains_list = frame["domain"].tolist()
    for _ in range(2):
        for domain in domains_list[:8]:
            ask(domain)
    p_wl = np.empty(len(domains_list), dtype=float)
    p_dga = np.empty(len(domains_list), dtype=float)
    latency_ms = np.empty(len(domains_list), dtype=float)
    for i, domain in enumerate(domains_list):
        t0 = time.perf_counter()
        answers = ask(domain)
        latency_ms[i] = (time.perf_counter() - t0) * 1000.0
        p_wl[i] = prob_of(answers["wl"], WL_KEY)
        p_dga[i] = prob_of(answers["dga"], DGA_KEY)
        if i % 5000 == 0:
            print(i, "of", len(domains_list), flush=True)
    out = frame.copy()
    out["model"] = DECISION_MODEL
    out["p_wl"] = p_wl
    out["p_dga"] = p_dga
    out["latency_ms"] = latency_ms
    return out
"""),
        md("""
## 7. Run Decision Model
"""),
        code("""
predictions = run_decision_model(domains)
display(predictions.groupby("dga_type")[["p_wl", "p_dga", "latency_ms"]].mean())
"""),
        md("""
## 8. Sanity Checks

These checks must pass before using the file.
"""),
        code("""
if len(predictions) != len(domains):
    raise ValueError(f"Expected {len(domains)} rows, got {len(predictions)}")
if predictions.duplicated(subset=["domain", "family", "split"]).any():
    raise ValueError("Duplicated domain/family/split rows found")
for col in ("p_wl", "p_dga"):
    if predictions[col].isna().any() or not predictions[col].between(0, 1).all():
        raise ValueError(f"{col} must be a probability")
if (predictions["latency_ms"] < 0).any():
    raise ValueError("Negative latency found")
"""),
        md("""
## 9. Export

Save the prediction table and metadata. Bring these files back into the repository under `experiments/predictions/`.
"""),
        code("""
columns = ["domain", "family", "dga_type", "split", "y_true", "model", "p_wl", "p_dga", "latency_ms"]
predictions[columns].to_csv(OUTPUT_CSV, index=False)

metadata = {
    "seed": SEED,
    "run_name": RUN_NAME,
    "decision_model": DECISION_MODEL,
    "model_revision": MODEL_REVISIONS[DECISION_MODEL],
    "device": device,
    "dtype": dtype_used,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "questions": QUESTIONS,
    "runtime": runtime_info,
    "torch": torch.__version__,
    "num_input_domains": int(len(domains)),
    "num_output_rows": int(len(predictions)),
    "output_csv": str(OUTPUT_CSV),
}
RUN_METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
str(OUTPUT_CSV), str(RUN_METADATA_JSON)
"""),
        md("""
## 10. Next Steps

After exporting from Colab:

- download `<run>_<model>_decision_predictions.csv` and `<run>_<model>_runtime_metadata.json`;
- place both files in `experiments/predictions/` in this repository;
- for `gate1_doubtful` and `router_train_dm_sample`, run `python3 extension/scripts/dev_c3c4.py` locally;
- run `final_test_doubtful` only after `extension/protocol/PREREGISTRO_V2.md` is frozen.
"""),
    ]
    nb = {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print("notebook:", NOTEBOOK)


def build_packages(paths: dict[str, Path]) -> None:
    for run, path in paths.items():
        zpath = ROOT / f"decision_models_{run}_colab_content_root_package.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(path, f"New_Paper_MOE/data/{path.name}")
            z.write(NOTEBOOK, f"New_Paper_MOE/notebooks/{NOTEBOOK.name}")
        print(zpath.name, hashlib.sha256(zpath.read_bytes()).hexdigest())


def main() -> None:
    paths = build_datasets()
    build_notebook()
    build_packages(paths)


if __name__ == "__main__":
    main()
