# Runs on the Colab VM: unpack one package and execute the decision-model notebook with nbconvert.
# Reads /content/dm_run.json -> {"run": "<run name>", "model": "d1"}.
import json
import os
import subprocess
import sys

cfg = json.load(open("/content/dm_run.json"))
run, model = cfg["run"], cfg["model"]
package = f"/content/decision_models_{run}_colab_content_root_package.zip"
subprocess.run(["unzip", "-o", "-q", package, "-d", "/content"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nbconvert", "ipykernel"], check=True)

env = dict(os.environ, DM_RUN_NAME=run, DM_MODEL=model)
notebook = "/content/New_Paper_MOE/notebooks/decision_models_colab_predictions.ipynb"
executed = f"/content/executed_{run}_{model}.ipynb"
proc = subprocess.run(
    [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
     "--ExecutePreprocessor.timeout=-1", "--output", executed, notebook],
    env=env, capture_output=True, text=True,
)
print(proc.stdout[-3000:])
print(proc.stderr[-6000:])
if proc.returncode != 0:
    raise SystemExit(f"nbconvert failed with code {proc.returncode}")
print("NOTEBOOK_OK", run, model)
