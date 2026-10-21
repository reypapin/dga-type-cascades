#!/usr/bin/env bash
# Usage: run_dm.sh SESSION RUN [RUN ...]
# One T4 session: for each run, upload its package, execute the notebook, and download the outputs
# into experiments/predictions/ right away (so a later lost VM does not lose finished runs).
set -u
S="$1"; shift
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PRED="$ROOT/experiments/predictions"
LOGS="$HERE/logs"
MODEL=d1
mkdir -p "$LOGS"

step() { echo "== $*"; "$@"; rc=$?; if [ $rc -ne 0 ]; then echo "FAILED rc=$rc: $*"; colab stop -s "$S"; exit $rc; fi; }

step colab new -s "$S" --gpu T4
step colab upload -s "$S" "$HERE/run_dm_notebook.py" /content/run_dm_notebook.py
for RUN in "$@"; do
  for f in "$PRED/${RUN}_${MODEL}_decision_predictions.csv" "$PRED/${RUN}_${MODEL}_runtime_metadata.json"; do
    if [ -e "$f" ]; then echo "refusing to overwrite $f"; colab stop -s "$S"; exit 3; fi
  done
  printf '{"run": "%s", "model": "%s"}\n' "$RUN" "$MODEL" > "$LOGS/dm_run_$RUN.json"
  step colab upload -s "$S" "$ROOT/decision_models_${RUN}_colab_content_root_package.zip" "/content/decision_models_${RUN}_colab_content_root_package.zip"
  step colab upload -s "$S" "$LOGS/dm_run_$RUN.json" /content/dm_run.json
  step colab exec -s "$S" -f "$HERE/run_dm_notebook.py" --timeout 3300
  step colab download -s "$S" "/content/New_Paper_MOE/experiments/predictions/${RUN}_${MODEL}_decision_predictions.csv" "$PRED/${RUN}_${MODEL}_decision_predictions.csv"
  step colab download -s "$S" "/content/New_Paper_MOE/experiments/predictions/${RUN}_${MODEL}_runtime_metadata.json" "$PRED/${RUN}_${MODEL}_runtime_metadata.json"
  step colab download -s "$S" "/content/executed_${RUN}_${MODEL}.ipynb" "$LOGS/executed_${RUN}_${MODEL}.ipynb"
  echo "RUN_DONE $RUN"
done
colab stop -s "$S"
echo "ALL_DONE"
