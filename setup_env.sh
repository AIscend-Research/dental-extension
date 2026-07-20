#!/usr/bin/env bash
# One-shot bootstrap for the STARTABLE part of the project (no detector).
# For the detector stack, follow SETUP.md instead -- it cannot be one-lined
# because it depends on your CUDA/torch versions.
set -euo pipefail

echo "[1/3] Installing core requirements (no GPU / detectron2 needed)..."
pip install -r requirements-core.txt

echo "[2/3] Cloning the HierarchicalDet baseline into external/..."
bash scripts/clone_baseline.sh

echo "[3/3] Running the pipeline smoke tests + demo..."
python -m tests.test_degradation
python -m tests.test_metrics
python demo_degradation.py

echo
echo "Ready. You can now work on:"
echo "  - src/data/degradation.py   (Phase 2, already runnable)"
echo "  - src/data/dentex.py        (Phase 2, after: python scripts/download_dentex.py)"
echo "  - src/eval/metrics.py       (Phase 4 metrics, already runnable)"
echo "  - src/models/*.py           (Phase 3 stubs, need the detector stack -- see SETUP.md)"
echo
echo "See TASKS.md for who picks up what."
