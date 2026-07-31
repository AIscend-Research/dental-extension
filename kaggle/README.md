# Kaggle notebooks

Four notebooks, run in order. Each one's code was verified locally (macOS
arm64, CPU/MPS, no GPU — see `SETUP.md`) against the actual repo, actual
DENTEX data, and actual pretrained weights before being written here — not
guessed. What that verification did and didn't cover is stated explicitly in
each notebook; the honest gaps are called out rather than papered over.

## Order

1. **`00_setup_and_sanity_check.ipynb`** — installs the detector stack
   (detectron2 + HierarchicalDet), downloads and correctly converts the
   backbone weights, confirms a real forward pass and training step on 2 real
   DENTEX images. Run this first on any new Kaggle notebook/session; if
   something's going to break, it breaks here, cheaply, not 30 minutes into
   the real training run.
2. **`01_train_baseline_detector.ipynb`** — the real training run (40k
   iterations, per the config). Needs multiple Kaggle sessions; has
   checkpoint/resume built in and tested (confirmed: killing training and
   resuming picks up from the last checkpoint's iteration, not from 0).
3. **`02_train_confidence_head.ipynb`** — trains the confidence/quality head
   standalone, without waiting on 01. Doesn't need detectron2 at all. Can run
   in parallel with 01.
4. **`03_evaluate.ipynb`** — real detection metrics (mAP, per-class F1)
   against a checkpoint from 01. The confidence/deferral section is an
   explicit **template**, not working code — see the notebook for why (it
   needs `src/models/detector.py`'s caries-only wrapper and a confidence head
   retrained against the real backbone, neither of which exist yet).

## Real bugs found and fixed while building these (read before debugging)

Everything below was confirmed by actually running the code, not inferred:

- **The config's backbone weights filename is wrong.**
  `diffdet.custom.swinbase.nonpretrain.yaml` points at
  `swin_base_patch4_window7_224_22k.pkl` but sets `MODEL.SWIN.SIZE: L-22k`
  (Large, not Base). Loading the Base weights "succeeds" with zero errors
  (detectron2's checkpointer only warns per-tensor) while silently leaving
  ~90% of the backbone randomly initialized. Fixed: download Microsoft's
  actual Swin-Large-22k checkpoint and convert it — see notebook 00, step 8,
  or `SETUP.md`.
- **`hierarchialdet.dataset_mapper.DiffusionDetDatasetMapper` is broken for
  anyone but the original author** — it unconditionally opens two hardcoded
  personal file paths in its constructor and crashes with `FileNotFoundError`
  otherwise. Notebook 01 defines its own `CariesDatasetMapper` instead, which
  trains directly from ground-truth boxes (confirmed: the model's own
  `prepare_inferred_boxes`/`prepare_targets` gracefully skip the
  pretrained-box curriculum when none is given).
- **`cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"`** (the config
  default) is invalid in the detectron2 version `pip install
  git+https://github.com/facebookresearch/detectron2.git` currently
  installs — raises `ValueError`. Fixed by overriding to `"norm"`.
- **`DataLoader` workers (`NUM_WORKERS > 0`) crashed in development** — likely
  a macOS spawn-vs-Linux-fork multiprocessing difference, not necessarily a
  Kaggle problem, but unverified on Kaggle. Shipped with `NUM_WORKERS=0`
  (confirmed working); try increasing it on Kaggle if you want the speed and
  watch for the same crash before trusting it.
- **`DiffusionDet.forward()` defaults to `k=0` at inference**, which only
  returns `pred_classes_1` (quadrant-level). For diagnosis/caries predictions,
  you need `model(batch, k=2)`, which returns `pred_classes_3`. Using the
  default silently gives you the wrong task's output instead of erroring.
- **detectron2's own `get_detection_dataset_dicts()` needs a flat
  `"category_id"` field** for its internal class-histogram logging, which
  DENTEX's hierarchical `category_id_1/2/3` schema doesn't have — raises
  `KeyError` otherwise. `src/data/dentex.py:to_detectron2_dicts()` includes a
  `category_id` alias (pointing at the diagnosis task) purely for this.

## What's genuinely still unverified

These notebooks were built and tested on **CPU** (no GPU available in
development). Real GPU-specific things to watch for on first run:
- Actual per-iteration training throughput (the ~35s/iteration CPU number in
  `docs/phase3_model_benchmarks.md` is not a GPU estimate — measure it fresh,
  notebook 01 step 5 does this for you before committing to the full run).
- Whether `NUM_WORKERS > 0` actually crashes on Kaggle's Linux/fork
  environment the way it did on macOS/spawn here (see above) — if it doesn't,
  turning it on is free throughput.
- Kaggle-specific paths/quotas (GPU session length, weekly quota) — these
  change over time; check Kaggle's current docs rather than trusting a
  number here.

If you hit something not on this list, please add it here (or to
`SETUP.md`) so the next person doesn't re-derive it.
