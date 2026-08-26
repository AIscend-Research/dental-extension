# Kaggle notebooks

Two paths through this directory:

- **Condensed path (recommended): `00` → `04` → `05`.** Two GPU sessions
  total (~11h + ~5h). `04_train_all.ipynb` trains both detector arms *and*
  the confidence head in one session, using a condensed recipe (10k
  iterations with a real LR schedule — the upstream config's 40k never
  anneals, its inherited `STEPS=(210000, 250000)` sit past the end of
  training — plus Swin stages 0–2 frozen). `05_evaluate_all.ipynb` produces
  the per-arm mAP/severity sweep, wires the real `DetectorChannel`, and runs
  the E6-protocol Docket leaderboard with the real detector. Rationale,
  budget math, and failure-recovery notes are in each notebook's header.
  Kaggle allows one active GPU session at a time, so merging costs no
  parallelism and saves the ~20-min setup per notebook.
- **Original granular path: `00` → `01` (twice) → `02` → `03` (twice).** The
  full 40k-iteration recipe, one concern per notebook, described below.
  ~30h/arm at the measured 2.7 s/iter — multi-session with checkpoint/resume.

Unlike 00–03, notebooks 04/05 have NOT been executed end to end (they need a
GPU session by definition); they reuse 00–03's run-verified cells verbatim
and state exactly which glue is new in their headers.

Four original notebooks, run in order. Each one's code was verified locally (macOS
arm64, CPU/MPS, no GPU — see `SETUP.md`) against the actual repo, actual
DENTEX data, and actual pretrained weights before being written here — not
guessed. What that verification did and didn't cover is stated explicitly in
each notebook; the honest gaps are called out rather than papered over.

## Order

1. **`00_setup_and_sanity_check.ipynb`** — installs the detector stack
   (detectron2 + HierarchicalDet), downloads and correctly converts the
   backbone weights, confirms a real forward pass on 2 real DENTEX images and
   a training step on 1 (2 OOMs on a T4 during backward). Run this first on
   any new Kaggle notebook/session; if
   something's going to break, it breaks here, cheaply, not 30 minutes into
   the real training run.
2. **`01_train_baseline_detector.ipynb`** — the real training run (40k
   iterations, per the config). Needs multiple Kaggle sessions; has
   checkpoint/resume built in and tested (confirmed: killing training and
   resuming picks up from the last checkpoint's iteration, not from 0).
   **Trains both arms**, selected by `TRAIN_ARM`: `baseline` (clean images,
   `degrade_prob=0.0`) and `robustness` (`0.7`, images passed through
   `src/data/degradation.py` with ground-truth boxes remapped to match). Each
   arm writes to its own `OUTPUT_DIR` so `resume=True` can't pick up the
   wrong arm's weights. Run both — the robustness claim is the difference
   between them, so one arm alone is not a result.
3. **`02_train_confidence_head.ipynb`** — trains the confidence/quality head
   standalone, without waiting on 01. Doesn't need detectron2 at all. Can run
   in parallel with 01.
4. **`03_evaluate.ipynb`** — real detection metrics (mAP, per-class F1)
   against a checkpoint from 01. Run once per arm (`EVAL_ARM`). The
   confidence/deferral section is an explicit **template**, not working code
   — see the notebook for why (it needs `src/models/detector.py`'s
   caries-only wrapper and a confidence head retrained against the real
   backbone, neither of which exist yet).

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
- **Degradation must remap ground-truth boxes, not just pixels.** `angle`
  rotates + perspective-warps the image, so feeding degraded images to the
  detector with the original boxes attached trains it on misaligned labels —
  no crash, no warning, just a worse model and a meaningless robustness
  comparison. `apply_degradations(..., boxes=...)` returns the boxes remapped
  through the same homography; notebook 01's mapper uses that path and drops
  boxes warped out of frame together with their class labels.
- **detectron2's default checkpointing fills Kaggle's 20 GB disk and kills
  the run** — confirmed on a real T4 ×2 session (failed after 8197s with
  "Your notebook tried to use more disk space than is available", 20.94 GB
  written). `DefaultTrainer`'s `PeriodicCheckpointer` has no retention cap,
  and each checkpoint here is ~3.2 GB (282M params fp32 + AdamW optimizer
  state), so at `CHECKPOINT_PERIOD=500` the `/kaggle/working` cap is hit
  around iteration ~3000 — long before the disk-cheap part of a 40k-iteration
  run. Fixed in notebook 01's `build_hooks`: `max_to_keep=2` on the
  checkpointer hook. When resuming across sessions, copy only the newest
  checkpoint (the one `last_checkpoint` names), not the whole directory.
  Silver lining from that failed run: ~2.7s/iteration on the T4, i.e. ~30h
  for the full 40k — the first real GPU throughput number.
- **detectron2's own `get_detection_dataset_dicts()` needs a flat
  `"category_id"` field** for its internal class-histogram logging, which
  DENTEX's hierarchical `category_id_1/2/3` schema doesn't have — raises
  `KeyError` otherwise. `src/data/dentex.py:to_detectron2_dicts()` includes a
  `category_id` alias (pointing at the diagnosis task) purely for this.

## Kaggle-environment fixes (why the setup cells look the way they do)

All four notebooks share one bootstrap, `src/utils/kaggle_env.py` (tested in
`tests/test_kaggle_env.py`). It exists because these failures only happen in
a Kaggle session, so a local run never catches them:

- **Never `pip install -r requirements-core.txt` here.** It can upgrade numpy,
  and Kaggle's torch -- plus the detectron2 built against it -- is compiled
  for the numpy already in the image. The upgrade succeeds and then torch dies
  at import with "compiled using NumPy 1.x cannot be run in NumPy 2.x", which
  looks like a code bug. `install_deps()` installs only what's missing, behind
  a pip constraints file pinning numpy/torch/torchvision.
- **Never `pip install torch`.** Match the image's build; replacing it breaks
  the compiled ops.
- **The bootstrap is idempotent.** The old `!git clone` + `%cd` errored on any
  re-run and then left the notebook in the wrong directory with every relative
  path silently wrong.
- **`DATA_ROOT` is looked up, not hardcoded** (`find_dentex_root()`) -- the
  mount path depends on what you named the Kaggle Dataset.
- **The detectron2 build is not `-q`.** It compiles for ~10 minutes and a
  silent cell that long is indistinguishable from a hang.

## What's genuinely still unverified

Every notebook has now been executed end to end locally (macOS arm64, CPU, no
GPU) against real DENTEX data and the real converted Swin-Large weights --
including 01's training loop and 03's inference and metrics. What that run
could NOT cover, because it needs a Kaggle session:

- **The detectron2 build against Kaggle's torch+CUDA.** The local runs used an
  already-built detectron2. If anything fails on first run, this is the most
  likely step; the error surfaces at the `from detectron2 import _C` check in
  notebook 00 step 5.

Real GPU-specific things to watch for on first run:
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
