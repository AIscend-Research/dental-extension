# Phase 3 — Model size/latency/FLOPs benchmark, and training-time audit

Measured 2026-07-29 on the real HierarchicalDet architecture (Swin-L backbone
+ DiffusionDet heads, config `diffdet.custom.swinbase.nonpretrain.yaml`,
untrained/random-init weights, `MODEL.WEIGHTS=""` since the pretrained
backbone `.pkl` isn't downloaded). Environment: macOS arm64, CPU only (no
GPU/MPS used for these numbers, to get a stable single-core-class baseline).
See SETUP.md for the install recipe this was run against.

## Model size

- **281,872,140 parameters** (281.9M), all trainable.
- **~1075 MB** in fp32 (before any quantization/pruning).

This is the real number behind the proposal's "lightweight" claim, and it's
worth being direct about: **Swin-L is not lightweight.** 1GB+ of fp32
parameters is far outside typical mobile/edge budgets (usually tens of MB).
`src/models/detector.py`'s own design note flagged this risk before any
number existed ("Swin-L is not small -- if latency is too high on target
hardware, that finding itself is worth reporting and may push toward a
lighter backbone"). That's no longer a hypothetical — it's confirmed. Two
honest paths forward, not resolved here:
1. Report this size/latency honestly as a finding and reframe "lightweight"
   as relative to the clean-image, full-quadrant-enumeration-diagnosis
   HierarchicalDet baseline (which is presumably similarly large), not
   relative to mobile-deployment budgets.
2. Swap to a genuinely small backbone (e.g. a MobileNet/EfficientNet-family
   FPN, or Swin-T instead of Swin-L) before claiming "low-compute deployment."
   This is a real design decision, not something to default into silently.

## Latency (CPU, single image, batch=1)

- **Forward pass (inference)**: 800×800 input, mean **2.52s** (min 2.51s, max
  2.54s over 5 runs after 2 warmup runs). Output: 0 instances, correct for
  untrained weights (nothing survives score thresholding).
- This is CPU-only and single-threaded-class hardware; a real GPU (even a
  modest one like Kaggle's T4/P100) would be substantially faster. No GPU was
  available to measure this directly — **do not use 2.52s as a GPU estimate**,
  re-measure on Kaggle before quoting an inference-latency number in the paper.

## FLOPs

- **Backbone-only: 508.1 GFLOPs** at 800×800 input (via `fvcore`'s
  `FlopCountAnalysis` on `model.backbone`).
- Full-model FLOPs (backbone + DiffusionDet's iterative denoising head) were
  **not** measured: `fvcore`'s static tracer can't cleanly walk DiffusionDet's
  dynamic diffusion-sampling loop (the number of refinement steps and proposal
  handling are data/config-dependent, not a fixed graph). Treat 508.1 GFLOPs
  as a **lower bound** on the true per-image FLOPs, not the full number.

## Training-time audit (the explicit Phase 3 "audit before training" task)

Timed one real training step (forward + backward + optimizer.step(), SGD,
lr=1e-4) with a synthetic batch matching the config's actual settings: batch
size 2, 800×800 images, 5 random GT boxes per image with all three
hierarchical label fields (`gt_classes_1/2/3`, matching DENTEX's real
`category_id_1/2/3` schema).

- **Mean training-step time: 35.1s** (3 steps, after 1 warmup step; loss
  converged to sane-looking values across all 5 diffusion refinement steps'
  `loss_ce1/2/3`, `loss_bbox`, `loss_giou` terms — confirms the training path,
  not just inference, actually runs end to end).
- Config's `SOLVER.MAX_ITER = 40000`. At the measured CPU rate:
  **40000 × 35.1s ≈ 390 hours ≈ 16.3 days** -- obviously infeasible on CPU,
  confirming what SETUP.md already assumed (GPU is mandatory), but now with a
  real number instead of an assumption.
- **This CPU number is the only measured one here.** A rough extrapolation to
  Kaggle-class GPU hardware (T4/P100, typical 15-40x speedup for
  conv+attention-heavy models over a CPU baseline like this) would put a
  single iteration somewhere in the ballpark of **1-2.5s**, i.e. roughly
  **11-28 hours** for the full 40k iterations. **This GPU figure is an
  unverified extrapolation, not a measurement** -- there was no GPU available
  in this environment to confirm it. Re-run this exact timing loop on the
  actual Kaggle GPU before committing to a training schedule; SETUP.md's
  advice to checkpoint/resume across sessions stands regardless of which
  estimate turns out right.

## Reproducing this

The timing script is not checked into the repo (it was a one-off benchmark,
not a reusable tool) -- reconstruct it from this doc plus SETUP.md's install
recipe: build the model via `build_model(cfg)`, construct a batch of
`{"image", "height", "width", "instances"}` dicts (Instances needs `gt_boxes`,
`gt_classes_1`, `gt_classes_2`, `gt_classes_3` for the hierarchical loss to
compute), call `model.train()` then time `forward()` + `backward()` +
`optimizer.step()`. For the FLOPs number, `fvcore.nn.FlopCountAnalysis` on
`model.backbone` alone (not the full model).
