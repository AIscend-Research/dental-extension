# Phase 3, roadmap item 22 — training the confidence head (standalone)

"Design + train confidence/quality head using degradation labels as weak
supervision" was the one Phase 3 item that turned out **not** to require the
full detector training run to make progress on. Here's why, and what was
actually done and found.

## Why this didn't need to wait for the GPU/Kaggle run

`ConfidenceHead` (`src/models/confidence_head.py`) is designed to sit on the
real detector's image-encoder features (FPN p5, 256 channels) -- but that
backbone only produces *meaningful* features once it's trained on real caries
data, which needs Phase 3's GPU run (not done). Training the head against an
*untrained* Swin-L backbone would mean learning from noise.

Instead: `scripts/train_confidence_head_standalone.py` trains the same
`ConfidenceHead`, unmodified, against a small dedicated CNN trunk (`TinyTrunk`,
4 conv blocks, 3→32→64→128→256 channels) built just for this script. This
validates the thing that was actually uncertain -- can this architecture learn
to predict degradation type/severity from an image at all -- using real
DENTEX images degraded synthetically via the already-tested
`src/data/degradation.py`, without needing the caries detector to exist first.
**When the real detector is eventually trained, retrain `ConfidenceHead`
against its real FPN p5 features instead -- this trunk is a stand-in, not a
replacement.**

## Setup

- 200 real DENTEX training images, 50 held-out validation images (from
  `patient_level_split`'s train/val partition -- image-level, per the known
  DENTEX limitation).
- Each image → 1 clean example (all-zero severity label) + 3 synthetically
  degraded variants (`apply_degradations`, random severities/types) = 800
  train examples, 200 val examples.
- Usability target: `1 - max(severity)`, the "cheap start" the docstring
  itself proposed (not learned from detector correctness -- there's no
  trained detector to be correct/incorrect yet).
- 15 epochs, Adam (lr=1e-3), SmoothL1 loss on both severity vector and
  usability scalar, batch size 16, CPU only (fast -- the whole run is under
  a minute; TinyTrunk is far smaller than Swin-L).

## Results (full validation set, not cherry-picked)

- **Dominant-degradation accuracy: 69.3%** on 150 non-clean val examples
  (chance level with 5 degradation types: 20%). Real, well-above-chance
  signal that the architecture can learn this task.
- **Usability score Pearson correlation (pred vs. true): 0.593** — moderate
  positive correlation, real signal, far from perfect.
- **Clean images**: mean predicted usability 0.996 (true 1.0) — correctly
  identifies undegraded images with near-perfect confidence.
- **Degraded images**: mean predicted usability 0.696 (true mean 0.321) —
  **a real, systematic overconfidence problem**, not just noise: the model
  under-penalizes degraded images by a wide margin on average. This is
  exactly the overconfidence failure mode `expected_calibration_error()`
  (added in the Phase 4 work) is built to catch — worth running that metric
  against this model's outputs directly as a follow-up.
- **Validation loss was unstable across epochs** (bounced between 0.03 and
  0.15 rather than monotonically decreasing), while train loss decreased
  smoothly. Signs of overfitting on a small dataset (800 examples, no
  augmentation beyond the degradation itself) -- expected at this scale, and
  a real limitation to state plainly rather than paper over by picking a
  favorable epoch to report.

## Honest takeaways

1. **The core Phase 3 bet is validated at small scale**: a confidence head
   predicting *which* degradation dominates, not just a trust score, is
   learnable from synthetic weak supervision alone. That was the open
   question this task existed to answer, and the answer is yes.
2. **It's measurably overconfident on degraded images** -- if this pattern
   holds at full scale, `decide()`'s thresholds (`retake_below`,
   `refer_below`) would need to be tuned with that bias in mind, or the loss
   function/training data would need adjusting (e.g. more degraded examples
   relative to clean ones, since this run used a 1:3 clean:degraded ratio).
3. **This is a small-scale, standalone proof of concept**, not the final
   confidence head. It used a toy trunk, 200 training images (of 705
   available), and 15 epochs with no hyperparameter search. Treat the 69.3%/
   0.593 numbers as "this direction works," not as numbers for the paper's
   results table -- those need the real backbone.

## Reproducing

```bash
source .venv-detector/bin/activate   # needs torch + opencv, see SETUP.md
python scripts/train_confidence_head_standalone.py
```

No trained weights are checked into the repo (consistent with
`models_weights/` being gitignored elsewhere in this project) -- rerun the
script to regenerate; it takes well under a minute on CPU.
