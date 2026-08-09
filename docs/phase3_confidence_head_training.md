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
- **Model selection: best epoch by validation loss** (early stopping), not
  the last epoch. See the correction below for why this is not a cosmetic
  choice.

## Correction: the earlier "overconfidence" result was a reporting artifact

The first version of this writeup reported **whatever the final epoch
produced**. Validation loss on this task is genuinely unstable (it bounces
between roughly 0.015 and 0.18 while train loss falls smoothly), so that made
every headline number a coin flip on where epoch 15 happened to land.

It landed badly, and the conclusion drawn from it was wrong. The reported
"mean predicted usability 0.696 on degraded images vs. a true mean of 0.321
-- a real, systematic overconfidence problem" was **an artifact of an unlucky
final epoch, not a property of the model**. Selecting the best-val-loss epoch
instead, the same script on the same data shows the model is roughly
calibrated and if anything slightly *under*-confident on degraded images.

Both the script and `kaggle/02_train_confidence_head.ipynb` now keep the
best-by-val-loss weights and print which epoch was selected alongside what
the final epoch would have given. Numbers below are from the corrected runs.
Anything elsewhere still citing 69.3% / 0.593 / 0.696-vs-0.321 predates this
correction.

## Results (full validation set, best-val-loss epoch)

Two runs, both real, differing only in scale:

| | script (200 train / 50 val) | notebook 02 (495 train / 100 val) |
|---|---|---|
| selected epoch | 13/15 (val_loss 0.029) | 9/15 (val_loss 0.015) |
| dominant-degradation accuracy | **66.7%** (chance 20%) | **77.0%** (chance 20%) |
| usability Pearson correlation | **0.894** | **0.933** |
| mean usability, clean images | 0.848 (true 1.0) | 0.983 (true 1.0) |
| mean usability, degraded | 0.265 (true mean 0.321) | 0.345 (true mean 0.314) |

- **Dominant-degradation accuracy is well above chance in both runs**, and
  improves with more data (66.7% -> 77.0% going from 200 to 495 images) --
  the architecture can learn this task, and is not yet data-saturated.
- **Usability correlation is strong** (0.894 / 0.933), much better than the
  0.593 the last-epoch run reported.
- **Calibration is reasonable, in both directions and at both scales**: the
  larger run predicts 0.345 mean usability on degraded images against a true
  mean of 0.314 (mildly under-confident, i.e. slightly *conservative*, which
  is the safe direction for a clinical deferral gate), and 0.983 on clean
  images against 1.0. Run `expected_calibration_error()` against these
  outputs for the real number rather than eyeballing the means.
- **Validation loss is still unstable across epochs** (0.015 to 0.18), while
  train loss decreases smoothly. That instability is real -- small dataset
  (800/1980 examples), no augmentation beyond the degradation itself -- and
  is now handled by model selection rather than ignored. State it as a
  limitation; do not present the selected-epoch numbers as clean held-out
  results, since the epoch was chosen on the same validation set.

## Honest takeaways

1. **The core Phase 3 bet is validated at small scale**: a confidence head
   predicting *which* degradation dominates, not just a trust score, is
   learnable from synthetic weak supervision alone. That was the open
   question this task existed to answer, and the answer is yes.
2. **It is not systematically overconfident** -- the opposite of what the
   first version of this doc claimed. At the selected epoch it is mildly
   conservative on degraded images (0.345 predicted vs 0.314 true at the
   larger scale). `decide()`'s thresholds still need tuning against real
   detector correctness rather than this proxy target, but they do not need
   to compensate for a large overconfidence bias, because there isn't one.
3. **Report which epoch you selected, every time.** The instability is real
   and large enough to flip a qualitative conclusion. Any number from this
   pipeline is meaningless without saying whether it came from the best or
   the last epoch -- that is the actual methodological lesson here, and it
   applies to the detector runs too.
4. **This is a small-scale, standalone proof of concept**, not the final
   confidence head. It used a toy trunk, 200-495 training images (of 705
   available), and 15 epochs with no hyperparameter search. Treat these
   numbers as "this direction works," not as numbers for the paper's results
   table -- those need the real backbone.

## Reproducing

```bash
source .venv-detector/bin/activate   # needs torch + opencv, see SETUP.md
python scripts/train_confidence_head_standalone.py
```

No trained weights are checked into the repo (consistent with
`models_weights/` being gitignored elsewhere in this project) -- rerun the
script to regenerate; it takes well under a minute on CPU.
