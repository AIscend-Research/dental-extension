# Phase 2 — data notes: annotation reliability, Kaggle usability, degradation-strategy scope

## Inter-rater label disagreement (DENTEX)

Checked 2026-07-29 against the actual DENTEX paper (Hamamci et al. 2023,
[arXiv:2305.19112](https://arxiv.org/abs/2305.19112)).

- **Annotation protocol**: each panoramic X-ray is labeled by a final-year
  dental student, then "verified and corrected" by one of three expert
  dentists (15+ years' experience). Two-stage review, not independent
  multi-annotator labeling.
- **No inter-annotator agreement statistic is reported anywhere in the paper**
  -- no Cohen's kappa, Fleiss' kappa, or similar. This isn't an oversight on
  our end; it simply wasn't measured/published.
- The paper itself acknowledges this as a limitation, quoted directly:
  *"despite the meticulousness of this approach in an attempt to minimize
  labeling bias, discrepancies may still exist due to variations in the
  expertise levels among students and dentists."*

**What this means for us**: we cannot compute or report a real inter-rater
agreement number for DENTEX -- it doesn't exist in the released data or the
source paper. The honest move is to **cite DENTEX's own acknowledged
limitation directly** in our limitations section (label noise from a
single-review-pass protocol, no measured agreement statistic), rather than
implying we checked and found it acceptable, or silently omitting the issue.
This is a real, citable limitation for Phase 5's writing, not something to
resolve ourselves -- we have no access to re-annotate or re-review DENTEX's
images.

## Kaggle usability (the Phase 2 "MAKE SURE IT IS USABLE ON KAGGLE COMPUTE" task)

- **Size**: DENTEX is ~11GB as downloaded (three zips + jsons). Kaggle's
  dataset limits (checked 2026-07-29) are 200GB per public or private dataset,
  well within range -- size is not a blocker.
- **Attachment method**: `scripts/download_dentex.py` uses
  `huggingface_hub.snapshot_download`, which needs internet access and HF
  auth. Kaggle notebooks either need internet explicitly enabled (limited,
  and the DENTEX HF repo is gated behind license acceptance, which complicates
  auth in a notebook environment) or -- the better path, and what SETUP.md
  already recommends -- **download once locally, then upload DENTEX as a
  Kaggle Dataset** (via `kaggle datasets create` or the web UI) and attach it
  to the training notebook. This also means training runs offline once
  attached, which is closer to the real deployment story anyway.
- **Practical steps** (not yet executed -- needs a Kaggle account/CLI action,
  not something this environment can do):
  1. `python scripts/download_dentex.py` locally (already done).
  2. Extract just the needed subfolders (`quadrant-enumeration-disease` from
     train/val, per `configs/default.yaml`'s documented paths) to keep the
     Kaggle Dataset lean rather than uploading the full ~11GB including the
     unlabelled pretraining images and other task splits this project doesn't
     use.
  3. `kaggle datasets create` (or web upload) with that trimmed folder.
  4. Attach the resulting Kaggle Dataset to the training notebook; point
     `configs/default.yaml` paths at wherever Kaggle mounts it
     (`/kaggle/input/<dataset-name>/...`).
- **Conclusion**: DENTEX is usable on Kaggle compute. No blocker found, but
  the actual upload/attach hasn't been done (needs a Kaggle account action).

## Degradation-strategy ablation: GAN-based phone-camera simulation (scoped, not built)

TASKS.md's Stream 1 lists three strategies to compare: hand-built OpenCV
(done), albumentations presets (done, see `src/data/degradation_albumentations.py`
and `docs/`), and "GAN-based 'phone camera' simulation." This third arm was
**not built** -- here's the honest scoping reasoning rather than silently
skipping it:

- A real GAN-based arm means either (a) training an image-to-image
  translation model (e.g. CycleGAN-style) to map clean scans to photographed-
  film style, which needs a paired or unpaired real-photo dataset to train
  against -- exactly the "real pilot set" TASKS.md flags as needing a human
  with a phone, which doesn't exist yet -- or (b) using an off-the-shelf
  pretrained style-transfer model not specifically trained for this domain,
  which would be a weaker, less-justified comparison than the two arms
  already built.
- Building (a) without real photos to train/validate against would produce a
  GAN that's degrading images in some learned style, but with no ground truth
  for whether that style matches real phone-shot artifacts any better than
  the hand-built or albumentations arms already do -- i.e. it can't actually
  be evaluated for realism without the same real pilot set that's already the
  known blocker.
- **Recommendation**: defer the GAN arm until after the real pilot set exists
  (needs a phone + printed films + IRB clearance per `docs/phase1_background.md`).
  At that point, the real photos serve double duty: validating the existing
  two synthetic arms *and* providing training/validation signal for a GAN
  arm, making it a much stronger comparison than building it blind now.
