# Phase 1 — Background & Framing

Notes from a literature pass done 2026-07-29. This is a starting point, not a
substitute for a full related-work read before submission — every claim below
should be re-verified against the actual paper (not just search-result
summaries) before it goes in the manuscript.

## Related work found

**Baseline / dataset**
- Hamamci et al., *Diffusion-Based Hierarchical Multi-Label Object Detection
  to Analyze Panoramic Dental X-Rays* (HierarchicalDet), MICCAI 2023.
  [Paper](https://conferences.miccai.org/2023/papers/205-Paper2550.html) ·
  [arXiv](https://ar5iv.labs.arxiv.org/html/2303.06500) ·
  [Code](https://github.com/ibrahimethemhamamci/HierarchicalDet). Adapts
  DiffusionDet (detection-as-denoising) with hierarchical noisy-box
  manipulation across the quadrant → enumeration → diagnosis tasks; reports
  beating RetinaNet, Faster R-CNN, DETR, and plain DiffusionDet on DENTEX.
  This is the backbone this project strips down to caries-only. No
  robustness-to-degradation or confidence/deferral mechanism in the baseline.

**Dental caries detection (general, not confidence-aware)**
- CariesNet — multi-stage caries segmentation on panoramic X-rays, 3127
  lesions. [PubMed](https://pubmed.ncbi.nlm.nih.gov/35017793/)
- Asci et al. 2024 — U-Net caries segmentation across primary/mixed/permanent
  dentition, >6000 pediatric radiographs.
- *Clinical application of deep learning for enhanced multistage caries
  detection in panoramic radiographs*, Scientific Reports 2025.
  [Nature](https://www.nature.com/articles/s41598-025-16591-4) ·
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/41022932/)
- Geometry-biased transformer for caries detection, addressing panoramic
  geometric distortion.
  [Springer](https://link.springer.com/article/10.1007/s00521-026-11939-x)
- Systematic review/meta-analysis of Faster R-CNN and Mask R-CNN for dental
  caries diagnosis. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12985255/)
- None of these address image-capture robustness (all assume clean digital
  radiographs) or a reject/defer mechanism.

**Smartphone-photo robustness for radiographs — the closest prior work**
- **CheXphoto** (Phillips, Rajpurkar et al., ML4H @ NeurIPS 2020) —
  10,000+ smartphone photos + synthetic photographic transformations of chest
  X-rays, built explicitly around the "photograph an X-ray and send it over
  WhatsApp" deployment story.
  [arXiv](https://arxiv.org/abs/2007.06199) ·
  [Code](https://github.com/stanfordmlgroup/cheXphoto). **This is the single
  closest prior work in spirit** — same motivating scenario (screen glare,
  poor angle, messaging-app compression), same synthetic + real photo
  strategy, but chest X-ray classification, no confidence/deferral mechanism,
  no multi-shot fusion.
- Recalibration of deep learning models for abnormality detection in
  smartphone-captured chest radiographs, *npj Digital Medicine* 2021.
  [Nature](https://www.nature.com/articles/s41746-021-00393-9) — shows
  recalibration (not just robustness training) matters once you're on phone
  photos; relevant to why a confidence head can't just be transplanted
  unchanged from the clean-image setting.
- Smartphone-Captured Chest X-Ray Photographs dataset on PhysioNet.
  [PhysioNet](https://physionet.org/content/cxr-phone/1.0.0/)

**Selective prediction / reject option (general ML, some medical)**
- SelectiveNet (Geifman & El-Yaniv, 2019) — integrated reject option via a
  learned selection head. [arXiv](https://arxiv.org/pdf/1901.09192)
- Selective classification survey/foundational work.
  [arXiv](https://arxiv.org/pdf/1705.08500)
- *Tackling prediction uncertainty in machine learning for healthcare*,
  Nature Biomedical Engineering 2022 — general framing of confidence-based
  deferral to a human expert in clinical ML.
  [Nature](https://www.nature.com/articles/s41551-022-00988-x)
- Uncertainty-aware diabetic retinopathy detection with rejection.
  [arXiv](https://arxiv.org/html/2510.00029)
- DEGRE — ensemble gating for trust-aware rejection in medical diagnostics,
  AAAI 2026. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/42473)
- None of these predict *which artifact* is causing low confidence (all are
  scalar trust/reject scores) — this project's confidence head predicting
  degradation *type* (for an actionable "retake with less glare" message) is
  not something this search surfaced elsewhere.

**Point-of-care image quality feedback (adjacent, non-dental)**
- AI-based real-time image quality feedback for chest radiographs, shown to
  improve positioning and reduce avoidable exposure in clinical routine.
  [medRxiv](https://www.medrxiv.org/content/10.1101/2021.06.10.21258326.full.pdf)
- Quality-check-then-diagnose two-stage AI pipelines for point-of-care chest
  X-ray. Same idea as this project's usability-score gate, different modality
  and no multi-shot fusion.
- *A Systematic Review of Medical Image Quality Assessment*, J. Imaging 2025.
  [DOI](https://doi.org/10.3390/jimaging11040100)

**Burst/multi-frame fusion (computational photography, not yet applied here)**
- Burst Image Restoration and Enhancement (Dudhane et al., CVPR 2022) and
  Burstormer (2023) — the general burst-fusion architectures this project's
  optional fusion module (`src/models/fusion.py`) is modeled after.
  [CVPR paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Dudhane_Burst_Image_Restoration_and_Enhancement_CVPR_2022_paper.pdf) ·
  [Burstormer](https://arxiv.org/pdf/2304.01194)
- Medical-imaging multiframe super-resolution/fusion exists (MRI/CT/SPECT/PET
  registration-based fusion), but nothing found combining burst *photography*
  of a static film/print with a detection+confidence pipeline the way this
  project proposes.

## Delta statement (draft)

CheXphoto (Phillips et al., 2020) established the "photograph-and-message" 
robustness problem for chest X-rays and is the closest prior work; selective
prediction / reject-option methods (SelectiveNet and successors) address
knowing-when-to-abstain in general and some medical settings; point-of-care
image-quality-gating systems exist for chest radiography positioning; and
HierarchicalDet is a strong clean-image dental detector with no robustness or
deferral mechanism at all. No prior work found combines all three for dental
caries specifically: (1) robustness to smartphone capture artifacts on
panoramic dental X-rays, (2) a confidence head that predicts *which*
degradation dominates (enabling an actionable "retake — reduce glare" message
rather than a bare reject), and (3) an optional multi-shot burst-fusion path
that uses cross-photo agreement as an additional confidence signal, sized for
offline/low-compute deployment. That combination, in the dental/low-resource
setting specifically, is this project's delta.

**Caveat**: this was a search-engine-summary pass, not a full paper read.
Before submission, actually read CheXphoto and SelectiveNet in full (they're
the two load-bearing citations for the delta claim), and re-run this search
closer to the submission date in case something new landed.

## Ethics / IRB

- DENTEX itself is already public, de-identified, CC BY-NC-SA 4.0 licensed
  data (see `data/dentex/DENTEX/README.md` after download) — using it as-is
  for method development does not appear to trigger new human-subjects review
  beyond what MICCAI/DENTEX's original release already cleared. Confirm this
  interpretation with your institution rather than relying on this note.
- If the "small real pilot set" from TASKS.md Phase 2 (re-photographing
  printed X-rays/lightbox films with a phone) goes ahead, that likely DOES
  need an actual IRB/ethics determination — it's new data collection, even if
  the underlying films are already de-identified. Resolve this **before**
  collecting any real photos, not after — flagged as a hard blocker on that
  specific task, not on the rest of Phase 2/3/4's work with DENTEX itself.
- No clinician feedback (Phase 4) has been collected yet; if that involves
  identifiable clinician participation (e.g. a named survey), check whether
  your institution requires even lightweight IRB review for that.

## Candidate workshops (checked 2026-07-29)

| Venue | Deadline | Pages | Fit |
|---|---|---|---|
| **ML4H 2026** (Findings track) | Sep 10, 2026 AoE | ~4 pages (historical; reconfirm) | Best fit — health-equity framing, low-resource deployment stories are exactly its wheelhouse, and CheXphoto itself was an ML4H paper. |
| ML4H 2026 (Proceedings track) | Sep 10, 2026 AoE | ~8 pages (historical; reconfirm) | Good fit if the ablations are further along by then; more competitive/reviewed than Findings. |
| GenAI4Health @ NeurIPS 2026 | Sep 5, 2026 | up to 9 (research) / 5 (position) | Plausible but this project isn't generative-AI-centric — would need to frame around "trustworthy/safety benchmarks for medical AI," a stretch fit. |
| MICCAI 2026 workshops | Depends on individual workshop (main conf: Oct 4 & 8, Strasbourg) | TBD per workshop | Workshop proposals aren't finalized until March 2026; would need to check the specific accepted-workshops list closer to the date for one matching this topic (e.g. a dental-imaging or clinical-deployment workshop). |

**Recommendation**: target **ML4H 2026, Findings track, Sep 10, 2026 AoE** —
closest topical fit (CheXphoto precedent), most achievable page count for a
project at this stage, and the latest safe deadline given where the roadmap
actually is as of 2026-07-29. This is later than the roadmap's original
Aug 22 OpenReview-submission placeholder — a real, favorable schedule finding,
not a problem: it buys roughly three more weeks. **This is a recommendation,
not a decision** — confirm the exact page limit/format on ml4h.ahli.cc's
current call for papers before committing, since the site did not have the
2026-specific writing guidelines fully published as of this search.

## Baseline models to benchmark against

- **HierarchicalDet** (the adapted backbone itself, clean-image performance)
- **Plain DiffusionDet** and **Faster R-CNN** — both already reported against
  in the HierarchicalDet paper on DENTEX, reuse those numbers as a starting
  reference point for the clean-image column of the results table.
- **CheXphoto's robustness-training recipe** (train on synthetic + some real
  photographic transforms) — closest methodological baseline for "does
  robustness training alone fix this," to compare against the confidence-head
  addition being the actual delta.
- **A confidence baseline with no type prediction** — e.g. plain softmax/
  detection-score thresholding (no separate confidence head, no degradation-
  type prediction) — needed to isolate what the confidence head adds beyond
  "just don't trust low detection scores."

## Still open (not resolved by this search pass)

- Actual full read of CheXphoto and SelectiveNet (do this before writing the
  related-work section for real).
- Confirmation from your institution on the IRB question above.
- ML4H 2026's exact 2026 page limits/template (site was incomplete as of this
  search — reconfirm close to submission).
