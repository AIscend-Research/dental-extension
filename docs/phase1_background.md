# Phase 1 — Background & Framing

Notes from a literature pass done 2026-07-29. This is a starting point, not a
substitute for a full related-work read before submission — every claim below
should be re-verified against the actual paper (not just search-result
summaries) before it goes in the manuscript.

**Update 2026-08-14: CheXphoto and SelectiveNet, the two load-bearing
citations for the delta claim, have now been read in full** (arXiv PDFs
fetched and read directly, not summarized from search results). See "Full
read: CheXphoto" and "Full read: SelectiveNet" below. The delta statement's
claims about both papers hold up under the full read; nothing needed to be
walked back, and a few details are now sharper than the search-pass version
had them. **The e-value literature underpinning the theory (not the delta
claim) has also now been read in full** — Vovk & Wang (2021) and Ramdas et
al. (2022), see `docs/theory_anytime_validity.md` §4's new grounding note.
The calibrator this project uses is confirmed admissible by Vovk & Wang's
own Proposition 2.1, and Ramdas et al.'s result is stronger than initially
assumed: nonnegative martingales are not merely *a* valid way to build
anytime-valid inference, they are *necessary* for any admissible
construction.

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

## Full read: CheXphoto (Phillips, Rajpurkar et al., ML4H 2020)

Read in full (all 9 pages, including references) rather than the abstract.
It is a **dataset/benchmark paper with no model of its own** -- confirms and
sharpens the search-pass summary:

- **Construction**: 10,507 CheXpert x-rays photographed two ways -- an
  automated rig (`Nokia10k`: tripod-mounted Nokia 6.1, a custom Android app
  synchronising shutter with an on-screen x-ray display across varied
  lighting/monitor/orientation conditions) and manual capture (`iPhone1k`: a
  physician hand-photographing a monitor with an iPhone 8) -- plus synthetic
  transformations of the same 10,507 images.
- **Synthetic transformations, exact parameters**: *digital* (contrast
  ±10-17% then brightness shifted the same way, both via PIL's
  `ImageEnhance`, plus small random noise) and *spatial* (glare matte: a
  circular highlight placed via a random 2-D multivariate normal mask; moiré:
  warped semi-transparent parallel lines simulating shutter/refresh-rate
  interference; tilt: corner-scaling perspective warp, factor 0-0.05). Blur,
  motion, rotation and translation are mentioned as present in the released
  code but not detailed with parameters in the paper text.
- **A validation-only subset photographs physical films** (not monitor
  displays): 200 photos of x-ray film for 200 patients, "on a light box in
  typical clinical lighting conditions" -- this is the closest CheXphoto gets
  to this project's exact scenario (a printed/physical medium on a lightbox,
  not a screen), and it is a small, held-out validation slice, not the main
  training resource.
- **No confidence, deferral, or reject mechanism of any kind.** The paper
  does not train a diagnostic model at all -- it releases the dataset and a
  CodaLab-hosted leaderboard for others' models. There is nothing resembling
  a usability score, a retake instruction, or a predict/refer decision.
- **No discussion of multiple photos of the same film**, sequential capture,
  or burst/fusion of any kind. Every image is one independent draw.
- **No formal statistical guarantee of any kind** -- it is a benchmark
  dataset, not a method with a claim to check.

**What this confirms for the delta statement**: "no confidence/deferral
mechanism, no multi-shot fusion" is accurate, not an assumption from a
search summary -- verified against the actual paper text. The physical-film
validation subset is worth citing specifically (rather than the dataset in
general) when motivating why this project targets photographed *films*
rather than photographed *screens*.

## Full read: SelectiveNet (Geifman & El-Yaniv, ICML 2019)

Read in full. Confirms the search-pass characterisation and adds detail
worth having exact for the related-work section:

- **Architecture**: a shared trunk feeding two heads -- the classifier
  `f(x)` and a sigmoid selection head `g(x) in [0,1]`, trained jointly.
- **Training objective**: classification loss on selected points plus a
  Lagrangian penalty pulling the empirical selection rate `E[g(x)]` toward a
  target coverage `phi_hat` -- coverage is a *tuned hyperparameter baked into
  training*, not a runtime knob, and not a statistical guarantee.
- **Single-shot, single-observation**: exactly one decision per input, no
  sequential refinement, no notion of asking for more evidence. This is the
  single sharpest point of contrast with this project's method, and it is
  now confirmed from the paper itself rather than inferred from the title:
  SelectiveNet has no mechanism that a "retake" could even attach to.
- **No formal statistical guarantee** on the realised error rate or
  coverage -- purely an empirical accuracy/coverage tradeoff, optimised, not
  bounded. This is the precise sense in which this project's e-value
  machinery is doing something SelectiveNet's architecture cannot: Ville's
  inequality gives an anytime-valid bound under adaptive retaking; SelectiveNet
  gives a trained operating point with no such guarantee attached, at a fixed
  coverage decided before training and not adaptable at inference.
- **Beats MC dropout and softmax-response thresholding** at matched coverage
  on CIFAR-10/ImageNet/MNIST -- this project's
  `confidence_threshold_selective` baseline (added to the Docket leaderboard,
  see `docs/experiments_results.md` E3) is closer to the *softmax-response*
  baseline SelectiveNet beats than to SelectiveNet itself, since it thresholds
  the raw score rather than training a joint selection head. That is an
  honest limitation of that baseline, not a claim to soften: a full
  SelectiveNet-style joint-trained selector remains a stronger non-evidential
  baseline than what is currently on the leaderboard, and is a legitimate
  follow-up if reviewers ask for a tighter comparison.
- **Stated limitations** (their own): requires retraining per target
  coverage, assumes the coverage target is known ahead of time, limited
  analysis under class imbalance.

**What this confirms for the delta statement**: "no formal statistical
guarantee" is accurate and is now the load-bearing contrast to lead with in
the related-work section, not "no multi-shot mechanism" (true, but a weaker
point than "no guarantee under adaptive use").

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

**A determination request is now drafted and ready to send — see
`docs/irb_determination_request.md`.** It separates the three questions below
into three explicit asks, assembles the facts each one needs, and says what a
sufficient written answer looks like. It is *unsent*: nothing in this repo can
produce an institutional determination, only the request for one. The notes
below remain this project's own reading, not a determination.

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
| **ML4H 2026** (Findings track) | Sep 10, 2026, 11:59 PM AoE -- **confirmed 2026-08-14 against the live CFP** | 4 pages at submission, appendices free | Best fit — health-equity framing, low-resource deployment stories are exactly its wheelhouse, and CheXphoto itself was an ML4H paper. |
| ML4H 2026 (Proceedings track) | Sep 10, 2026, 11:59 PM AoE -- confirmed | 8 pages at submission (+1 camera-ready) | Good fit if the ablations are further along by then; more competitive/reviewed than Findings. |
| GenAI4Health @ NeurIPS 2026 | Sep 5, 2026 | up to 9 (research) / 5 (position) | Plausible but this project isn't generative-AI-centric — would need to frame around "trustworthy/safety benchmarks for medical AI," a stretch fit. |
| MICCAI 2026 workshops | Depends on individual workshop (main conf: Oct 4 & 8, Strasbourg) | TBD per workshop | Workshop proposals aren't finalized until March 2026; would need to check the specific accepted-workshops list closer to the date for one matching this topic (e.g. a dental-imaging or clinical-deployment workshop). |

**Recommendation**: target **ML4H 2026, Findings track, Sep 10, 2026, 11:59 PM
AoE** — closest topical fit (CheXphoto precedent), most achievable page count
(4 pages) for a project at this stage, and the latest safe deadline given
where the roadmap actually is as of 2026-07-29. This is later than the
roadmap's original Aug 22 OpenReview-submission placeholder — a real,
favorable schedule finding, not a problem: it buys roughly three more weeks.
Page limit and deadline are now confirmed directly against the live CFP
(2026-08-14; see above) — this is a decision, not a placeholder recommendation
anymore.

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

## Still open

- Confirmation from your institution on the IRB question above. The request
  itself is drafted and ready to submit (`docs/irb_determination_request.md`,
  2026-08-19); what is still open is sending it and recording the answer.
  Required in writing for ML4H camera-ready, and a hard blocker *today* for
  the real phone pilot specifically.
- Both remaining "still open" items from the original search pass are now
  resolved (2026-08-14): CheXphoto and SelectiveNet have been read in full
  (see above), and ML4H 2026's page limits/format are confirmed directly
  from the current call for papers (see below) rather than inferred.

## ML4H 2026 call for papers, confirmed 2026-08-14

Checked directly against https://ml4h.ahli.cc/submit/call-for-papers/
(the 2026-07-29 search pass could not find the 2026-specific page yet):

- **Deadline: September 10, 2026, 11:59 PM AoE.** One deadline, no separate
  abstract deadline. Matches the recommendation already on record below.
- **Findings track: up to 4 pages at submission** (appendices permitted
  without penalty, excluding references) -- matches the roadmap's ~4-page
  historical estimate.
- **Proceedings track: up to 8 pages at submission**, +1 page allowed for
  camera-ready (excluding references/appendices).
- LaTeX template at the Overleaf link in the CFP; double-blind (proper
  anonymisation required, gross violations risk desk rejection); at least
  one author must commit to reviewing 3 papers; IRB status/exemption must be
  documented for camera-ready -- relevant to the IRB item still open above,
  since it will need to be a written determination by then, not a note in
  this file.
- Submission site: https://openreview.net/group?id=ML4H/2026/Symposium
