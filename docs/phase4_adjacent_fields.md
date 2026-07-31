# Phase 4 — adjacent-field inspiration for improvements

The roadmap's explicit Phase 4 task: look at computational photography,
speech recognition, astronomy, and triage systems for ideas. Notes from a
2026-07-29 search pass -- read as inspiration/framing, not settled design
decisions.

## Computational photography (burst fusion)

Already the direct inspiration for `src/models/fusion.py` (Night Sight-style
multi-frame merge). Nothing new to add beyond what's already built, except:
image-stacking research (astronomy, below) reinforces one thing worth
double-checking empirically once real bursts exist -- **simple averaging is
a strong, hard-to-beat baseline**. Before trusting the learned
attention-weighted fusion over a plain average, benchmark against it; the
literature on both burst photography and astronomical stacking treats
uniform/weighted averaging as the default that learned methods have to beat,
not a strawman.

## Speech recognition: confidence scoring and rejection

Deepest structural analogy of the four. Classic ASR confidence work computes
accept/reject scores at multiple granularities -- phone, word, *and*
utterance level -- and combines them, rather than one scalar per utterance.
[Reference work](https://ieeexplore.ieee.org/document/940849) reports
rejecting ~90% of word-level recognition errors while only rejecting 33% of
correct words, using multi-level confidence features, not a single score.

**Applicable idea, not yet built**: this project's confidence head currently
predicts one usability score per image. The ASR analogy suggests a
**per-region or per-tooth confidence signal** (which specific tooth/area is
under-trusted, not just "this whole image is degraded") could let `decide()`
be more precise -- e.g. "predict on teeth 11-16, refer only the glare-obscured
region near tooth 27" instead of an all-or-nothing per-image decision. This is
a real extension worth scoping for a v2, not something to build speculatively
now without a trained detector to attach it to.

## Astronomy: multi-exposure stacking

Standard result, worth citing directly: **signal-to-noise improves with the
square root of the number of exposures**, and simple pixel-wise weighted
averaging ("coaddition") is the default baseline multi-frame reconstruction
method, with learned/deconvolution methods judged against it. Directly
supports the point above -- benchmark `BurstFusion`'s learned attention
against plain averaging, don't assume the learned version wins.

Also relevant: astronomical stacking explicitly handles "varying levels of
blur from exposure to exposure" and treats outlier/corrupted frames specially
(sigma-clipping, rejecting frames that disagree with the stack). That's a
second, distinct notion of "cross-frame agreement" from the one
`BurstFusion.cross_frame_agreement()` currently implements (entropy of
attention weights) -- astronomy's version is closer to "does this frame's
*content* agree with what the others show," not "how peaked is the fusion
module's trust distribution." Worth trying as an alternative
`cross_frame_agreement` formula once real burst data exists, per the caveat
already in `fusion.py`'s docstring.

## Clinical triage: graded confidence tiers

Real-world triage/CDS systems don't use a binary accept/reject -- they use
**multi-tier escalation** (e.g. passive order defaults → interruptive alerts
→ hard-stop requiring override), and hold reliability standards per tier
(one cited example: cases at ≥90% model confidence should show ≥87% observed
correctness on review). Two takeaways:

1. This project's three-way `decide()` (predict / retake / refer_clinician)
   is already structurally aligned with this pattern -- worth saying so
   explicitly in the paper as a deliberate choice, not just a convenient
   3-way split.
2. The "≥90% confidence should mean ≥87% actual correctness" framing is a
   **calibration check**, not just a ranking check. `safe_deferral_rate()`
   and the risk-coverage curve in `src/eval/metrics.py` measure whether
   confidence *ranks* correct predictions above incorrect ones, but don't
   check whether the usability score's numeric value is calibrated (whether
   a 0.9 usability score really does correspond to ~90% accuracy). Once real
   model outputs exist, add a calibration check (e.g. a reliability diagram /
   expected calibration error) alongside the existing selective-prediction
   metrics -- a real gap in the current metric suite, not present in
   `src/eval/metrics.py` today.

## Summary of concrete follow-ups this pass surfaced

- Benchmark `BurstFusion` against plain frame averaging once real bursts
  exist -- don't assume the learned version wins (computational photography +
  astronomy both flag this).
- Consider a per-region confidence signal as a v2 extension, not now (speech
  recognition analogy).
- Try a content-agreement-based alternative to the current entropy-based
  `cross_frame_agreement` (astronomy analogy) -- already flagged as
  unvalidated in `fusion.py`'s docstring, this gives a concrete alternative
  to try.
- **Add a calibration metric (reliability diagram / ECE) to
  `src/eval/metrics.py`** once real model outputs exist -- a genuine gap the
  triage-tier analogy surfaced, since the existing metrics check ranking, not
  calibration.
