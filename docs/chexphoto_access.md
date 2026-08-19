# CheXphoto access — what is gated, what is not

Checked 2026-08-19. This corrects a standing assumption in this repo: E13 and
`docs/paper_draft.md` both describe CheXphoto as "gated behind a CheXpert data
use agreement, not obtainable in this environment", full stop. That is right
about the photographs and wrong about the dataset as a whole — **the half that
matters most for a head-to-head was never gated.**

## The split

| Component | Gated? | Where |
|---|---|---|
| **Synthetic transformations (code)** | **No — MIT License** | [`stanfordmlgroup/cheXphoto`](https://github.com/stanfordmlgroup/cheXphoto), `transforms/*.py` |
| **Natural photographs** (Nokia10k, iPhone1k, the 250-image VinBrain film subset) | Yes | Stanford, via Redivis |
| **Original CheXpert x-rays** the photos were taken of | Yes | Stanford AIMI |

CheXphoto is a dataset paper with no model of its own (`docs/phase1_background.md`,
full-read notes), and its *corruption model* was released as source code. That
code applies to any radiograph, so it can be run against the freely licensed
chest radiographs this project already uses — which is exactly what
`experiments/e14_chexphoto_headtohead.py` does, using the verified port in
`src/data/chexphoto_transforms.py` (reference vendored under
`third_party/chexphoto/`, checked by `tests/test_chexphoto_transforms.py`).

## Getting the gated half

The dataset moved: the competition page
(https://stanfordmlgroup.github.io/competitions/chexphoto/) now points at
Redivis, https://stanford.redivis.com/datasets/2qwg-aav2k243j — not the old
AIMI/CodaLab route this repo's notes assumed.

1. Create a Redivis account (Stanford SSO is not required for external
   researchers; any account can request access).
2. Open the dataset page and request access. Stanford datasets of this kind
   require electronically signing a **Stanford University Dataset Research Use
   Agreement** — non-commercial research use only, no redistribution of the
   data.
3. Stanford's own documentation says access requests are typically answered
   within **about one business day**. Treat that as the planning number, not a
   guarantee.
4. The Redivis API needs a token (`https://redivis.com/api/v1/...` returns
   `401 invalid_token` without one), so download either through the web UI or
   with an API token from the account that signed the agreement.

Nothing about this needs an IRB determination — it is public, de-identified,
already-released data under a use agreement, the same category as DENTEX (see
`docs/irb_determination_request.md` §A, which should cover it in the same
request if CheXphoto is going to be used).

## What the gated half would add

E14 already runs CheXphoto's corruption model against real chest radiographs.
Getting the photographs adds exactly one thing, and it is worth being precise
about it: **real optics**. Every image in every experiment in this repo,
including E14, is a synthetic corruption of a digital radiograph. CheXphoto's
Nokia10k/iPhone1k sets are actual photographs taken by actual cameras, and its
250-image VinBrain subset photographs **physical films on a lightbox** — the
one published corpus that matches this project's own capture scenario
(`docs/simulator_grounding.md`) rather than monitor photography.

That subset is small (250 images, validation-only) and is not enough to train
on. What it is enough for is the check nobody in this repo can currently run:
whether a reader and a calibrator built on synthetic corruption hold up on
photographs taken with a real camera of a real film. That is the same question
the real phone pilot (`docs/irb_determination_request.md` §B) is meant to
answer — and unlike the pilot, this version needs no new data collection, no
films, and no ethics determination. **If both routes are open, this is the
cheaper one, and it should be tried first.**

## What the ungated route already produced

E14 ran and is written up in `docs/experiments_results.md`: under CheXphoto's
corruption model the guarantee held with zero violations and the targeting
ordering reproduced, and calibration fitted under this project's corruption
model failed to transfer to CheXphoto's (false-conviction rate 0.62-0.80
against a 0.50 bound). So the gated photographs are no longer blocking the
generalisation claim -- they are blocking exactly one thing, real optics.

## Honest limits of the ungated route

- E14 uses CheXphoto's *code*, not its *images*. No result from it may be
  described as "on CheXphoto data" or "a CheXphoto benchmark number".
- The transforms are applied to Kermany et al.'s pediatric pneumonia chest
  radiographs (CC BY 4.0), not to CheXpert images, so nothing here is
  comparable to numbers on CheXphoto's leaderboard.
- CheXphoto's corruption model was built for photographing a *monitor*; moiré
  in particular is a screen artifact. This project's taxonomy was grounded on
  photographing *films*. E14 runs both models on the same images precisely so
  that difference shows up as data rather than as an argument.
