# Vendored CheXphoto transform reference

Unmodified copies of the synthetic-perturbation implementations from
[`stanfordmlgroup/cheXphoto`](https://github.com/stanfordmlgroup/cheXphoto)
(`transforms/*.py`, MIT License, see `LICENSE`), fetched 2026-08-19.

**Why they are here.** `src/data/chexphoto_transforms.py` is a numpy/OpenCV
port of these transforms, so that this project's corruption-model head-to-head
(E14) runs against CheXphoto's *actual* photographic corruption model rather
than a description of it. Vendoring the reference is what lets
`tests/test_chexphoto_transforms.py` check the port against the original
implementation numerically instead of by eye.

**Not imported by anything in `src/`.** Only the equivalence test touches this
directory, and it skips itself when the reference's own dependencies (Pillow,
SciPy) are absent — they are deliberately not in `requirements-core.txt`.

**The reference does not run as-is on current Pillow/NumPy.** It was written
against Pillow <10 and NumPy <1.24, and uses three APIs that have since been
removed: `Image.ANTIALIAS`, `np.float`, and `Image.PERSPECTIVE`/`np.matrix`
usage that still works but warns. The test applies those shims at import time
rather than editing these files, so what is on disk here stays a verbatim copy
of upstream.

Nothing in this directory is CheXphoto *data* — the dataset itself is gated
(see `docs/chexphoto_access.md`). These are the released transform sources.
