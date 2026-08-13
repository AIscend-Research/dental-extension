"""The benchmark specification: what a case is, and what a docket freezes.

A `Docket` is deliberately a *specification*, not a directory of images. It
fixes the case labels, the lesion difficulties, the clinic difficulty, the
capture budget, the burden of proof and the master seed -- and from those, the
per-case scene and operator are generated deterministically. Anyone with the
same docket id runs the same benchmark, on any machine, without downloading
anything.

That is a property the image-corruption benchmarks it descends from do not
have: ImageNet-C ships the corrupted images because its corruptions are
static. Here the corruptions are a *process* that responds to the policy under
test, so the benchmark cannot be a fixed set of files -- two policies issuing
different instructions genuinely see different photographs. Shipping the
process, seeded, is the only reproducible option, and it is also what makes
the benchmark a standing challenge rather than a leaderboard on a frozen tensor.

Case difficulty is drawn, not constant, because a nonzero share of cases must
be undecidable from any photograph. Without them, every escalation would be an
image-quality failure, escalation rate would collapse to a proxy for capture
quality, and a policy could look perfect by simply retaking forever. Some
teeth just need a dentist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.evidence.ladder import BurdenSpec
from src.models.diagnostic import Case


@dataclass(frozen=True)
class DocketCase:
    """One case on the docket: a tooth, plus the seed that generates its session."""

    case_id: int
    label: int
    difficulty: float
    seed: int
    # index into a real-image pool, used only by the real-image arm
    image_index: int | None = None


@dataclass
class Docket:
    """A frozen benchmark instance.

    Attributes:
        name: identifier printed in results tables.
        cases: the cases to adjudicate.
        budget: maximum captures allowed per case (K).
        burden: the mandated two-sided standard of proof.
        clinic_difficulty: in [0, 1]; shifts scenes and operators toward worse
            conditions. 0.2 is a district hospital, 0.8 a torch-lit outpost.
        seed: master seed; every per-case seed derives from it.
    """

    name: str
    cases: list[DocketCase]
    budget: int = 3
    burden: BurdenSpec = field(default_factory=BurdenSpec)
    clinic_difficulty: float = 0.5
    seed: int = 0

    @property
    def prevalence(self) -> float:
        return float(np.mean([c.label for c in self.cases])) if self.cases else float("nan")

    def __len__(self) -> int:
        return len(self.cases)


def make_docket(
    name: str,
    n_cases: int = 2000,
    prevalence: float = 0.35,
    budget: int = 3,
    burden: BurdenSpec | None = None,
    clinic_difficulty: float = 0.5,
    seed: int = 0,
    difficulty_alpha: float = 2.0,
    difficulty_beta: float = 4.0,
    n_images: int | None = None,
) -> Docket:
    """Generate a docket.

    Args:
        n_cases: how many teeth to adjudicate.
        prevalence: fraction with caries. 0.35 is in the range DENTEX's own
            diagnosis distribution implies for caries among annotated teeth;
            it is swept in the sensitivity experiment because prevalence
            drives how often each burden is even reachable.
        difficulty_alpha/beta: Beta parameters for lesion difficulty. The
            default (2, 4) puts most mass on resolvable lesions with a real
            tail of ones that no photograph settles.
        n_images: if set, each case is assigned an index into a real-image
            pool of this size (used by the DENTEX arm; ignored otherwise).
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n_cases) < prevalence).astype(int)
    difficulties = rng.beta(difficulty_alpha, difficulty_beta, n_cases)
    seeds = rng.integers(0, 2**31 - 1, n_cases)
    images = rng.integers(0, n_images, n_cases) if n_images else [None] * n_cases

    cases = [
        DocketCase(
            case_id=i,
            label=int(labels[i]),
            difficulty=float(difficulties[i]),
            seed=int(seeds[i]),
            image_index=(int(images[i]) if n_images else None),
        )
        for i in range(n_cases)
    ]
    return Docket(
        name=name,
        cases=cases,
        budget=budget,
        burden=burden or BurdenSpec(),
        clinic_difficulty=clinic_difficulty,
        seed=seed,
    )


def make_image_docket(
    name: str,
    pool_labels: list[int],
    n_cases: int = 2000,
    budget: int = 4,
    burden: BurdenSpec | None = None,
    clinic_difficulty: float = 0.5,
    seed: int = 0,
) -> Docket:
    """Docket over a fixed pool of real labelled images.

    Unlike `make_docket`, the label is not drawn -- it is read off whichever
    pooled image the case points at, so the case's ground truth and its
    photograph always agree. Prevalence is therefore whatever the image pool's
    prevalence is, and cannot be set independently.

    With a small pool each image recurs across many cases under different
    scenes and operators. That is a Monte Carlo over *capture conditions*, not
    over images: it tightens the estimate of how the capture process behaves
    and does nothing whatever for image diversity. Any interval computed from
    it understates uncertainty in the image dimension, and experiments using it
    are expected to say so.

    `difficulty` is set to 0.0 because lesion difficulty is a construct of the
    analytic surrogate; a real reader's difficulty is whatever the actual
    radiograph makes it, and is not something this docket gets to specify.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(pool_labels), n_cases)
    seeds = rng.integers(0, 2**31 - 1, n_cases)
    cases = [
        DocketCase(
            case_id=i,
            label=int(pool_labels[int(idx[i])]),
            difficulty=0.0,
            seed=int(seeds[i]),
            image_index=int(idx[i]),
        )
        for i in range(n_cases)
    ]
    return Docket(
        name=name,
        cases=cases,
        budget=budget,
        burden=burden or BurdenSpec(),
        clinic_difficulty=clinic_difficulty,
        seed=seed,
    )


def to_model_case(dc: DocketCase, payload: object = None) -> Case:
    """Adapt a DocketCase to the channel-facing Case type."""
    return Case(label=dc.label, difficulty=dc.difficulty, payload=payload)
