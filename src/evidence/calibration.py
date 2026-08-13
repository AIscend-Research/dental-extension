"""Conformal p-values, stratified by predicted degradation.

The wealth process needs, at each shot, a statistic whose null distribution is
known. We get one the usual split-conformal way: hold out a labelled
calibration set, and score a new observation by its rank among calibration
scores. The rank of a fresh draw among n exchangeable draws is (super-)uniform,
which is exactly the input a p-to-e calibrator wants.

The subtlety, and the reason this file is not three lines long:

A retake loop *selects* its shots. It keeps photographing until the image looks
usable, so the shots a session ends on are systematically cleaner than the ones
it starts with, while the calibration set is built from first shots only (see
src/bench/runner.py). A fresh shot is therefore not exchangeable with the
calibration pool, which is exactly the assumption split-conformal rests on --
and E1 confirms the non-exchangeability is real and strongly significant.

Stratifying weakens what has to be assumed. Conditioning the calibration pool
on the *predicted* degradation stratum replaces "the capture sequence is
exchangeable" (false here by construction) with "within a degradation stratum,
this shot is exchangeable with calibration shots of the same stratum" -- which
the retake loop does not disturb, because the stratum is what it selects on.
That weaker assumption is the one docs/theory_anytime_validity.md uses.

What the experiments actually found, which is worth stating because it is not
what this design anticipated: the selection shift runs in the *conservative*
direction, so `MarginalCalibrator` does not violate the guarantee either. A
retake loop yields cleaner shots, a sound tooth in a clean shot scores low, and
comparing that score against a pool containing dirtier shots makes the p-value
larger, not smaller. Marginal calibration is therefore safe here and merely
slightly less powerful (E4, ablation 1). Stratification remains the defensible
default -- the conservative direction is a property of this capture process,
not a theorem, and a process whose retakes made images *worse* would flip it --
but the honest claim is "weaker assumptions and a little more power", not
"the only thing standing between you and invalid coverage".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Minimum calibration points a stratum needs before it is trusted on its own.
# Below this the conformal p-value is too coarse to be useful (its resolution
# is 1/(n+1)), so we fall back to the marginal pool and record that we did.
MIN_STRATUM_N = 20


def usability_stratum(
    predicted_usability: float,
    n_strata: int = 4,
    edges: np.ndarray | None = None,
) -> int:
    """Bucket a shot by how usable the confidence head thinks it is.

    With `edges` given, those fixed cut points are used; otherwise the bins
    are equal-width on [0, 1].

    Equal-width bins waste the stratification: predicted usability is
    concentrated in the middle of its range, so the extreme bins end up with
    too few calibration points to give a usable p-value resolution (a stratum
    with n points cannot produce a p-value below 1/(n+1), which caps the
    evidence any shot in it can ever supply). Quantile edges fitted once on
    the calibration set fix that and remain deployable: the edges are frozen
    constants after fitting, so scoring a single shot in the field still needs
    nothing but that shot.
    """
    if n_strata < 1:
        raise ValueError("n_strata must be >= 1")
    u = float(np.clip(predicted_usability, 0.0, 1.0))
    if edges is None:
        return int(min(n_strata - 1, int(u * n_strata)))
    return int(np.clip(np.searchsorted(edges, u, side="right"), 0, n_strata - 1))


def quantile_edges(usabilities: np.ndarray, n_strata: int) -> np.ndarray:
    """Interior cut points splitting `usabilities` into `n_strata` equal shares."""
    if n_strata < 2:
        return np.array([])
    qs = np.linspace(0.0, 1.0, n_strata + 1)[1:-1]
    return np.quantile(np.asarray(usabilities, dtype=np.float64), qs)


@dataclass
class _Pool:
    """Sorted calibration scores for one stratum, split by label."""

    healthy: np.ndarray  # scores of y=0 calibration cases
    diseased: np.ndarray  # scores of y=1 calibration cases


class Calibrator:
    """Base: turns a diagnosis score into two one-sided conformal p-values.

    Both nulls are tested, because a courtroom needs to be able to acquit as
    well as convict:

        p_convict   -- p-value for H0: the tooth is SOUND. Small when the
                       score is high relative to sound calibration cases.
        p_discharge -- p-value for H0: the tooth is DISEASED. Small when the
                       score is low relative to diseased calibration cases.

    Both use the standard conservative (1 + #{as or more extreme}) / (n + 1)
    form, which is valid for finite n rather than only asymptotically.
    """

    def __init__(self, n_strata: int = 4):
        self.n_strata = n_strata
        self._pools: dict[int, _Pool] = {}
        self._marginal: _Pool | None = None
        self.fallback_strata: set[int] = set()
        self.edges: np.ndarray | None = None

    # -- fitting ------------------------------------------------------------

    def stratum_of(self, predicted_usability: float) -> int:
        """Stratum for one shot, using the edges frozen at fit time.

        Every consumer goes through this rather than computing strata itself,
        so the calibration pools and the runtime lookup can never drift apart
        -- a mismatch there would silently compare a shot against the wrong
        null distribution, which is the kind of bug that produces confident
        wrong coverage rather than a crash.
        """
        return usability_stratum(predicted_usability, self.n_strata, self.edges)

    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        usabilities: np.ndarray,
    ) -> "Calibrator":
        """Build the calibration pools and freeze the stratum edges.

        Args:
            scores: (n,) diagnosis scores, higher = more suspicious.
            labels: (n,) 0 = sound, 1 = caries.
            usabilities: (n,) predicted usability per calibration shot. The
                stratum edges are set to this sample's quantiles and then
                frozen; MarginalCalibrator ignores the resulting strata.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        usabilities = np.asarray(usabilities, dtype=np.float64)
        if not (len(scores) == len(labels) == len(usabilities)):
            raise ValueError("scores, labels and usabilities must be the same length")

        self.edges = quantile_edges(usabilities, self.n_strata)
        strata = np.array([self.stratum_of(u) for u in usabilities], dtype=int)

        self._marginal = _Pool(
            healthy=np.sort(scores[labels == 0]),
            diseased=np.sort(scores[labels == 1]),
        )
        self._pools = {}
        self.fallback_strata = set()
        for z in range(self.n_strata):
            mask = strata == z
            pool = _Pool(
                healthy=np.sort(scores[mask & (labels == 0)]),
                diseased=np.sort(scores[mask & (labels == 1)]),
            )
            self._pools[z] = pool
            if min(len(pool.healthy), len(pool.diseased)) < MIN_STRATUM_N:
                self.fallback_strata.add(z)
        return self

    # -- lookup -------------------------------------------------------------

    def _pool_for(self, stratum: int) -> _Pool:
        raise NotImplementedError

    def p_values(self, score: float, stratum: int) -> tuple[float, float]:
        """(p_convict, p_discharge) for one shot.

        Returns 1.0 for a side whose calibration pool is empty -- the maximally
        conservative answer, i.e. "this shot provides no admissible evidence in
        that direction", never a spuriously small p-value.
        """
        pool = self._pool_for(stratum)
        s = float(score)

        n_h = len(pool.healthy)
        # sound cases scoring at least as suspicious as this one
        n_ge = int(n_h - np.searchsorted(pool.healthy, s, side="left"))
        p_convict = (1.0 + n_ge) / (n_h + 1.0) if n_h else 1.0

        n_d = len(pool.diseased)
        # diseased cases scoring at least as innocent as this one
        n_le = int(np.searchsorted(pool.diseased, s, side="right"))
        p_discharge = (1.0 + n_le) / (n_d + 1.0) if n_d else 1.0

        return float(min(p_convict, 1.0)), float(min(p_discharge, 1.0))

    def e_values(self, score: float, stratum: int) -> tuple[float, float]:
        """(e_convict, e_discharge) for one shot.

        The default route is conformal p-value -> Vovk-Wang calibrator, which
        is exactly valid in finite samples and needs no density estimate. It
        is also lossy: a calibrator has to be valid against *every* alternative,
        so it cannot exploit knowing what the alternative looks like.
        `LikelihoodRatioCalibrator` overrides this with the powerful-but-
        estimated route.
        """
        from src.evidence.ewealth import p_to_e

        p_c, p_d = self.p_values(score, stratum)
        return float(p_to_e(p_c)), float(p_to_e(p_d))


class StratifiedCalibrator(Calibrator):
    """Sound version: p-values conditional on the predicted degradation stratum.

    Falls back to the marginal pool for strata with too few calibration points
    (recorded in `fallback_strata`). That fallback is a real, reportable
    weakening of the guarantee for those strata, not a free pass -- experiments
    print it rather than hiding it.
    """

    def _pool_for(self, stratum: int) -> _Pool:
        assert self._marginal is not None, "call fit() first"
        z = int(np.clip(stratum, 0, self.n_strata - 1))
        if z in self.fallback_strata:
            return self._marginal
        return self._pools[z]


class LikelihoodRatioCalibrator(StratifiedCalibrator):
    """Powerful route: the density ratio between the two calibration pools.

    The likelihood ratio is the canonical e-value -- under the null,
    E[f1(S)/f0(S)] = integral of f1 = 1 exactly -- and by Neyman-Pearson it is
    the most powerful statistic available. The p-value route cannot match it,
    because a p-to-e calibrator must be valid against every alternative at once
    and so cannot use the fact that we know exactly what "caries" scores look
    like: we have a labelled calibration set of them.

    The catch, stated plainly: E[e] = 1 holds for the *true* densities. These
    are estimated from finite calibration data, so validity here is empirical
    rather than proved, and E2 measures it directly instead of assuming it.
    Two guards keep the estimate honest:

      * histogram densities on quantile bins of the pooled scores within a
        stratum, with Laplace smoothing, so no bin can produce a divide-by-zero
        or an unbounded ratio from a single calibration point;
      * an upper clip on the ratio. Clipping downward can only *lower* E[e],
        so it is safe in the direction that matters -- it costs power, never
        validity.

    Use `Calibrator`/`StratifiedCalibrator` when the guarantee must be provable,
    and this when power matters more; the benchmark reports both.
    """

    def __init__(self, n_strata: int = 4, n_bins: int = 12, e_max: float = 50.0, smoothing: float = 1.0):
        super().__init__(n_strata=n_strata)
        self.n_bins = n_bins
        self.e_max = e_max
        self.smoothing = smoothing
        self._bins: dict[int, np.ndarray] = {}
        self._ratio: dict[int, np.ndarray] = {}

    def fit(self, scores, labels, usabilities) -> "LikelihoodRatioCalibrator":
        super().fit(scores, labels, usabilities)
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels).astype(int)
        strata = np.array([self.stratum_of(u) for u in np.asarray(usabilities)], dtype=int)

        for z in range(self.n_strata):
            mask = strata == z
            pool_scores = scores[mask] if mask.sum() >= MIN_STRATUM_N else scores
            pool_labels = labels[mask] if mask.sum() >= MIN_STRATUM_N else labels
            # quantile bin edges on the pooled scores: equal-width bins would
            # leave most bins empty, since scores concentrate near 0 and 1
            qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
            edges = np.unique(np.quantile(pool_scores, qs))
            idx = np.searchsorted(edges, pool_scores, side="right")
            n_bins_actual = len(edges) + 1

            c1 = np.bincount(idx[pool_labels == 1], minlength=n_bins_actual) + self.smoothing
            c0 = np.bincount(idx[pool_labels == 0], minlength=n_bins_actual) + self.smoothing
            ratio = (c1 / c1.sum()) / (c0 / c0.sum())
            self._bins[z] = edges
            self._ratio[z] = np.clip(ratio, 1.0 / self.e_max, self.e_max)
        return self

    def e_values(self, score: float, stratum: int) -> tuple[float, float]:
        """(e_convict, e_discharge) = (f1/f0, f0/f1) in this score's bin."""
        z = int(np.clip(stratum, 0, self.n_strata - 1))
        edges = self._bins[z]
        b = int(np.searchsorted(edges, float(score), side="right"))
        lr = float(self._ratio[z][min(b, len(self._ratio[z]) - 1)])
        return lr, float(np.clip(1.0 / lr, 1.0 / self.e_max, self.e_max))


class MarginalCalibrator(Calibrator):
    """One pool, ignoring the degradation channel: the default anyone would build.

    Split-conformal as usually implemented, with no acknowledgement that the
    retake loop selects on image quality. Kept as a first-class arm so the
    ablation can measure what ignoring that costs.

    Measured outcome (E2, E4): it stays under its nominal error rate -- the
    selection shift happens to run conservative in this capture process -- and
    costs only a little power. See the module docstring for why that is a
    property of this process rather than a guarantee.
    """

    def _pool_for(self, stratum: int) -> _Pool:
        assert self._marginal is not None, "call fit() first"
        return self._marginal
