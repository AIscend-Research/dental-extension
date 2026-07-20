"""
Evaluation metrics.

The detection metrics (mAP, per-class F1) are thin wrappers you will point at
detectron2 / pycocotools output in Phase 4 -- left as stubs because they need
the detector running.

The selective-prediction metrics ARE fully implemented here, because they are
the paper's real contribution and they do not need the detector: they operate
on two arrays you can produce from any model -- whether each prediction was
correct, and how confident the model was. Build and unit-test these now.

"Safe deferral": if the model defers (asks for a retake / refers to a
clinician) on its least-confident cases, how much does accuracy on the cases it
keeps improve, and at what deferral cost? That is a risk-coverage curve, and
the single number that summarises it is the area under it.
"""

from __future__ import annotations

import numpy as np

# np.trapz was renamed to np.trapezoid in numpy 2.0; support both.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def risk_coverage_curve(
    correct: np.ndarray,
    confidence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy on kept predictions as a function of coverage.

    Args:
        correct: bool/0-1 array, whether each prediction was right.
        confidence: float array, the model's confidence per prediction.

    Returns:
        (coverage, accuracy) arrays. coverage[i] is the fraction of samples
        kept when we keep the i most-confident; accuracy[i] is the accuracy on
        that kept set. Sorted from most-confident (low coverage) to keeping
        everything (coverage = 1.0).
    """
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    if correct.shape != confidence.shape:
        raise ValueError("correct and confidence must have the same shape")
    n = len(correct)
    if n == 0:
        return np.array([]), np.array([])

    order = np.argsort(-confidence)  # most confident first
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    kept = np.arange(1, n + 1)
    coverage = kept / n
    accuracy = cum_correct / kept
    return coverage, accuracy


def area_under_rc(correct: np.ndarray, confidence: np.ndarray) -> float:
    """Area under the accuracy-vs-coverage curve. Higher is better.

    A perfect confidence ranking (all correct predictions ranked above all
    wrong ones) pushes this toward 1.0; random confidence gives roughly the
    base accuracy. Use this as the headline confidence-quality number.
    """
    coverage, accuracy = risk_coverage_curve(correct, confidence)
    if len(coverage) < 2:
        return float("nan")
    return float(_trapz(accuracy, coverage))


def accuracy_at_coverage(
    correct: np.ndarray, confidence: np.ndarray, coverage_target: float
) -> float:
    """Accuracy when we keep the top `coverage_target` fraction of predictions."""
    coverage, accuracy = risk_coverage_curve(correct, confidence)
    if len(coverage) == 0:
        return float("nan")
    idx = np.searchsorted(coverage, coverage_target, side="left")
    idx = min(idx, len(coverage) - 1)
    return float(accuracy[idx])


def safe_deferral_rate(
    correct: np.ndarray,
    confidence: np.ndarray,
    target_accuracy: float,
) -> float:
    """Minimum fraction we must defer to reach `target_accuracy` on the rest.

    This is the operational number for a clinic: "to be 95% accurate on the
    images we act on, we have to send back X% for a retake or a clinician."
    Returns the deferral rate (1 - coverage). Returns 0.0 if the target is met
    while keeping everything, and 1.0 if it is never reachable.
    """
    coverage, accuracy = risk_coverage_curve(correct, confidence)
    if len(coverage) == 0:
        return float("nan")
    # walk from full coverage down; find the largest coverage meeting the target
    meeting = np.where(accuracy >= target_accuracy)[0]
    if len(meeting) == 0:
        return 1.0
    best_coverage = coverage[meeting].max()
    return float(1.0 - best_coverage)


# ---------------------------------------------------------------------------
# Detection metrics -- wire these to the detector output in Phase 4.
# ---------------------------------------------------------------------------


def coco_map(predictions, ground_truth):
    """mAP via pycocotools. TODO(phase4): feed detectron2 COCOEvaluator output.

    HierarchicalDet already vendors a 3-class COCO evaluator
    (external/HierarchicalDet/hierarchialdet/util/coco_3class_eval.py). Reuse it
    rather than re-implementing mAP.
    """
    raise NotImplementedError("Connect to detectron2 COCOEvaluator in Phase 4.")


def per_class_f1(predictions, ground_truth, iou_threshold: float = 0.5):
    """Per-diagnosis F1 at a fixed IoU. TODO(phase4)."""
    raise NotImplementedError("Implement once detector predictions exist.")


if __name__ == "__main__":
    # sanity check: a good confidence signal should beat a random one
    rng = np.random.default_rng(0)
    n = 2000
    correct = rng.random(n) < 0.8  # 80% base accuracy
    # "good" confidence correlates with correctness; "random" does not
    good_conf = correct * rng.uniform(0.5, 1.0, n) + (~correct) * rng.uniform(0, 0.5, n)
    rand_conf = rng.random(n)

    print("base accuracy:", correct.mean().round(3))
    print("AURC good  :", round(area_under_rc(correct, good_conf), 3))
    print("AURC random:", round(area_under_rc(correct, rand_conf), 3))
    print("defer to hit 0.95 (good)  :",
          round(safe_deferral_rate(correct, good_conf, 0.95), 3))
    print("defer to hit 0.95 (random):",
          round(safe_deferral_rate(correct, rand_conf, 0.95), 3))
