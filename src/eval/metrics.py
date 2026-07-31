"""
Evaluation metrics.

The selective-prediction metrics ARE fully implemented here, because they are
the paper's real contribution and they do not need the detector: they operate
on two arrays you can produce from any model -- whether each prediction was
correct, and how confident the model was. Build and unit-test these now.

"Safe deferral": if the model defers (asks for a retake / refers to a
clinician) on its least-confident cases, how much does accuracy on the cases it
keeps improve, and at what deferral cost? That is a risk-coverage curve, and
the single number that summarises it is the area under it.

The detection metrics (mAP, per-class F1) ARE also implemented (Phase 4), and
also do not need a live detector -- they operate on plain COCO-format
predictions/ground-truth (image_id, category_id, bbox, score), which is the
interface any detector (HierarchicalDet or otherwise) ultimately produces.
Fully testable today on synthetic fixtures; real numbers still require Phase 3
(the detector stack, GPU) to actually produce predictions.
"""

from __future__ import annotations

from collections import defaultdict

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


def expected_calibration_error(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error (Guo et al. 2017): |accuracy - confidence|.

    AURC/safe_deferral_rate check whether confidence *ranks* correct
    predictions above incorrect ones -- they say nothing about whether the
    numeric confidence value is calibrated (whether a 0.9 usability score
    really corresponds to ~90% observed accuracy). A model can rank perfectly
    while being wildly overconfident or underconfident; that distinction
    matters for a clinic deciding what "usability_score < 0.4" should mean in
    practice, so track this alongside AURC, not instead of it.

    Bins predictions into n_bins equal-width confidence bins, and returns the
    bin-size-weighted average gap between each bin's mean confidence and its
    actual accuracy. 0.0 is perfectly calibrated.
    """
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    n = len(correct)
    if n == 0:
        return float("nan")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        if not mask.any():
            continue
        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def reliability_diagram_bins(
    correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10
) -> list[dict]:
    """Per-bin (mean_confidence, accuracy, count) for plotting a reliability diagram.

    A perfectly calibrated model has accuracy == mean_confidence in every bin
    (the diagonal); systematic deviation above/below the diagonal shows over-
    or under-confidence at that confidence range. See src/eval/plots.py for
    the plotting function that consumes this.
    """
    correct = np.asarray(correct, dtype=np.float64)
    confidence = np.asarray(confidence, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi if i < n_bins - 1 else confidence <= hi)
        count = int(mask.sum())
        rows.append({
            "bin_lo": float(lo),
            "bin_hi": float(hi),
            "count": count,
            "mean_confidence": float(confidence[mask].mean()) if count else float("nan"),
            "accuracy": float(correct[mask].mean()) if count else float("nan"),
        })
    return rows


def ablation_table(
    arms: dict[str, tuple[np.ndarray, np.ndarray]],
    target_accuracy: float = 0.95,
) -> list[dict]:
    """Compare named ablation arms on the selective-prediction metrics.

    This is the Phase 4 ablation deliverable: baseline vs. robustness-only vs.
    confidence-head-only vs. combined vs. +fusion, isolated by comparing each
    arm's (correct, confidence) arrays from a run on the same held-out set.

    Args:
        arms: {arm_name: (correct, confidence)}, e.g. {"baseline": (...),
            "robustness-only": (...), "confidence-head-only": (...),
            "combined": (...), "+fusion": (...)}.
        target_accuracy: passed to safe_deferral_rate for the deferral-rate
            column (the operational "defer X% to hit target_accuracy" number).

    Returns:
        One dict per arm: arm, n, base_accuracy, aurc, safe_deferral_rate.
        Print or write to CSV for the paper's ablation table.
    """
    rows = []
    for name, (correct, confidence) in arms.items():
        correct = np.asarray(correct, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        rows.append({
            "arm": name,
            "n": len(correct),
            "base_accuracy": float(correct.mean()) if len(correct) else float("nan"),
            "aurc": area_under_rc(correct, confidence),
            "safe_deferral_rate": safe_deferral_rate(correct, confidence, target_accuracy),
        })
    return rows


def sweep_decision_thresholds(
    usability: np.ndarray,
    detection_score: np.ndarray,
    correct: np.ndarray,
    retake_belows: np.ndarray | list[float],
    refer_belows: np.ndarray | list[float],
) -> list[dict]:
    """Grid-sweep confidence_head.decide()'s two thresholds against outcomes.

    This is the "sweeping thresholds" step Phase 4 calls for: for each
    (retake_below, refer_below) pair, applies the same decision rule as
    src/models/confidence_head.py:decide() to every sample and reports how
    that operating point performs -- accuracy on the cases it actually
    predicts on, and what fraction get sent back as a retake or a referral.

    Args:
        usability: per-sample usability score in [0, 1] (confidence head output).
        detection_score: per-sample detector confidence in [0, 1].
        correct: per-sample bool/0-1, whether the kept prediction was right.
        retake_belows / refer_belows: threshold values to try (refer_below
            should be >= the paired retake_below for decide()'s ordering to
            make sense, but every pair is tried -- inspect the output).

    Returns:
        One dict per (retake_below, refer_below) pair with keys:
        retake_below, refer_below, coverage, accuracy_on_predicted,
        retake_rate, refer_rate.
    """
    from src.models.confidence_head import decide

    usability = np.asarray(usability, dtype=np.float64)
    detection_score = np.asarray(detection_score, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = len(usability)
    if not (len(detection_score) == len(correct) == n):
        raise ValueError("usability, detection_score, and correct must be same length")

    results = []
    for retake_below in retake_belows:
        for refer_below in refer_belows:
            decisions = [
                decide(u, d, retake_below=retake_below, refer_below=refer_below)
                for u, d in zip(usability, detection_score)
            ]
            decisions = np.array(decisions)
            predict_mask = decisions == "predict"
            coverage = float(predict_mask.mean()) if n else float("nan")
            acc_on_predicted = (
                float(correct[predict_mask].mean()) if predict_mask.any() else float("nan")
            )
            results.append({
                "retake_below": retake_below,
                "refer_below": refer_below,
                "coverage": coverage,
                "accuracy_on_predicted": acc_on_predicted,
                "retake_rate": float((decisions == "retake").mean()) if n else float("nan"),
                "refer_rate": float((decisions == "refer_clinician").mean()) if n else float("nan"),
            })
    return results


# ---------------------------------------------------------------------------
# Detection metrics.
#
# These operate on plain COCO-format predictions/ground-truth, so they are
# fully implemented and testable now -- they do NOT need a live detector.
# What they DO need is real predictions to evaluate, which needs Phase 3 (the
# detector stack, GPU). Until then, exercise them on synthetic fixtures (see
# tests/test_metrics.py) so they're ready the moment real predictions exist.
# ---------------------------------------------------------------------------


def coco_map(predictions: list[dict], ground_truth: dict) -> dict[str, float]:
    """COCO mAP (bbox) via pycocotools.

    Args:
        predictions: list of COCO result dicts, each
            {"image_id", "category_id", "bbox": [x, y, w, h], "score"}.
        ground_truth: a COCO-format dict with "images", "annotations",
            "categories" (e.g. what src/data/dentex.load_coco returns). Each
            annotation must be COCO-complete -- "id", "iscrowd", "area" as
            well as bbox/category_id/image_id -- or pycocotools raises
            KeyError. Real DENTEX annotations already carry all of these.

    Returns:
        {"mAP": ..., "mAP50": ..., "mAP75": ...} (COCO's AP, AP@.5, AP@.75).
        pycocotools returns -1.0 (not 0.0 or NaN) when a category has no
        ground-truth instances to score against -- that's "undefined", not
        "zero AP"; don't average it in with real scores without filtering it.
    """
    if not predictions:
        # pycocotools' loadRes([]) crashes (IndexError on anns[0]) rather than
        # returning empty results -- a detector predicting nothing is a real
        # scenario (untrained/degenerate model), so handle it directly: zero
        # predictions means zero AP at every threshold, not a crash.
        return {"mAP": 0.0, "mAP50": 0.0, "mAP75": 0.0}

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO()
    coco_gt.dataset = ground_truth
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "mAP": float(coco_eval.stats[0]),
        "mAP50": float(coco_eval.stats[1]),
        "mAP75": float(coco_eval.stats[2]),
    }


def _bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes (COCO convention)."""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def per_class_f1(
    predictions: list[dict], ground_truth: dict, iou_threshold: float = 0.5
) -> dict[str, dict]:
    """Per-diagnosis precision/recall/F1 at a fixed IoU, via greedy matching.

    Pure Python (no pycocotools) so it is easy to unit-test and to reuse for
    the caries-only collapse. Predictions are matched to ground-truth boxes of
    the same category and image, highest-score first, one-to-one.

    Args:
        predictions: list of {"image_id", "category_id", "bbox", "score"}.
        ground_truth: COCO-format dict with "annotations" and "categories".

    Returns:
        {category_name: {"precision", "recall", "f1", "n_gt"}}.
    """
    id_to_name = {c["id"]: c["name"] for c in ground_truth.get("categories", [])}

    gts_by_cat_img: dict[tuple, list] = defaultdict(list)
    for ann in ground_truth.get("annotations", []):
        gts_by_cat_img[(ann["category_id"], ann["image_id"])].append(ann["bbox"])

    preds_by_cat: dict[int, list] = defaultdict(list)
    for p in predictions:
        preds_by_cat[p["category_id"]].append(p)

    results: dict[str, dict] = {}
    for cat_id, name in id_to_name.items():
        preds = sorted(preds_by_cat.get(cat_id, []), key=lambda p: -p["score"])
        matched = {
            key: [False] * len(boxes)
            for key, boxes in gts_by_cat_img.items()
            if key[0] == cat_id
        }
        n_gt = sum(len(boxes) for key, boxes in gts_by_cat_img.items() if key[0] == cat_id)

        tp = 0
        for p in preds:
            key = (cat_id, p["image_id"])
            gt_boxes = gts_by_cat_img.get(key, [])
            best_iou, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                if matched[key][j]:
                    continue
                iou = _bbox_iou(p["bbox"], gt_box)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou >= iou_threshold:
                matched[key][best_j] = True
                tp += 1

        n_pred = len(preds)
        fp = n_pred - tp
        fn = n_gt - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_gt": n_gt,
        }
    return results


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
