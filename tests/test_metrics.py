"""Runnable checks for the selective-prediction metrics."""

import numpy as np

from src.eval.metrics import (
    risk_coverage_curve,
    area_under_rc,
    safe_deferral_rate,
    accuracy_at_coverage,
    ablation_table,
    sweep_decision_thresholds,
    coco_map,
    per_class_f1,
)


def test_perfect_confidence_beats_random():
    rng = np.random.default_rng(0)
    n = 1000
    correct = rng.random(n) < 0.8
    good = correct * rng.uniform(0.5, 1, n) + (~correct) * rng.uniform(0, 0.5, n)
    rand = rng.random(n)
    assert area_under_rc(correct, good) > area_under_rc(correct, rand)


def test_full_coverage_equals_base_accuracy():
    correct = np.array([1, 1, 0, 1, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    coverage, accuracy = risk_coverage_curve(correct, conf)
    assert coverage[-1] == 1.0
    assert np.isclose(accuracy[-1], correct.mean())


def test_safe_deferral_bounds():
    correct = np.array([1, 1, 1, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.7, 0.1])
    # deferring the single wrong (least confident) case hits 100% accuracy
    assert safe_deferral_rate(correct, conf, 1.0) == 0.25
    # target met at full coverage -> zero deferral
    assert safe_deferral_rate(correct, conf, 0.5) == 0.0
    # unreachable target -> defer everything
    assert safe_deferral_rate(correct, conf, 1.01) == 1.0


def test_accuracy_at_coverage_monotone_region():
    correct = np.array([1, 1, 0, 0], dtype=float)
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    assert accuracy_at_coverage(correct, conf, 0.5) == 1.0


def test_ablation_table_ranks_arms_by_confidence_quality():
    rng = np.random.default_rng(0)
    n = 1000
    correct = rng.random(n) < 0.8
    good_conf = correct * rng.uniform(0.5, 1.0, n) + (~correct) * rng.uniform(0, 0.5, n)
    rand_conf = rng.random(n)
    rows = ablation_table(
        {"baseline (random conf)": (correct, rand_conf), "full system (good conf)": (correct, good_conf)},
        target_accuracy=0.95,
    )
    by_arm = {r["arm"]: r for r in rows}
    assert by_arm["full system (good conf)"]["aurc"] > by_arm["baseline (random conf)"]["aurc"]
    assert (by_arm["full system (good conf)"]["safe_deferral_rate"]
            < by_arm["baseline (random conf)"]["safe_deferral_rate"])
    assert np.isclose(by_arm["baseline (random conf)"]["base_accuracy"],
                       by_arm["full system (good conf)"]["base_accuracy"])


def test_sweep_decision_thresholds_low_retake_keeps_everything():
    # usability always 1.0 -> never below any retake_below/refer_below tried,
    # so every sample is "predict" regardless of the threshold pair.
    usability = np.array([1.0, 1.0, 1.0, 1.0])
    detection = np.array([1.0, 1.0, 1.0, 1.0])
    correct = np.array([1, 1, 0, 1], dtype=float)
    rows = sweep_decision_thresholds(usability, detection, correct, [0.1], [0.2])
    assert len(rows) == 1
    assert rows[0]["coverage"] == 1.0
    assert np.isclose(rows[0]["accuracy_on_predicted"], correct.mean())
    assert rows[0]["retake_rate"] == 0.0


def test_sweep_decision_thresholds_high_retake_defers_everything():
    usability = np.array([0.3, 0.3, 0.3, 0.3])
    detection = np.array([0.9, 0.9, 0.9, 0.9])
    correct = np.array([1, 1, 0, 1], dtype=float)
    rows = sweep_decision_thresholds(usability, detection, correct, [0.9], [0.95])
    assert rows[0]["coverage"] == 0.0
    assert rows[0]["retake_rate"] == 1.0
    assert np.isnan(rows[0]["accuracy_on_predicted"])


def _tiny_coco_fixture():
    # pycocotools requires each annotation to be COCO-complete (id, iscrowd,
    # area, bbox, category_id, image_id) -- real DENTEX annotations already
    # carry all of these (see src/data/dentex.py), so this fixture matches
    # that rather than the minimal shape used elsewhere in this file.
    ground_truth = {
        "images": [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "b.png"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20],
             "iscrowd": 0, "area": 400},
            {"id": 2, "image_id": 2, "category_id": 0, "bbox": [5, 5, 10, 10],
             "iscrowd": 0, "area": 100},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [50, 50, 10, 10],
             "iscrowd": 0, "area": 100},
        ],
        "categories": [{"id": 0, "name": "caries"}, {"id": 1, "name": "impacted"}],
    }
    predictions = [
        # near-perfect match on image 1 -> true positive for "caries"
        {"image_id": 1, "category_id": 0, "bbox": [10, 10, 20, 20], "score": 0.95},
        # correct box, correct image, but no ground truth for "impacted" there
        # -> false positive
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 5, 5], "score": 0.6},
    ]
    return predictions, ground_truth


def test_per_class_f1_matches_expected_tp_fp_fn():
    predictions, ground_truth = _tiny_coco_fixture()
    result = per_class_f1(predictions, ground_truth, iou_threshold=0.5)
    # caries: 1 predicted (matches image-1 gt), image-2 caries gt unmatched -> fn
    assert result["caries"]["precision"] == 1.0
    assert np.isclose(result["caries"]["recall"], 0.5)
    # impacted: 1 predicted, 0 iou overlap with the real impacted box -> fp, and
    # the real impacted gt is never matched -> fn
    assert result["impacted"]["precision"] == 0.0
    assert result["impacted"]["recall"] == 0.0
    assert result["impacted"]["n_gt"] == 1


def test_coco_map_runs_and_returns_expected_keys():
    predictions, ground_truth = _tiny_coco_fixture()
    result = coco_map(predictions, ground_truth)
    assert set(result.keys()) == {"mAP", "mAP50", "mAP75"}
    assert all(isinstance(v, float) for v in result.values())


if __name__ == "__main__":
    for fn in [
        test_perfect_confidence_beats_random,
        test_full_coverage_equals_base_accuracy,
        test_safe_deferral_bounds,
        test_accuracy_at_coverage_monotone_region,
        test_ablation_table_ranks_arms_by_confidence_quality,
        test_sweep_decision_thresholds_low_retake_keeps_everything,
        test_sweep_decision_thresholds_high_retake_defers_everything,
        test_per_class_f1_matches_expected_tp_fp_fn,
        test_coco_map_runs_and_returns_expected_keys,
    ]:
        fn()
        print("PASS", fn.__name__)
