"""Tests for the chest X-ray loader (src/data/chest_xray_crops.py).

Only the pure functions are exercised here -- loading real parquet data is
covered by actually running E13 against the downloaded dataset, not by unit
tests, the same convention DENTEX's loader follows (see test_dentex.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.chest_xray_crops import ChestCrop, describe, load_chest_crops, split_test


def _fake_crops(n_normal: int, n_pneumonia: int) -> list[ChestCrop]:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    crops = [ChestCrop(image=img, label=0, source_file=f"normal:{i}") for i in range(n_normal)]
    crops += [ChestCrop(image=img, label=1, source_file=f"pneumonia:{i}") for i in range(n_pneumonia)]
    return crops


def test_split_test_partitions_without_overlap_or_loss():
    crops = _fake_crops(30, 20)
    cal, held = split_test(crops, fraction=0.5, seed=0)
    assert len(cal) + len(held) == len(crops)
    cal_ids = {c.source_file for c in cal}
    held_ids = {c.source_file for c in held}
    assert cal_ids.isdisjoint(held_ids)


def test_split_test_respects_the_fraction():
    crops = _fake_crops(100, 100)
    cal, held = split_test(crops, fraction=0.25, seed=0)
    assert len(cal) == 50  # 0.25 * 200


def test_split_test_is_deterministic_given_a_seed():
    crops = _fake_crops(30, 20)
    cal1, _ = split_test(crops, seed=3)
    cal2, _ = split_test(crops, seed=3)
    assert [c.source_file for c in cal1] == [c.source_file for c in cal2]


def test_split_test_differs_across_seeds():
    crops = _fake_crops(30, 20)
    cal1, _ = split_test(crops, seed=1)
    cal2, _ = split_test(crops, seed=2)
    assert [c.source_file for c in cal1] != [c.source_file for c in cal2]


def test_describe_reports_correct_counts():
    crops = _fake_crops(3, 7)
    text = describe(crops, "test")
    assert "10" in text  # total
    assert "7" in text  # pneumonia count


def test_describe_handles_empty_list():
    assert describe([], "empty") == "empty        empty"


def test_load_chest_crops_raises_with_a_helpful_message_when_missing():
    with pytest.raises(FileNotFoundError, match="snapshot_download"):
        load_chest_crops(root=Path("/nonexistent/definitely/not/here"))
