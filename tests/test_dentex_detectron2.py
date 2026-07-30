"""Checks for the detectron2-registration helpers in src/data/dentex.py.

Skipped entirely if detectron2 is not installed -- these need the detector
venv (see SETUP.md Track B), not the core one. Uses a small synthetic COCO
fixture shaped like the real DENTEX schema rather than requiring the full
dataset to be downloaded just to run these unit tests.
"""

import pytest

detectron2 = pytest.importorskip("detectron2")

from detectron2.data import DatasetCatalog, MetadataCatalog

from src.data.dentex import register_dentex_detectron2, to_detectron2_dicts


def _real_schema_coco():
    return {
        "images": [
            {"id": 1, "file_name": "train_1.png", "height": 100, "width": 200},
            {"id": 2, "file_name": "train_2.png", "height": 100, "width": 200},
        ],
        "annotations": [
            {"image_id": 1, "bbox": [10, 10, 20, 20], "category_id_1": 0,
             "category_id_2": 0, "category_id_3": 1, "iscrowd": 0},
            {"image_id": 2, "bbox": [5, 5, 10, 10], "category_id_1": 0,
             "category_id_2": 0, "category_id_3": 0, "iscrowd": 0},
        ],
        "categories_1": [{"id": 0, "name": "1"}],
        "categories_2": [{"id": 0, "name": "1"}],
        "categories_3": [
            {"id": 0, "name": "Impacted"},
            {"id": 1, "name": "Caries"},
            {"id": 2, "name": "Periapical Lesion"},
            {"id": 3, "name": "Deep Caries"},
        ],
    }


def test_to_detectron2_dicts_shape_and_fields():
    coco = _real_schema_coco()
    dicts = to_detectron2_dicts(coco, "/some/root")
    assert len(dicts) == 2
    d = dicts[0]
    assert d["file_name"] == "/some/root/train_1.png"
    assert d["image_id"] == 1
    assert len(d["annotations"]) == 1
    ann = d["annotations"][0]
    assert ann["bbox"] == [10, 10, 20, 20]
    assert ann["category_id_3"] == 1
    assert "bbox_mode" in ann
    # flat "category_id" alias (== category_id_3) -- detectron2's own
    # get_detection_dataset_dicts() needs this for its internal class
    # histogram logging and raises KeyError without it (confirmed by running
    # build_detection_train_loader against real DENTEX data).
    assert ann["category_id"] == ann["category_id_3"]


def test_to_detectron2_dicts_filters_by_image_ids():
    coco = _real_schema_coco()
    dicts = to_detectron2_dicts(coco, "/some/root", image_ids=[2])
    assert len(dicts) == 1
    assert dicts[0]["image_id"] == 2


def test_register_dentex_detectron2_registers_expected_names():
    coco = _real_schema_coco()
    split = {"train": [1], "val": [2]}
    register_dentex_detectron2(coco, "/some/root", split, name_prefix="test_reg")
    assert "test_reg_train_class" in DatasetCatalog.list()
    assert "test_reg_validation_class" in DatasetCatalog.list()
    assert "test_reg_test_class" not in DatasetCatalog.list()  # no "test" key in split

    train_dicts = DatasetCatalog.get("test_reg_train_class")
    assert len(train_dicts) == 1
    assert train_dicts[0]["image_id"] == 1

    meta = MetadataCatalog.get("test_reg_train_class")
    assert meta.thing_classes_3 == ["Impacted", "Caries", "Periapical Lesion", "Deep Caries"]


def test_register_dentex_detectron2_is_idempotent():
    coco = _real_schema_coco()
    split = {"train": [1]}
    register_dentex_detectron2(coco, "/some/root", split, name_prefix="test_idempotent")
    register_dentex_detectron2(coco, "/some/root", split, name_prefix="test_idempotent")  # should not raise


if __name__ == "__main__":
    for fn in [
        test_to_detectron2_dicts_shape_and_fields,
        test_to_detectron2_dicts_filters_by_image_ids,
        test_register_dentex_detectron2_registers_expected_names,
        test_register_dentex_detectron2_is_idempotent,
    ]:
        fn()
        print("PASS", fn.__name__)
