"""Runnable checks for the DENTEX loader against its real (multi-task) schema.

Regression coverage for two bugs found by running src/data/dentex.py against
an actual DENTEX download: the real disease-labelled json has no flat
"categories"/"category_id" (it's categories_1/2/3 + category_id_1/2/3), and
DENTEX filenames (train_<n>.png) carry no patient id, so the old
strip-trailing-number heuristic collapsed every image into one bucket.
"""

from src.data.dentex import (
    CARIES_ONLY_MAP,
    DIAGNOSIS_CLASSES,
    _patient_key,
    class_balance,
    index_annotations,
    patient_level_split,
)


def _real_schema_coco():
    """A tiny fixture shaped like the actual DENTEX diagnosis json."""
    return {
        "images": [
            {"id": 1, "file_name": "train_1.png"},
            {"id": 2, "file_name": "train_2.png"},
            {"id": 3, "file_name": "train_3.png"},
            {"id": 4, "file_name": "train_4.png"},
        ],
        "annotations": [
            {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "category_id_3": 1},
            {"image_id": 2, "category_id_1": 0, "category_id_2": 0, "category_id_3": 1},
            {"image_id": 3, "category_id_1": 0, "category_id_2": 0, "category_id_3": 0},
            {"image_id": 4, "category_id_1": 0, "category_id_2": 0, "category_id_3": 3},
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


def test_class_balance_reads_diagnosis_task_not_flat_schema():
    coco = _real_schema_coco()
    counts = class_balance(coco)
    assert counts == {"Caries": 2, "Impacted": 1, "Deep Caries": 1}


def test_class_balance_still_supports_flat_coco_schema():
    coco = {
        "annotations": [{"category_id": 0}, {"category_id": 0}, {"category_id": 1}],
        "categories": [{"id": 0, "name": "caries"}, {"id": 1, "name": "bg"}],
    }
    assert class_balance(coco) == {"caries": 2, "bg": 1}


def test_patient_key_does_not_collapse_dentex_sequential_filenames():
    # train_1.png .. train_4.png must NOT all map to the same key -- that was
    # the bug: stripping the trailing index treated every image as one patient.
    keys = {_patient_key(f"train_{i}.png") for i in range(1, 5)}
    assert len(keys) == 4


def test_patient_level_split_covers_all_images_on_real_filenames():
    coco = _real_schema_coco()
    split = patient_level_split(coco, val_frac=0.25, test_frac=0.25, seed=0)
    total = sum(len(v) for v in split.values())
    assert total == len(coco["images"])
    assert len(split["val"]) > 0
    assert len(split["test"]) > 0


def test_index_annotations_groups_by_image_and_preserves_count():
    coco = _real_schema_coco()
    by_image = index_annotations(coco)
    assert sum(len(v) for v in by_image.values()) == len(coco["annotations"])
    # image 3 has one annotation (category_id_3=0) in the fixture
    assert len(by_image[3]) == 1
    # an image_id with no annotations should just be absent, not a KeyError
    assert by_image.get(999, []) == []


def test_diagnosis_classes_match_real_categories_3_names_and_order():
    # Confirmed against the downloaded categories_3: id 0=Impacted, 1=Caries,
    # 2=Periapical Lesion, 3=Deep Caries. A lowercased/reordered constant here
    # would silently fail to match real annotation category names.
    real_categories_3 = _real_schema_coco()["categories_3"]
    expected = [c["name"] for c in sorted(real_categories_3, key=lambda c: c["id"])]
    assert DIAGNOSIS_CLASSES == expected


def test_caries_only_map_keys_match_real_category_names():
    real_names = {c["name"] for c in _real_schema_coco()["categories_3"]}
    assert set(CARIES_ONLY_MAP.keys()) <= real_names


if __name__ == "__main__":
    for fn in [
        test_class_balance_reads_diagnosis_task_not_flat_schema,
        test_class_balance_still_supports_flat_coco_schema,
        test_patient_key_does_not_collapse_dentex_sequential_filenames,
        test_patient_level_split_covers_all_images_on_real_filenames,
        test_index_annotations_groups_by_image_and_preserves_count,
        test_diagnosis_classes_match_real_categories_3_names_and_order,
        test_caries_only_map_keys_match_real_category_names,
    ]:
        fn()
        print("PASS", fn.__name__)
