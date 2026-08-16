"""Checks for the DENTEX -> Kaggle Dataset upload.

Everything here is the part that must be right *before* a multi-gigabyte
upload starts, since every one of these failures is only observable after the
bytes have already gone over the wire (or, worse, silently produces a dataset
with images and no labels). None of it touches the network or an account.
"""

import json

import pytest

from src.utils import kaggle_upload as ku


def _make_training_split(root, *, images=3, annotation=True):
    d = root / "DENTEX" / "training_data" / "quadrant-enumeration-disease"
    (d / "xrays").mkdir(parents=True)
    for i in range(images):
        (d / "xrays" / f"train_{i}.png").write_bytes(b"\x89PNG" + b"0" * 100)
    if annotation:
        (d / "train_quadrant_enumeration_disease.json").write_text("{}")
    return d


def _make_validation_split(root, *, triple_at_root=True):
    d = root / "DENTEX" / "validation_data" / "quadrant_enumeration_disease" / "xrays"
    d.mkdir(parents=True)
    (d / "val_0.png").write_bytes(b"\x89PNG")
    if triple_at_root:
        (root / "DENTEX" / "validation_triple.json").write_text("{}")
    return d


# --- preflight ------------------------------------------------------------


def test_preflight_measures_a_complete_split(tmp_path):
    target = _make_training_split(tmp_path, images=5)
    info = ku.preflight(tmp_path, ku.SPLITS["training"])
    assert info["path"] == target
    assert info["images"] == 5
    assert info["files"] == 6  # 5 images + the annotation json
    assert info["bytes"] > 0


def test_preflight_rejects_a_missing_split_with_download_instructions(tmp_path):
    with pytest.raises(ku.UploadError) as exc:
        ku.preflight(tmp_path, ku.SPLITS["training"])
    assert "download_dentex.py" in str(exc.value)


def test_preflight_rejects_images_without_annotations(tmp_path):
    """The failure mode that produces an unusable dataset rather than an error."""
    _make_training_split(tmp_path, annotation=False)
    with pytest.raises(ku.UploadError) as exc:
        ku.preflight(tmp_path, ku.SPLITS["training"])
    assert "train_quadrant_enumeration_disease.json" in str(exc.value)


def test_preflight_rejects_an_empty_xrays_dir(tmp_path):
    target = _make_training_split(tmp_path, images=0)
    assert target.exists()
    with pytest.raises(ku.UploadError) as exc:
        ku.preflight(tmp_path, ku.SPLITS["training"])
    assert "xrays/" in str(exc.value)


# --- staging the validation labels ---------------------------------------


def test_validation_staging_pulls_in_the_labels_from_above(tmp_path):
    _make_validation_split(tmp_path)
    spec = ku.SPLITS["validation"]

    # Without staging, the upload dir has images and no labels.
    assert not (ku.split_dir(tmp_path, spec) / "validation_triple.json").exists()

    assert ku.stage_split(tmp_path, spec) == ["validation_triple.json"]
    assert (ku.split_dir(tmp_path, spec) / "validation_triple.json").exists()
    ku.preflight(tmp_path, spec)  # complete now


def test_staging_is_idempotent_and_never_overwrites(tmp_path):
    _make_validation_split(tmp_path)
    spec = ku.SPLITS["validation"]
    staged = ku.split_dir(tmp_path, spec) / "validation_triple.json"
    staged.write_text('{"mine": true}')

    assert ku.stage_split(tmp_path, spec) == []
    assert json.loads(staged.read_text()) == {"mine": True}


def test_dry_run_staging_reports_without_copying(tmp_path):
    _make_validation_split(tmp_path)
    spec = ku.SPLITS["validation"]
    assert ku.stage_split(tmp_path, spec, dry_run=True) == ["validation_triple.json"]
    assert not (ku.split_dir(tmp_path, spec) / "validation_triple.json").exists()


# --- slugs and metadata ---------------------------------------------------


@pytest.mark.parametrize("slug", ["dentex", "dentex-quadrant-enumeration-disease", "abc123"])
def test_valid_slugs(slug):
    assert ku.validate_slug(slug) == slug


@pytest.mark.parametrize(
    "slug", ["short", "Dentex-Caps", "dentex_underscore", "-leading", "trailing-", "d" * 51]
)
def test_kaggle_rejects_these_slugs_so_we_reject_them_first(slug):
    with pytest.raises(ku.UploadError):
        ku.validate_slug(slug)


def test_metadata_is_shaped_the_way_kaggle_wants_it():
    meta = ku.build_metadata(ku.SPLITS["training"], "someuser")
    assert meta["id"] == "someuser/dentex-quadrant-enumeration-disease"
    assert meta["licenses"] == [{"name": "CC BY-NC-SA 4.0"}]
    assert 6 <= len(meta["title"]) <= 50
    assert "non-commercial" in meta["description"]


def test_metadata_rejects_a_bad_explicit_slug():
    with pytest.raises(ku.UploadError):
        ku.build_metadata(ku.SPLITS["training"], "someuser", slug="Bad_Slug")


def test_write_metadata_refuses_to_silently_retarget_an_existing_dataset(tmp_path):
    first = ku.build_metadata(ku.SPLITS["training"], "someuser")
    ku.write_metadata(tmp_path, first)

    other = ku.build_metadata(ku.SPLITS["training"], "otheruser")
    with pytest.raises(ku.UploadError) as exc:
        ku.write_metadata(tmp_path, other)
    assert "--force" in str(exc.value)

    ku.write_metadata(tmp_path, other, force=True)
    assert json.loads((tmp_path / ku.METADATA_FILENAME).read_text())["id"] == other["id"]


def test_write_metadata_is_a_noop_when_it_already_matches(tmp_path):
    meta = ku.build_metadata(ku.SPLITS["training"], "someuser")
    ku.write_metadata(tmp_path, meta)
    assert ku.write_metadata(tmp_path, meta).exists()


# --- credentials ----------------------------------------------------------


def test_username_from_env_beats_the_credentials_file(tmp_path):
    path = tmp_path / "kaggle.json"
    path.write_text(json.dumps({"username": "fromfile", "key": "x"}))
    assert ku.resolve_username({"KAGGLE_USERNAME": "fromenv"}, path) == "fromenv"


def test_username_falls_back_to_the_credentials_file(tmp_path):
    path = tmp_path / "kaggle.json"
    path.write_text(json.dumps({"username": "fromfile", "key": "x"}))
    assert ku.resolve_username({}, path) == "fromfile"


def test_missing_credentials_say_where_to_get_them(tmp_path):
    with pytest.raises(ku.UploadError) as exc:
        ku.resolve_username({}, tmp_path / "nope.json")
    assert "kaggle.com/settings" in str(exc.value)


# --- the command itself ---------------------------------------------------


def test_create_command_zips_the_directory(tmp_path):
    cmd = ku.upload_command(tmp_path)
    assert cmd[1:] == ["-m", "kaggle", "datasets", "create", "-p", str(tmp_path), "--dir-mode", "zip"]


def test_update_command_versions_instead_of_creating(tmp_path):
    cmd = ku.upload_command(tmp_path, update=True, message="re-unzipped")
    assert "version" in cmd and "create" not in cmd
    assert cmd[-2:] == ["-m", "re-unzipped"]


def test_human_bytes():
    assert ku.human_bytes(512) == "512 B"
    assert ku.human_bytes(2 * 1024**3) == "2.0 GB"
