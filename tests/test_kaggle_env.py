"""Checks for the Kaggle session bootstrap.

These exist because the failures they guard against are invisible locally --
the notebooks ran fine on a dev machine while carrying Kaggle-only breakage
(a numpy upgrade that kills compiled torch, a non-idempotent clone, a
hardcoded dataset mount path). Testing the logic here is the only way to
exercise it without a Kaggle session.
"""

import os
from pathlib import Path

import pytest

from src.utils import kaggle_env


def _make_dentex_tree(root: Path, name="dentex-full"):
    d = root / name / "DENTEX" / "training_data" / "quadrant-enumeration-disease"
    (d / "xrays").mkdir(parents=True)
    (d / "train_quadrant_enumeration_disease.json").write_text("{}")
    return d


def test_find_dentex_root_locates_an_attached_dataset(tmp_path):
    expected = _make_dentex_tree(tmp_path)
    assert kaggle_env.find_dentex_root([tmp_path], include_defaults=False) == expected


def test_find_dentex_root_ignores_json_without_xrays(tmp_path):
    """A stray annotation file is not a usable dataset root."""
    stray = tmp_path / "somewhere"
    stray.mkdir()
    (stray / "train_quadrant_enumeration_disease.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        kaggle_env.find_dentex_root([tmp_path], include_defaults=False)


def test_find_dentex_root_error_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        kaggle_env.find_dentex_root([tmp_path / "nope"], include_defaults=False)
    message = str(exc.value)
    assert "Add Data" in message  # tells you how to fix it on Kaggle
    assert "download_dentex.py" in message  # ...and locally


def test_find_dentex_root_works_regardless_of_dataset_slug(tmp_path):
    """The mount path depends on what the person named their Kaggle Dataset."""
    expected = _make_dentex_tree(tmp_path, name="my-weird-dataset-name-v3")
    assert kaggle_env.find_dentex_root([tmp_path], include_defaults=False) == expected


def test_pip_constraints_pins_versions(tmp_path):
    path = kaggle_env.pip_constraints(
        {"numpy": "1.26.4", "torch": "2.1.0"}, tmp_path / "c.txt"
    )
    text = path.read_text()
    assert "numpy==1.26.4" in text
    assert "torch==2.1.0" in text


def test_pip_constraints_skips_unknown_versions(tmp_path):
    path = kaggle_env.pip_constraints({"numpy": "1.26.4", "torch": None}, tmp_path / "c.txt")
    assert path.read_text().strip() == "numpy==1.26.4"


def test_missing_requirements_excludes_what_is_importable():
    """numpy is installed in every env this runs in, so it must never appear."""
    missing = kaggle_env.missing_requirements({"numpy": "numpy", "cv2": "opencv-python-headless"})
    assert "numpy" not in missing


def test_missing_requirements_reports_absent_modules():
    missing = kaggle_env.missing_requirements({"definitely_not_a_real_module_xyz": "ghost-pkg"})
    assert missing == ["ghost-pkg"]


def test_ensure_repo_is_idempotent_inside_the_repo(tmp_path, monkeypatch):
    """Re-running the setup cell must not re-clone or chdir somewhere wrong.

    The old notebook cell (`!git clone` + `%cd`) failed on re-run and then
    silently continued from the wrong directory, breaking every relative path.
    """
    fake_repo = tmp_path / "repo"
    (fake_repo / "src" / "data").mkdir(parents=True)
    (fake_repo / "src" / "data" / "degradation.py").write_text("")
    monkeypatch.chdir(fake_repo)

    def explode(*a, **k):
        raise AssertionError("must not clone when already inside the repo")

    monkeypatch.setattr(kaggle_env.subprocess, "run", explode)
    assert kaggle_env.ensure_repo() == fake_repo
    assert Path.cwd() == fake_repo


def test_ensure_repo_reuses_an_existing_clone(tmp_path, monkeypatch):
    workdir = tmp_path / "working"
    workdir.mkdir()
    existing = workdir / kaggle_env.REPO_DIRNAME
    (existing / "src" / "data").mkdir(parents=True)
    (existing / "src" / "data" / "degradation.py").write_text("")
    monkeypatch.chdir(workdir)

    def explode(*a, **k):
        raise AssertionError("must not clone over an existing checkout")

    monkeypatch.setattr(kaggle_env.subprocess, "run", explode)
    assert kaggle_env.ensure_repo() == existing
    assert Path.cwd() == existing


def test_summarize_environment_reports_the_basics():
    info = kaggle_env.summarize_environment()
    assert "python" in info and "numpy" in info and "cwd" in info
    assert isinstance(info["on_kaggle"], bool)


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_find_dentex_root_locates_an_attached_dataset(Path(d))
    print("PASS (run the full suite with pytest for the rest)")
