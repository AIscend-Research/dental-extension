"""Load configs/default.yaml.

configs/default.yaml's own header says "load this in your scripts; do not
hardcode paths in modules" -- but nothing in the repo actually read it, so it
was documentation pretending to be configuration, free to drift out of sync
with the code (it did: it named the wrong backbone checkpoint for a while).
This module is the loader that makes it real.

Kaggle notebooks are the deliberate exception: they run against
/kaggle/input/... paths that only exist inside a Kaggle session, so they set
their own DATA_ROOT and treat this file as the record of where things live
locally rather than as the source of truth.

    from src.utils.config import load_config
    cfg = load_config()
    cfg["degradation"]["severity_range"]   # -> [0.3, 0.9]
    cfg.paths.image_root                   # -> absolute Path, repo-root-relative
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access, so cfg.model.task reads as well as cfg["model"]["task"]."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path: str | Path | None = None) -> Config:
    """Read the project config into a Config (dict + attribute access).

    Args:
        path: config file to read; defaults to configs/default.yaml.

    Raises:
        FileNotFoundError: with the resolved path, rather than a bare KeyError
            somewhere downstream.
    """
    path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    import yaml

    with open(path) as f:
        return Config(yaml.safe_load(f) or {})


def resolve_path(relative: str | Path) -> Path:
    """Turn a repo-root-relative config path into an absolute one.

    The config documents its paths as "relative to repo root", which only
    works if callers resolve them against the repo root rather than against
    whatever directory they happen to be run from.
    """
    relative = Path(relative)
    return relative if relative.is_absolute() else REPO_ROOT / relative


if __name__ == "__main__":
    cfg = load_config()
    print("task:", cfg.model.task)
    print("backbone weights:", cfg.model.backbone_weights)
    print("severity range:", cfg.degradation.severity_range)
    print("thresholds:", dict(cfg.confidence))
    print("annotations exist:", resolve_path(cfg.data.coco_annotations).exists())
