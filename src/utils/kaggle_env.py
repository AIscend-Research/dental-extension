"""Kaggle session setup: the parts that actually break, handled once.

The notebooks used to carry the same shell snippets copy-pasted four times.
Each copy had the same Kaggle-specific failure modes, none of which show up
when you run the code locally -- which is why they survived a local
verification pass:

1. `pip install -r requirements-core.txt` can **upgrade numpy**, and Kaggle's
   preinstalled torch (and anything compiled against it, including the
   detectron2 you are about to build) is compiled for the numpy already in
   the image. The upgrade "succeeds", then torch dies at import with
   "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x".
   `install_deps()` pins numpy and torch to the installed versions via a pip
   constraints file, so nothing can silently move them.

2. Re-running the clone cell after a session restart fails with "destination
   path already exists", and the `%cd` that follows then runs from the wrong
   directory -- so every relative path in the notebook silently points
   somewhere else. `ensure_repo()` is idempotent.

3. `DATA_ROOT` was hardcoded to `/kaggle/input/dentex/...`, but the mount
   path is whatever the person named their Kaggle Dataset.
   `find_dentex_root()` locates it instead of guessing.

Nothing here is Kaggle-only in a way that stops it running elsewhere: every
function works on a normal machine too, which is how it gets tested.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/christopherh-88/dental-extension.git"
REPO_DIRNAME = "dental-extension"

# The annotation file that identifies a real DENTEX diagnosis split.
DENTEX_ANNOTATION_GLOB = "**/train_quadrant_enumeration_disease.json"

# Runtime imports the notebooks need, mapped to the pip name that provides
# them. Only the missing ones get installed -- Kaggle already ships most.
RUNTIME_REQUIREMENTS = {
    "cv2": "opencv-python-headless",
    "pycocotools": "pycocotools",
    "yaml": "pyyaml",
    "matplotlib": "matplotlib",
    "timm": "timm",
    "scipy": "scipy",
    "fvcore": "fvcore",
}


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def ensure_repo(url: str = REPO_URL, dirname: str = REPO_DIRNAME) -> Path:
    """Clone the repo and chdir into it, safely on a re-run.

    Returns the repo root. If we're already inside it (session restart, or
    the cell was run twice), this is a no-op rather than an error.
    """
    cwd = Path.cwd()
    if (cwd / "src" / "data" / "degradation.py").exists():
        return cwd  # already at the repo root
    target = cwd / dirname
    if not target.exists():
        subprocess.run(["git", "clone", "--depth", "1", url, str(target)], check=True)
    os.chdir(target)
    if str(target) not in sys.path:
        sys.path.insert(0, str(target))
    return target


def _installed_version(module: str) -> str | None:
    try:
        mod = __import__(module)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def pip_constraints(pins: dict[str, str], path: str | Path = "/tmp/pip-constraints.txt") -> Path:
    """Write a pip constraints file pinning packages to their current versions.

    This is what stops `pip install <anything>` from dragging numpy or torch
    to a new major version as a transitive dependency -- the failure that
    breaks compiled extensions in a way that looks like a code bug.
    """
    path = Path(path)
    path.write_text("".join(f"{name}=={version}\n" for name, version in pins.items() if version))
    return path


def missing_requirements(requirements: dict[str, str] | None = None) -> list[str]:
    """Pip names for the runtime imports that aren't importable yet."""
    requirements = requirements or RUNTIME_REQUIREMENTS
    return sorted({pkg for mod, pkg in requirements.items() if _installed_version(mod) is None
                   and not _importable(mod)})


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:
        return False


def install_deps(extra: list[str] | None = None, quiet: bool = True) -> list[str]:
    """Install only what's missing, with numpy and torch pinned in place.

    Returns the list of packages it actually installed (empty if none were
    needed), so the notebook can print something meaningful instead of a wall
    of pip output.
    """
    pins = {}
    for name in ("numpy", "torch", "torchvision"):
        version = _installed_version(name)
        if version:
            pins[name] = version.split("+")[0]  # strip local CUDA suffix, e.g. 2.1.0+cu121

    needed = missing_requirements() + list(extra or [])
    needed = [p for p in dict.fromkeys(needed)]
    if not needed:
        return []

    cmd = [sys.executable, "-m", "pip", "install"]
    if quiet:
        cmd.append("-q")
    if pins:
        cmd += ["-c", str(pip_constraints(pins))]
    cmd += needed
    subprocess.run(cmd, check=True)
    return needed


def find_dentex_root(
    extra_search_paths: list[str | Path] | None = None, include_defaults: bool = True
) -> Path:
    """Locate the DENTEX quadrant-enumeration-disease directory.

    Searches attached Kaggle datasets and the local checkout, rather than
    assuming a dataset slug. Returns the directory containing both the
    annotation json and the `xrays/` folder.

    Args:
        extra_search_paths: searched before the defaults.
        include_defaults: also search /kaggle/input and ./data. Pass False to
            search only what you named (tests rely on this -- otherwise a real
            local dataset satisfies a lookup that was meant to fail).

    Raises:
        FileNotFoundError listing where it looked -- an explicit message beats
        an `AssertionError` on a hardcoded path that was never going to exist.
    """
    roots = [Path(p) for p in (extra_search_paths or [])]
    if include_defaults:
        roots += [Path("/kaggle/input"), Path("data/dentex"), Path("data")]

    searched = []
    for root in roots:
        if not root.exists():
            searched.append(f"{root} (does not exist)")
            continue
        searched.append(str(root))
        for match in sorted(root.glob(DENTEX_ANNOTATION_GLOB)):
            if (match.parent / "xrays").is_dir():
                return match.parent
    raise FileNotFoundError(
        "Could not find DENTEX's quadrant-enumeration-disease folder (needs "
        f"{DENTEX_ANNOTATION_GLOB.split('/')[-1]} next to an xrays/ directory).\n"
        f"Looked in: {', '.join(searched)}\n"
        "On Kaggle: attach the DENTEX dataset via 'Add Data' in the sidebar "
        "(see docs/phase2_data_notes.md for how to upload it). Locally: run "
        "python scripts/download_dentex.py."
    )


def import_hierarchicaldet(baseline_path: str | Path = "external/HierarchicalDet"):
    """Import hierarchialdet against the REAL detectron2, not the vendored copy.

    HierarchicalDet vendors `detectron2/` (no compiled ops) and `pycocotools/`
    (no compiled `_mask`). Adding its directory to sys.path before importing
    the real packages makes `import detectron2` silently resolve to the broken
    vendored copy. Importing the real ones first caches them in sys.modules,
    which the later path insertion cannot undo. See SETUP.md.

    Returns `add_diffusiondet_config`.
    """
    import pycocotools.coco  # noqa: F401  -- cache the real ones first
    import pycocotools.cocoeval  # noqa: F401
    import pycocotools.mask  # noqa: F401
    import detectron2  # noqa: F401

    try:
        from detectron2 import _C  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "detectron2's compiled ops (_C) failed to import. Nothing "
            "downstream will work. This usually means detectron2 was built "
            "against a different torch than the one now loaded -- on Kaggle, "
            "never reinstall torch; build detectron2 against the image's "
            "version. See SETUP.md Track B."
        ) from exc

    baseline_path = str(baseline_path)
    if baseline_path not in sys.path:
        sys.path.insert(0, baseline_path)
    from hierarchialdet.config import add_diffusiondet_config

    return add_diffusiondet_config


def summarize_environment() -> dict:
    """torch/CUDA/GPU facts worth printing at the top of a session."""
    info = {"python": sys.version.split()[0], "on_kaggle": on_kaggle(), "cwd": str(Path.cwd())}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
    except ImportError:
        info["torch"] = None
    try:
        import numpy

        info["numpy"] = numpy.__version__
    except ImportError:
        info["numpy"] = None
    return info
