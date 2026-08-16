"""Publish a local DENTEX split to Kaggle as a Dataset.

This is step 1 of `docs/kaggle_instructions.md` -- the account action every
notebook in `kaggle/` is blocked on, because `find_dentex_root()` can only
find a dataset somebody actually uploaded.

The upload itself is three `kaggle` CLI calls, and that is exactly why it is
worth scripting: the three calls are easy, and the things that go wrong
around them are not, and all of them fail *after* you have waited for a
multi-gigabyte upload:

1. **The wrong directory.** `training_data/` is 11 GB and only its
   `quadrant-enumeration-disease/` subfolder carries diagnosis labels --
   the only split `configs/default.yaml` and all four notebooks read.
   Uploading the parent wastes hours and Kaggle quota on data nothing uses.
2. **A silently incomplete validation upload.** The validation split's
   annotations (`validation_triple.json`) live one level *above*
   `validation_data/`, so the obvious `-p validation_data` upload produces a
   dataset with images and no labels. `stage_split()` copies it in.
3. **A slug that doesn't match.** Kaggle rejects a slug shorter than 6 chars
   or with underscores/capitals after the upload has streamed, and
   `find_dentex_root()` locates the mount by looking for
   `train_quadrant_enumeration_disease.json` -- not by name -- so the slug is
   free to be readable rather than magic.

Every function here works off a plain directory, so the whole thing is
testable without a Kaggle account (`tests/test_kaggle_upload.py`).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# DENTEX is CC BY-NC-SA 4.0 (non-commercial) -- see scripts/download_dentex.py.
# Kaggle's license list uses exactly this spelling; a name outside its list is
# rejected at create time.
DEFAULT_LICENSE = "CC BY-NC-SA 4.0"

METADATA_FILENAME = "dataset-metadata.json"

# Kaggle slug rules: lowercase alphanumerics and dashes, 6-50 characters.
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{4,48}[a-z0-9]$")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class UploadError(RuntimeError):
    """Anything that should stop us before a multi-GB upload starts."""


@dataclass(frozen=True)
class SplitSpec:
    """One uploadable DENTEX split."""

    name: str
    relpath: str  # relative to the dentex root (the dir holding DENTEX/)
    slug: str
    title: str
    subtitle: str
    # Files that must exist under relpath before the upload is worth starting.
    # Relative to the split dir; a trailing "/" means "a non-empty directory".
    required: tuple[str, ...]
    # Files to copy in from elsewhere under the dentex root, {dest: source}.
    staged: tuple[tuple[str, str], ...] = ()


SPLITS: dict[str, SplitSpec] = {
    "training": SplitSpec(
        name="training",
        relpath="DENTEX/training_data/quadrant-enumeration-disease",
        slug="dentex-quadrant-enumeration-disease",
        title="DENTEX Quadrant Enumeration Disease",
        subtitle="705 panoramic dental X-rays with quadrant, tooth and diagnosis labels",
        required=("train_quadrant_enumeration_disease.json", "xrays/"),
    ),
    "validation": SplitSpec(
        name="validation",
        relpath="DENTEX/validation_data",
        slug="dentex-validation-quadrant-enumeration-disease",
        title="DENTEX Validation (Quadrant Enumeration Disease)",
        subtitle="Held-out DENTEX validation panoramics with the validation_triple labels",
        required=("quadrant_enumeration_disease/xrays/",),
        # validation_triple.json sits at DENTEX/, one level above the folder
        # we upload -- without this the dataset would ship images and no labels.
        staged=(("validation_triple.json", "DENTEX/validation_triple.json"),),
    ),
}


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------


def credentials_path() -> Path:
    """Where the kaggle CLI looks for kaggle.json (honouring KAGGLE_CONFIG_DIR)."""
    base = os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle")
    return Path(base) / "kaggle.json"


def resolve_username(env: dict[str, str] | None = None, path: Path | None = None) -> str:
    """The Kaggle username the dataset will be created under.

    Environment wins over kaggle.json, matching the CLI's own precedence, so
    a session can override without editing the file.
    """
    env = os.environ if env is None else env
    if env.get("KAGGLE_USERNAME"):
        return env["KAGGLE_USERNAME"]

    path = credentials_path() if path is None else path
    if not path.exists():
        raise UploadError(
            f"No Kaggle credentials: {path} does not exist and KAGGLE_USERNAME is unset.\n"
            "Get them from https://www.kaggle.com/settings -> API -> 'Create New Token',\n"
            f"then: mkdir -p {path.parent} && mv ~/Downloads/kaggle.json {path} "
            f"&& chmod 600 {path}"
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise UploadError(f"{path} is not valid JSON: {exc}") from exc
    username = data.get("username")
    if not username:
        raise UploadError(f"{path} has no 'username' key -- re-download it from Kaggle settings.")
    return username


# --------------------------------------------------------------------------
# the split on disk
# --------------------------------------------------------------------------


def split_dir(dentex_root: Path | str, spec: SplitSpec) -> Path:
    return Path(dentex_root) / spec.relpath


def stage_split(dentex_root: Path | str, spec: SplitSpec, dry_run: bool = False) -> list[str]:
    """Copy in files the split needs but that live outside its directory.

    Returns what was copied (empty if everything was already in place). Never
    overwrites: a file already there is assumed to be the one you want.
    """
    root = Path(dentex_root)
    target = split_dir(root, spec)
    copied = []
    for dest_rel, source_rel in spec.staged:
        dest, source = target / dest_rel, root / source_rel
        if dest.exists() or not source.exists():
            continue
        if not dry_run:
            shutil.copy2(source, dest)
        copied.append(dest_rel)
    return copied


def preflight(dentex_root: Path | str, spec: SplitSpec) -> dict:
    """Check the split is complete and measure it. Raises before any upload."""
    target = split_dir(dentex_root, spec)
    if not target.is_dir():
        raise UploadError(
            f"{target} does not exist.\n"
            "The DENTEX zips are gated behind the HF license and are gitignored, so a\n"
            "fresh clone has no data. Get it first:\n"
            "    pip install huggingface_hub && huggingface-cli login\n"
            "    python scripts/download_dentex.py\n"
            "    unzip data/dentex/DENTEX/training_data.zip -d data/dentex/DENTEX/\n"
            "    unzip data/dentex/DENTEX/validation_data.zip -d data/dentex/DENTEX/"
        )

    missing = []
    for rel in spec.required:
        needed = target / rel.rstrip("/")
        if rel.endswith("/"):
            if not needed.is_dir() or not any(needed.iterdir()):
                missing.append(f"{rel} (missing or empty)")
        elif not needed.is_file():
            missing.append(rel)
    if missing:
        raise UploadError(
            f"{target} is not a complete '{spec.name}' split -- missing: {', '.join(missing)}.\n"
            "Unzip the split fully before uploading; a partial upload has to be versioned over."
        )

    files = [p for p in target.rglob("*") if p.is_file()]
    images = [p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    return {
        "path": target,
        "files": len(files),
        "images": len(images),
        "bytes": sum(p.stat().st_size for p in files),
    }


def human_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"  # unreachable, keeps type checkers happy


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def validate_slug(slug: str) -> str:
    if not SLUG_RE.match(slug):
        raise UploadError(
            f"Invalid Kaggle dataset slug {slug!r}: use 6-50 lowercase letters, digits or\n"
            "dashes (no underscores, no capitals). Kaggle rejects a bad slug only after the\n"
            "upload has finished streaming."
        )
    return slug


def build_metadata(
    spec: SplitSpec,
    username: str,
    slug: str | None = None,
    license_name: str = DEFAULT_LICENSE,
) -> dict:
    slug = validate_slug(slug or spec.slug)
    return {
        "title": spec.title,
        "id": f"{username}/{slug}",
        "subtitle": spec.subtitle,
        "licenses": [{"name": license_name}],
        "description": (
            f"The `{spec.relpath}` split of the DENTEX dataset "
            "(https://huggingface.co/datasets/ibrahimhamamci/DENTEX), uploaded unmodified "
            "for use from Kaggle notebooks.\n\n"
            "Licensed CC BY-NC-SA 4.0 by the original authors -- non-commercial use only. "
            "Cite the DENTEX challenge paper (Hamamci et al.), not this mirror.\n\n"
            "Used by https://github.com/AIscend-Research/dental-extension "
            "(`kaggle/*.ipynb`), which locates this mount by looking for "
            "`train_quadrant_enumeration_disease.json`, not by dataset name."
        ),
    }


def write_metadata(target: Path, metadata: dict, force: bool = False) -> Path:
    path = Path(target) / METADATA_FILENAME
    if path.exists() and not force:
        existing = json.loads(path.read_text())
        if existing.get("id") != metadata["id"]:
            raise UploadError(
                f"{path} already exists and points at {existing.get('id')!r}, not "
                f"{metadata['id']!r}.\nPass --force to overwrite it, or --slug to match it."
            )
        return path
    path.write_text(json.dumps(metadata, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------
# the upload
# --------------------------------------------------------------------------


def upload_command(target: Path, update: bool = False, message: str = "") -> list[str]:
    """The kaggle CLI invocation, as a list (never a shell string).

    `--dir-mode zip` matters: DENTEX ships ~700 loose images, and Kaggle's
    default per-file mode makes an unusable dataset page out of them.
    """
    base = [sys.executable, "-m", "kaggle", "datasets"]
    if update:
        return base + ["version", "-p", str(target), "--dir-mode", "zip", "-m", message or "update"]
    return base + ["create", "-p", str(target), "--dir-mode", "zip"]


def kaggle_available() -> bool:
    try:
        subprocess.run(
            [sys.executable, "-m", "kaggle", "--version"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, OSError):
        return False


def run_upload(cmd: list[str]) -> int:
    """Stream the CLI's progress straight through -- a silent multi-GB upload
    is indistinguishable from a hang."""
    return subprocess.run(cmd).returncode
