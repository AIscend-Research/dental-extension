#!/usr/bin/env python
"""Upload a local DENTEX split to Kaggle as a Dataset (step 1 of docs/kaggle_instructions.md).

    pip install kaggle
    # https://www.kaggle.com/settings -> API -> Create New Token
    mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

    python scripts/upload_dentex_to_kaggle.py --dry-run          # check everything, upload nothing
    python scripts/upload_dentex_to_kaggle.py                    # the training split (~2-3 GB)
    python scripts/upload_dentex_to_kaggle.py --split validation # optional held-out eval split

Re-running after the dataset exists fails on purpose (Kaggle would 409); use
`--update -m "what changed"` to push a new version of a dataset you already own.

Every check runs before a single byte is uploaded, because each of these
failures otherwise costs you the whole upload: missing data, missing
credentials, a slug Kaggle will reject, or a validation split whose labels
were left behind in the parent directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.kaggle_upload import (  # noqa: E402
    DEFAULT_LICENSE,
    SPLITS,
    UploadError,
    build_metadata,
    human_bytes,
    kaggle_available,
    preflight,
    resolve_username,
    run_upload,
    stage_split,
    upload_command,
    validate_slug,
    write_metadata,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--split", choices=sorted(SPLITS), default="training",
                   help="which DENTEX split to publish (default: training)")
    p.add_argument("--dentex-root", default="data/dentex",
                   help="the directory containing DENTEX/ (default: data/dentex)")
    p.add_argument("--slug", help="dataset slug; defaults to a per-split name")
    p.add_argument("--username", help="Kaggle username (default: KAGGLE_USERNAME or ~/.kaggle/kaggle.json)")
    p.add_argument("--license", dest="license_name", default=DEFAULT_LICENSE,
                   help=f"license name as Kaggle spells it (default: {DEFAULT_LICENSE})")
    p.add_argument("--update", action="store_true",
                   help="push a new version of an existing dataset instead of creating one")
    p.add_argument("-m", "--message", default="", help="version message, with --update")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing dataset-metadata.json")
    p.add_argument("--dry-run", action="store_true",
                   help="run every check and print the command, but do not upload")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    spec = SPLITS[args.split]

    try:
        if not args.dry_run and not kaggle_available():
            raise UploadError("The kaggle CLI is not installed in this interpreter: pip install kaggle")

        username = args.username or resolve_username()
        slug = validate_slug(args.slug or spec.slug)

        staged = stage_split(args.dentex_root, spec, dry_run=args.dry_run)
        info = preflight(args.dentex_root, spec)
        metadata = build_metadata(spec, username, slug, args.license_name)
    except UploadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for name in staged:
        print(f"staged {name} into the upload directory (it lives above it on disk)")

    print(f"split      {spec.name}  ({spec.relpath})")
    print(f"directory  {info['path']}")
    print(f"contents   {info['files']} files, {info['images']} images, {human_bytes(info['bytes'])}")
    print(f"dataset    {metadata['id']}   [{args.license_name}]")

    cmd = upload_command(info["path"], update=args.update, message=args.message)

    if args.dry_run:
        print("\ndry run -- nothing written, nothing uploaded. Would run:")
        print("  " + " ".join(cmd))
        return 0

    try:
        path = write_metadata(info["path"], metadata, force=args.force)
    except UploadError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"wrote      {path}")

    print("\nuploading (this streams for a while -- progress below is the CLI's own):")
    code = run_upload(cmd)
    if code != 0:
        print(
            "\nUpload failed. Common causes:\n"
            "  403 -> the token is stale, or you have not accepted DENTEX's terms on Hugging Face\n"
            "  409 -> that slug already exists; re-run with --update -m 'what changed'\n"
            "  a stall near 100% is Kaggle processing the zip -- give it a few minutes before retrying",
            file=sys.stderr,
        )
        return code

    print(
        f"\nDone. Kaggle slug: {metadata['id']}\n"
        f"  page: https://www.kaggle.com/datasets/{metadata['id']}\n"
        "Next: attach it to kaggle/00_setup_and_sanity_check.ipynb (Add Data -> Your Datasets).\n"
        "No path edits needed -- find_dentex_root() locates the mount by annotation file,\n"
        "not by dataset name."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
