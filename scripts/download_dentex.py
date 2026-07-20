"""
Download the DENTEX dataset from Hugging Face into data/dentex.

DENTEX is on HF at ibrahimhamamci/DENTEX (CC BY-SA 4.0). You need to accept the
dataset terms on the HF page and be logged in (`huggingface-cli login`) before
this works -- it is gated behind agreeing to the license.

    pip install huggingface_hub
    huggingface-cli login
    python scripts/download_dentex.py

Only the fully-labelled split (quadrant-enumeration-diagnosis) has the diagnosis
classes (caries, deep_caries, periapical_lesion, impacted). Point the config at
that split's COCO json once you see the exact filenames after download.
"""

from __future__ import annotations

from pathlib import Path

OUT = Path("data/dentex")


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "pip install huggingface_hub, then `huggingface-cli login` first."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading ibrahimhamamci/DENTEX -> {OUT} ...")
    path = snapshot_download(
        repo_id="ibrahimhamamci/DENTEX",
        repo_type="dataset",
        local_dir=str(OUT),
    )
    print("Downloaded to:", path)
    print("\nNow inspect the layout and update configs/default.yaml paths:")
    print("  find", OUT, "-name '*.json' | head")
    print("  find", OUT, "-type d")


if __name__ == "__main__":
    main()
