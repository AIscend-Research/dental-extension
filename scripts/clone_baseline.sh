#!/usr/bin/env bash
# Clone the HierarchicalDet baseline into external/. It is NOT a submodule on
# purpose -- you will be editing it (stripping heads to caries-only, adding the
# confidence head), so you want your own copy under version control with the
# rest of the project, not a pinned pointer.
set -euo pipefail

TARGET="external/HierarchicalDet"
if [ -d "$TARGET/.git" ]; then
  echo "$TARGET already cloned."
  exit 0
fi

echo "Cloning HierarchicalDet (DiffusionDet + vendored Detectron2, Swin-L)..."
git clone https://github.com/ibrahimethemhamamci/HierarchicalDet.git "$TARGET"

echo
echo "Done. Note:"
echo "  - external/HierarchicalDet/detectron2 is a MODIFIED detectron2 with no"
echo "    build files and no compiled ops. See SETUP.md for how to get the"
echo "    compiled ops (detectron2._C) working -- this is the main install snag."
echo "  - Backbone weights (swin_large_patch4_window7_224_22k.pkl -- LARGE, not"
echo "    Base, despite what the upstream config's filename says) are NOT in"
echo "    the repo. SETUP.md has the download + convert recipe; using the Base"
echo "    file loads with zero errors and silently randomizes most of the"
echo "    backbone."
