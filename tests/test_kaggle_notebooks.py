"""Preflight for the `kaggle/` notebooks: everything checkable without a session.

The notebooks are the one part of this repo that nothing else exercises. They
run on hardware this project has never had, they cost GPU hours, and a
mistake in them surfaces 10 minutes (notebook 00) or many hours (notebook 01)
into a session — after the queue, the pip builds and the 900 MB backbone
download.

So the cheap failures get caught here instead:

- **A stale clone URL.** Every notebook starts by cloning this repo into the
  session. If that URL points at a fork, the session runs *that* fork's code
  regardless of what is merged here, and nothing in the output says so. This
  test is the reason `REPO_URL` has exactly one definition.
- **A rename this repo made and the notebooks didn't.** They import from
  `src.*` by name; those names are checked against the modules' ASTs (not by
  importing, which would need cv2/torch/detectron2).
- **A syntax error in a cell**, which otherwise shows up only when the cell
  is reached, possibly hours in.
- **Committed outputs**, which carry patient-image renderings and megabytes
  of pip logs into git.
"""

import ast
import json
from pathlib import Path

import pytest

from src.utils import kaggle_env

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO_ROOT / "kaggle").glob("*.ipynb"))


def _ids(path):
    return path.name


def test_there_are_notebooks_to_check():
    """Guards the glob itself -- an empty parametrize passes silently."""
    assert len(NOTEBOOKS) == 4


def code_cells(nb_path: Path) -> list[str]:
    nb = json.loads(nb_path.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def strip_magics(source: str) -> str:
    """Drop `!shell` and `%magic` lines so the rest can be parsed as Python.

    Two details, both from notebook 00's guarded backbone download:
    the `!curl` line ends in a backslash, so its continuation lines have to go
    too; and it is the entire body of an `else:`, so magics become `pass` at
    their original indentation rather than blank lines.
    """
    out, skipping = [], False
    for line in source.splitlines():
        stripped = line.strip()
        if skipping or stripped.startswith(("!", "%")):
            indent = line[: len(line) - len(line.lstrip())]
            out.append("" if skipping else f"{indent}pass")
            skipping = stripped.endswith("\\")
            continue
        out.append(line)
    return "\n".join(out)


def module_toplevel_names(module_path: Path) -> set[str]:
    """Names a module defines at top level, read from its AST.

    Deliberately not `import` + `dir()`: these modules pull in cv2, torch and
    detectron2, none of which need to be installed to check that a name the
    notebook imports still exists.

    Recurses through `try`/`if` bodies, because that is where the optional-
    dependency fallbacks live -- `ConfidenceHead` is defined twice in
    `src/models/confidence_head.py`, once under `import torch` and once under
    its ImportError, and a top-level-only scan would call it missing.
    """
    names = set()

    def visit(body):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                names.update(t.id for t in node.targets if isinstance(t, ast.Name))
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update((a.asname or a.name).split(".")[0] for a in node.names)
            elif isinstance(node, ast.Try):
                visit(node.body + node.orelse + node.finalbody)
                for handler in node.handlers:
                    visit(handler.body)
            elif isinstance(node, ast.If):
                visit(node.body + node.orelse)

    visit(ast.parse(module_path.read_text()).body)
    return names


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_notebook_is_valid_json_with_code_cells(nb_path):
    nb = json.loads(nb_path.read_text())
    assert nb["cells"], f"{nb_path.name} has no cells"
    assert code_cells(nb_path)


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_no_committed_outputs(nb_path):
    """Outputs carry rendered patient X-rays and pip logs into git."""
    nb = json.loads(nb_path.read_text())
    with_output = [
        i for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code" and c.get("outputs")
    ]
    assert not with_output, (
        f"{nb_path.name} has outputs saved in cells {with_output} -- "
        "clear them before committing (Kernel -> Restart & Clear Output)"
    )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_every_cell_parses_as_python(nb_path):
    for i, source in enumerate(code_cells(nb_path)):
        try:
            ast.parse(strip_magics(source))
        except SyntaxError as exc:
            pytest.fail(f"{nb_path.name} code cell {i} does not parse: {exc}")


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_clone_url_matches_the_canonical_repo(nb_path):
    """A fork URL here runs a fork's code in the session, silently."""
    text = nb_path.read_text()
    assert kaggle_env.REPO_URL in text, (
        f"{nb_path.name} does not clone {kaggle_env.REPO_URL}. Every notebook's bootstrap "
        "must use the same URL as src/utils/kaggle_env.REPO_URL, or a Kaggle session runs "
        "code from somewhere else."
    )
    for stale in ("christopherh-88/dental-extension",):
        assert stale not in text, f"{nb_path.name} still points at {stale}"


def test_repo_url_is_not_a_fork():
    assert kaggle_env.REPO_URL == "https://github.com/AIscend-Research/dental-extension.git"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_names_imported_from_src_still_exist(nb_path):
    """Catches a rename in src/ that the notebooks were not updated for."""
    for i, source in enumerate(code_cells(nb_path)):
        tree = ast.parse(strip_magics(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("src."):
                continue
            module_path = REPO_ROOT / Path(*node.module.split(".")).with_suffix(".py")
            assert module_path.exists(), (
                f"{nb_path.name} cell {i} imports from {node.module}, which does not exist"
            )
            available = module_toplevel_names(module_path)
            for alias in node.names:
                assert alias.name in available, (
                    f"{nb_path.name} cell {i}: {node.module} has no {alias.name!r} "
                    f"(it was probably renamed; the notebook fails at this cell in-session)"
                )


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_repo_scripts_the_notebooks_shell_out_to_exist(nb_path):
    """`!bash scripts/foo.sh` fails in-session if the script moved."""
    for source in code_cells(nb_path):
        for line in source.splitlines():
            stripped = line.strip().lstrip("!").strip()
            for token in stripped.split():
                if token.startswith(("scripts/", "configs/")) and "*" not in token:
                    assert (REPO_ROOT / token).exists(), f"{nb_path.name} references missing {token}"


@pytest.mark.parametrize("nb_path", NOTEBOOKS, ids=_ids)
def test_dentex_mount_is_looked_up_not_hardcoded(nb_path):
    """The mount path is whatever the uploader named the dataset.

    `find_dentex_root()` exists precisely because a hardcoded
    `/kaggle/input/<slug>` breaks for everyone but its author.
    """
    text = nb_path.read_text()
    assert "/kaggle/input/dentex" not in text, (
        f"{nb_path.name} hardcodes a dataset mount path -- use find_dentex_root()"
    )
