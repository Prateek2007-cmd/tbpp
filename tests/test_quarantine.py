"""Quarantine guard (Phase 4).

``uav3d.modeling.legacy`` holds code that invents geometry (symmetry
completion, LoD-2 minaret templates, the duplicate TSDF dense path). No
production module may import it — if a future change re-wires it, this test
fails loudly instead of letting invented geometry reach a demo.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src" / "uav3d"
LEGACY = SRC / "modeling" / "legacy"

# Import roots that must never appear outside the legacy directory itself.
FORBIDDEN_ROOTS = (
    "uav3d.modeling.legacy",
    "uav3d.modeling.completion",
    "uav3d.modeling.lod2",
    "uav3d.modeling.dense",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_no_production_import_of_quarantined_code() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if LEGACY in path.parents:
            continue
        for module in _imported_modules(path):
            if module.startswith(FORBIDDEN_ROOTS):
                offenders.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    assert not offenders, (
        "Quarantined hallucination code reached a production module:\n"
        + "\n".join(offenders)
    )


def test_quarantined_files_still_exist_with_readme() -> None:
    # Quarantine is a move, not a deletion — the code stays characterised.
    assert (LEGACY / "completion.py").exists()
    assert (LEGACY / "lod2.py").exists()
    assert (LEGACY / "dense_tsdf.py").exists()
    readme = (LEGACY / "README.md").read_text(encoding="utf-8")
    assert "Quarantined" in readme
    assert "invent" in readme.lower()


def test_legacy_is_not_a_package() -> None:
    # No __init__.py: the modules are not part of the package surface.
    assert not (LEGACY / "__init__.py").exists()
