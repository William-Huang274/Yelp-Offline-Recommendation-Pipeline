from __future__ import annotations

import re
from pathlib import Path


LINK_RE = re.compile(r"\]\((?!https?://|#)([^)]+)\)")


def _assert_links_resolve(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    missing: list[str] = []
    for target in LINK_RE.findall(text):
        full = (path.parent / target).resolve()
        if not full.exists():
            missing.append(target)
    assert not missing, f"{path.name} has unresolved links: {missing}"


def test_public_readme_links_resolve(repo_root: Path) -> None:
    for rel in ("README.md", "README.zh-CN.md", "docs/README.md", "docs/README.zh-CN.md"):
        _assert_links_resolve(repo_root / rel)
