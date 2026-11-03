import re
from pathlib import Path

import unzipy

ROOT = Path(__file__).resolve().parent.parent
HINT = "run: uv run python scripts/release.py <patch|minor|major|X.Y.Z>"


def _pyproject_version():
    text = (ROOT / "pyproject.toml").read_text()
    return re.search(r'(?m)^version = "(.+)"$', text).group(1)


def test_package_version_matches_pyproject():
    assert unzipy.__version__ == _pyproject_version(), (
        "reinstall the package (uv sync)"
    )


def test_citation_version_matches_pyproject():
    text = (ROOT / "CITATION.cff").read_text()
    found = re.search(r"(?m)^version: (.+)$", text).group(1)
    assert found == _pyproject_version(), f"CITATION.cff is stale; {HINT}"


def test_changelog_has_an_entry_for_the_version():
    text = (ROOT / "CHANGELOG.md").read_text()
    heading = f"## {_pyproject_version()} - "
    assert heading in text, f"CHANGELOG.md has no '{heading}...'; {HINT}"
