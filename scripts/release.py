"""Bump the version everywhere with one command.

Usage:
    uv run python scripts/release.py patch   # 0.1.1 -> 0.1.2
    uv run python scripts/release.py minor   # 0.1.1 -> 0.2.0
    uv run python scripts/release.py 0.3.0   # an explicit version

It updates ``pyproject.toml`` and ``uv.lock`` (through ``uv version``),
``CITATION.cff`` (version and release date) and ``CHANGELOG.md`` (the
"Unreleased" entries move under the new version). ``unzipy.__version__`` is
read from the installed metadata, so it needs no edit.
"""

import re
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(spec):
    """Apply the bump described by ``spec`` (patch, minor, major or X.Y.Z).

    Args:
        spec (str): ``patch``, ``minor``, ``major`` or an explicit version.
    """
    if spec in ("patch", "minor", "major"):
        cmd = ["uv", "version", "--bump", spec]
    else:
        cmd = ["uv", "version", spec]
    subprocess.run(cmd, cwd=ROOT, check=True)
    new = subprocess.run(
        ["uv", "version", "--short"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    today = date.today().isoformat()

    cff = ROOT / "CITATION.cff"
    text = cff.read_text()
    text = re.sub(r"(?m)^version: .*$", f"version: {new}", text)
    text = re.sub(r"(?m)^date-released: .*$", f"date-released: {today}", text)
    cff.write_text(text)

    log = ROOT / "CHANGELOG.md"
    text = log.read_text()
    text = text.replace(
        "## Unreleased\n", f"## Unreleased\n\n## {new} - {today}\n", 1
    )
    log.write_text(text)
    print(f"Version {new}. Review CHANGELOG.md, then open a PR.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
