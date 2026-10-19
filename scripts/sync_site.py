"""Mirror the dashboard into the Crow Intelligence site repo, byte for byte.

The dashboard is published from ``crow-intelligence.github.io``, folder
``projects/parlamonitor-dashboard/``. That copy used to be integrated by hand
after every re-copy -- canonical and Open Graph tags, the site bar, the
consent script, the JSON-LD -- and a plain ``cp`` silently destroyed that work.
All of it now lives in ``dashboard/index.html`` here, so the published copy is
an exact mirror and this script is the only way it changes.

What it does:

* copies the published file set (page, scripts, vendor, assets, and the two
  data files) into the site repo, and deletes files there that the source no
  longer has;
* refuses when the site copy has uncommitted changes, because those are hand
  edits this would overwrite -- move them into ``dashboard/`` first;
* refuses when any published text file contains a home-directory path;
* never commits, pushes or deploys. Publishing stays a deliberate step in the
  site repo: branch, commit, PR, merge.

Usage::

    uv run python scripts/build_dashboard_data.py      # refresh the data first
    uv run python scripts/sync_site.py --check         # report drift; exit 1 if any
    uv run python scripts/sync_site.py                 # mirror it
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "dashboard"
DEFAULT_SITE = Path(
    os.environ.get(
        "CROW_SITE", Path.home() / "projects" / "crow-intelligence.github.io"
    )
)
TARGET = Path("projects") / "parlamonitor-dashboard"

PUBLISHED = ("index.html", "app.js", "rtd.js", "README.md", "assets", "vendor")
"""Tracked files and directories that make up the page, relative to ``dashboard/``."""

DATA = ("data/dashboard.json", "data/vocabulary.json", "data/lines.json")
"""Built by ``scripts/build_dashboard_data.py``; gitignored here, committed there,
because Pages serves only what is in the repo."""

DATA_DIRS = ("data/living",)
"""Built directories, mirrored whole: one file per sitting day for the
"Élő parlament" tab."""

TEXT_SUFFIXES = {".html", ".js", ".json", ".md", ".css", ".svg", ".txt"}
HOME_PATH = re.compile(r"/home/[A-Za-z0-9_.-]+/|/Users/[A-Za-z0-9_.-]+/")
IGNORED_NAMES = {".DS_Store", "__pycache__"}


@dataclass
class Plan:
    """What a sync would do. Paths are relative to the dashboard directory."""

    added: list[str] = field(default_factory=list)
    changed: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        """Whether there is nothing to add, update or delete."""
        return not (self.added or self.changed or self.removed)


def published_files(source: Path) -> list[str]:
    """Every file that belongs to the published page, relative and sorted.

    Raises:
        FileNotFoundError: If a data file is missing -- publishing a page whose
            data was never built would ship a broken dashboard.
    """
    out: set[str] = set()
    for entry in PUBLISHED:
        path = source / entry
        if path.is_dir():
            out.update(
                str(f.relative_to(source))
                for f in path.rglob("*")
                if f.is_file() and not set(f.parts) & IGNORED_NAMES
            )
        elif path.is_file():
            out.add(entry)
        else:
            raise FileNotFoundError(f"{path} is missing from the dashboard source")
    for entry in DATA:
        if not (source / entry).is_file():
            raise FileNotFoundError(
                f"{source / entry} is missing; run scripts/build_dashboard_data.py"
            )
        out.add(entry)
    for entry in DATA_DIRS:
        folder = source / entry
        files = (
            sorted(f for f in folder.glob("*") if f.is_file())
            if folder.is_dir()
            else []
        )
        if not files:
            raise FileNotFoundError(
                f"{folder} is missing or empty; run scripts/build_dashboard_data.py"
            )
        out.update(str(f.relative_to(source)) for f in files)
    return sorted(out)


def plan_sync(source: Path, target: Path) -> Plan:
    """Compare the source file set with the target directory, by content."""
    plan = Plan()
    wanted = published_files(source)
    for rel in wanted:
        there = target / rel
        if not there.exists():
            plan.added.append(rel)
        elif not filecmp.cmp(source / rel, there, shallow=False):
            plan.changed.append(rel)
    if target.is_dir():
        present = {
            str(f.relative_to(target))
            for f in target.rglob("*")
            if f.is_file() and not set(f.parts) & IGNORED_NAMES
        }
        plan.removed = sorted(present - set(wanted))
    return plan


def leaked_paths(source: Path, files: Sequence[str]) -> list[str]:
    """Published text files that mention a home directory."""
    return [
        rel
        for rel in files
        if Path(rel).suffix in TEXT_SUFFIXES
        and HOME_PATH.search(
            (source / rel).read_text(encoding="utf-8", errors="replace")
        )
    ]


def uncommitted(site: Path, target: Path) -> list[str] | None:
    """Uncommitted changes under the target in the site repo; ``None`` if not a repo."""
    try:
        result = subprocess.run(
            ["git", "-C", str(site), "status", "--porcelain", "--", str(target)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


def apply(plan: Plan, source: Path, target: Path) -> None:
    """Carry out a plan: copy added and changed files, delete removed ones."""
    for rel in plan.added + plan.changed:
        (target / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / rel, target / rel)
    for rel in plan.removed:
        (target / rel).unlink()
    # Directories emptied by a removal are left behind otherwise.
    for directory in sorted({(target / r).parent for r in plan.removed}, reverse=True):
        if directory != target and directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def report(plan: Plan, source: Path, target: Path) -> None:
    for label, rels in (
        ("add", plan.added),
        ("update", plan.changed),
        ("delete", plan.removed),
    ):
        for rel in rels:
            path = source / rel if label != "delete" else target / rel
            print(f"  {label:6s} {rel}  ({path.stat().st_size / 1e3:,.1f} kB)")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--site", type=Path, default=DEFAULT_SITE, help="site repo root ($CROW_SITE)"
    )
    parser.add_argument(
        "--check", action="store_true", help="report drift; exit 1 if any"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite even if the site copy has uncommitted changes",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target = args.site / TARGET
    if not args.site.is_dir():
        print(
            f"No site repo at {args.site}; pass --site or set CROW_SITE.",
            file=sys.stderr,
        )
        return 2

    files = published_files(SOURCE)
    leaks = leaked_paths(SOURCE, files)
    if leaks:
        print("Refusing: these files contain a home-directory path:", file=sys.stderr)
        for rel in leaks:
            print(f"  {rel}", file=sys.stderr)
        return 2

    plan = plan_sync(SOURCE, target)
    if plan.clean:
        print(f"In sync: {TARGET} matches dashboard/ ({len(files)} files).")
        return 0
    print(f"dashboard/ -> {args.site.name}/{TARGET}:")
    report(plan, SOURCE, target)
    if args.check:
        return 1

    dirty = uncommitted(args.site, TARGET)
    if dirty is None:
        print("  (not a git repo; cannot check for hand edits)", file=sys.stderr)
    elif dirty and not args.force:
        print(
            "\nRefusing: the site copy has uncommitted changes, which would be "
            "overwritten. Move them into dashboard/ here, or commit/stash them "
            "there, or pass --force:",
            file=sys.stderr,
        )
        for line in dirty:
            print(f"  {line}", file=sys.stderr)
        return 2

    apply(plan, SOURCE, target)
    after = plan_sync(SOURCE, target)
    if not after.clean:
        print("Sync incomplete; files still differ.", file=sys.stderr)
        return 2
    print(
        f"\nMirrored {len(plan.added) + len(plan.changed)} files, "
        f"deleted {len(plan.removed)}. Nothing was committed: review and commit in "
        f"{args.site.name} (branch + PR), as with any site change."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
