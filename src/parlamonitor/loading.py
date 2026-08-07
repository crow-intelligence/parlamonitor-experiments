"""Readers for the cycle-43 parliamentary exports in ``data/raw/``.

The readers do not reshape, filter, or normalise anything: a record comes back
exactly as the export wrote it, so the chain from ``data/raw/`` to a number
stays a straight line. Filtering that *changes the numbers* -- a minimum word
count for length-sensitive lexical diversity, dropping incomplete Q&A pairs --
is the caller's decision and is made at the call site, not hidden in here.

The exports themselves are already filtered upstream: procedural speeches
(2,114), the MP oath (8), and speeches with no published transcript (154) are
absent. :func:`load_manifest` returns those counts verbatim.

See ``data/raw/README.md`` for the field-by-field schema. Use ``text_clean``
or ``sentences`` for measurement; ``text`` still carries the speaker
attribution and the editorial stage directions.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# src/parlamonitor/loading.py -> src/parlamonitor -> src -> repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW: Path = Path(os.environ.get("PARLAMONITOR_DATA", _REPO_ROOT / "data" / "raw"))
"""Directory holding the raw exports.

Resolved from the ``PARLAMONITOR_DATA`` environment variable when set,
otherwise from the repository layout (``<repo>/data/raw``). The layout
fallback assumes an editable install, which is what ``uv sync`` produces; set
the environment variable if the package is installed elsewhere.
"""

SPEECHES_FILE = "cycle43-speeches.jsonl"
QA_FILE = "cycle43-qa.jsonl"
MANIFEST_FILE = "cycle43-manifest.json"


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    r"""Yield each line of a JSONL file as a decoded object.

    Blank lines are skipped. Nothing is cached, so this streams a 22 MB export
    without holding it in memory.

    Args:
        path: Path to a UTF-8 JSONL file, one JSON object per line.

    Yields:
        One decoded object per non-blank line, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line is not valid JSON. The message carries the
            1-based line number, which :class:`json.JSONDecodeError` cannot
            report because each line is decoded on its own.

    Example:
        >>> import pathlib, tempfile
        >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "toy.jsonl"
        >>> _ = tmp.write_text(
        ...     '{"uid": "43003-1"}\n\n{"uid": "43003-2"}\n', encoding="utf-8"
        ... )
        >>> [record["uid"] for record in iter_jsonl(tmp)]
        ['43003-1', '43003-2']

        A malformed line names itself:

        >>> _ = tmp.write_text('{"uid": "ok"}\nnot json\n', encoding="utf-8")
        >>> list(iter_jsonl(tmp))
        Traceback (most recent call last):
            ...
        ValueError: malformed JSON on line 2 of toy.jsonl
    """
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"malformed JSON on line {lineno} of {path.name}"
                raise ValueError(msg) from exc


def load_speeches(data_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Load every speech record from the cycle-43 export.

    Args:
        data_dir: Directory holding the export. Defaults to :data:`DATA_RAW`.

    Returns:
        The 1,693 speech records, in file order, unfiltered.

    Raises:
        FileNotFoundError: If the export is missing. It is gitignored -- copy
            it into ``data/raw/`` or point ``PARLAMONITOR_DATA`` at it.

    Note:
        Plain type-token ratio is length-dependent and these records span 1 to
        5,281 words, so a length filter (``n_words >= 200``) or a
        length-robust measure (MATTR, MTLD) is required for lexical diversity.
        Neither is applied here; ``speech_type`` correlates strongly with
        length, since ``kétperces felszólalás`` is a two-minute format by rule.

    Example:
        >>> speeches = load_speeches()  # doctest: +SKIP
        >>> len(speeches)  # doctest: +SKIP
        1693
    """
    return list(iter_jsonl(_resolve(data_dir) / SPEECHES_FILE))


def load_qa(data_dir: Path | str | None = None) -> list[dict[str, Any]]:
    """Load every question-answer exchange from the cycle-43 export.

    Args:
        data_dir: Directory holding the export. Defaults to :data:`DATA_RAW`.

    Returns:
        All 215 exchanges, in file order, including the 13 whose
        ``text_complete`` is ``False``.

    Raises:
        FileNotFoundError: If the export is missing.

    Note:
        The 13 incomplete rows are genuine gaps, not export bugs: 12 on sitting
        43020, whose transcript is not published yet, and one where the
        minister's microphone failed. Filter on ``text_complete`` for any text
        work -- that leaves 202. They are kept in the return value so a caller
        counting coverage can see them.

    Example:
        >>> exchanges = load_qa()  # doctest: +SKIP
        >>> sum(x["text_complete"] for x in exchanges)  # doctest: +SKIP
        202
    """
    return list(iter_jsonl(_resolve(data_dir) / QA_FILE))


def load_manifest(data_dir: Path | str | None = None) -> dict[str, Any]:
    """Load the export manifest: row counts and what was filtered out.

    Args:
        data_dir: Directory holding the export. Defaults to :data:`DATA_RAW`.

    Returns:
        The manifest verbatim -- source database, ``db_data_updated_at``, row
        counts, and the per-reason skip counts for both datasets.

    Raises:
        FileNotFoundError: If the manifest is missing.

    Example:
        >>> load_manifest()["period_number"]  # doctest: +SKIP
        43
    """
    path = _resolve(data_dir) / MANIFEST_FILE
    return json.loads(path.read_text(encoding="utf-8"))


def provenance(data_dir: Path | str | None = None) -> dict[str, Any]:
    """Build a reproducibility record for whatever the exports currently are.

    Pair this with any published number so the result names its own source.
    It reports the manifest's own account of the data alongside the byte size
    of each file actually on disk, which is how a silently truncated or
    re-synced export gets caught.

    Args:
        data_dir: Directory holding the export. Defaults to :data:`DATA_RAW`.

    Returns:
        A dict with ``data_dir``, ``period_number``, ``db_data_updated_at``,
        ``manifest`` (the full manifest), and ``files`` mapping each export
        filename to its size in bytes, or ``None`` when the file is absent.

    Raises:
        FileNotFoundError: If the manifest is missing. Sizes for the two JSONL
            exports are reported as ``None`` rather than raising, since the
            manifest alone is enough to describe an export that has not been
            copied in yet.

    Example:
        >>> record = provenance()  # doctest: +SKIP
        >>> record["db_data_updated_at"]  # doctest: +SKIP
        '2026-07-29T22:34:01.486174+00:00'
    """
    directory = _resolve(data_dir)
    manifest = load_manifest(directory)
    files: dict[str, int | None] = {}
    for name in (SPEECHES_FILE, QA_FILE, MANIFEST_FILE):
        candidate = directory / name
        files[name] = candidate.stat().st_size if candidate.exists() else None
    return {
        "data_dir": str(directory),
        "period_number": manifest.get("period_number"),
        "db_data_updated_at": manifest.get("db_data_updated_at"),
        "manifest": manifest,
        "files": files,
    }


def _resolve(data_dir: Path | str | None) -> Path:
    """Return ``data_dir`` as a path, falling back to :data:`DATA_RAW`."""
    return DATA_RAW if data_dir is None else Path(data_dir)
