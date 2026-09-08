"""The note-takers' parentheticals: five cycles of chamber stage directions.

``data/raw/parantheticals/`` holds one plain-text file per parliamentary cycle
(39--43). Each line is one bracketed remark as the shorthand writers recorded
it -- ``Taps a kormánypártok soraiban.``, ``Derültség.``, ``Az elnök
csenget.``, ``Közbeszólás a Jobbik soraiból: Ne már!`` -- with no metadata
column of any kind. The cycle is the filename and that is the whole schema.

Two properties of the files govern how they may be counted.

**Duplication is heavy and very uneven.** 26% of cycle 41's lines are
consecutive repeats, against about 6% elsewhere; ``Folyamatos sípolás.``
alone occurs 15,840 times there. Whether those repeats are genuine (the
whistle protest really did go on) or an artefact of how the text was pulled
out of the transcripts cannot be settled from the files themselves, so this
module refuses to choose: :func:`collapse_consecutive` provides the second
reading and :class:`LineStats` measures the gap between them.

**Most lines are not distinct.** 64,052 of 261,935 lines are, corpus-wide,
which is what makes it affordable to send every distinct line through emtsv
once and expand the result by count.

Nothing here rewrites the raw text. The files in ``data/raw/`` are the
provenance record and are read exactly as they are.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from parlamonitor.loading import DATA_RAW

CYCLES: tuple[int, ...] = (39, 40, 41, 42, 43)
"""The parliamentary cycles exported to ``data/raw/parantheticals/``."""

SUBDIRECTORY = "parantheticals"
"""Directory name under ``data/raw``. Spelled as the export spells it."""

AGGREGATE_LABEL = "ALL"
"""The ``cycle`` value used for rows pooling every cycle."""

SOFT_HYPHEN = "\u00ad"
"""U+00AD, the zero-width hyphenation hint left behind by the PDF extraction."""

NORMALISATION_VERSION = "strip-soft-hyphen-1"
"""Identifier for the normalisation applied before analysis, for cache keys.

Bump this whenever :func:`strip_soft_hyphens` changes behaviour.
"""


@dataclass(frozen=True, slots=True)
class LineStats:
    """How much text a cycle holds, and how much of it repeats.

    Attributes:
        cycle: The parliamentary cycle.
        path: The file these numbers were read from.
        sha256: Hex digest of the file's bytes, for the reproducibility record.
        n_lines: Lines in the file, one parenthetical each.
        n_words: Whitespace-separated words across those lines.
        n_distinct: Distinct line strings.
        n_consecutive_repeats: Lines identical to the line immediately before
            them. This is ``n_lines`` minus the length of
            :func:`collapse_consecutive`'s output.
    """

    cycle: int
    path: Path
    sha256: str
    n_lines: int
    n_words: int
    n_distinct: int
    n_consecutive_repeats: int

    @property
    def repeat_fraction(self) -> float:
        """Share of lines that repeat the line before them, in ``[0, 1)``.

        Returns ``0.0`` for an empty file rather than dividing by zero.

        Example:
            >>> stats = LineStats(43, Path("x"), "", 100, 700, 40, 25)
            >>> stats.repeat_fraction
            0.25
        """
        return self.n_consecutive_repeats / self.n_lines if self.n_lines else 0.0


def parentheticals_dir(data_dir: Path | str | None = None) -> Path:
    """Locate the parentheticals directory.

    Args:
        data_dir: Root of the raw data. Defaults to
            :data:`parlamonitor.loading.DATA_RAW`, which honours the
            ``PARLAMONITOR_DATA`` environment variable.

    Returns:
        The directory holding ``cycle{N}-parentheticals.txt``.
    """
    root = DATA_RAW if data_dir is None else Path(data_dir)
    return root / SUBDIRECTORY


def parentheticals_path(cycle: int, data_dir: Path | str | None = None) -> Path:
    """Return the path of one cycle's file, whether or not it exists.

    Args:
        cycle: The parliamentary cycle.
        data_dir: Root of the raw data. See :func:`parentheticals_dir`.

    Returns:
        The path ``<data_dir>/parantheticals/cycle<N>-parentheticals.txt``.
    """
    return parentheticals_dir(data_dir) / f"cycle{cycle}-parentheticals.txt"


def load_parentheticals(
    cycle: int,
    data_dir: Path | str | None = None,
    *,
    limit: int | None = None,
    normalise: bool = True,
) -> tuple[list[str], int]:
    """Read one cycle's parentheticals, one per line.

    Trailing newlines are stripped; nothing else is touched. Blank lines are
    dropped -- the exports contain none, so a blank line would be a defect in
    the file rather than a parenthetical with no content.

    Args:
        cycle: The parliamentary cycle.
        data_dir: Root of the raw data. See :func:`parentheticals_dir`.
        limit: Keep only the first this-many lines. For smoke runs; ``None``
            (the default) reads the file.
        normalise: Whether to strip soft hyphens. Defaults to ``True``. See
            :func:`strip_soft_hyphens` -- leaving them in corrupts 371 tokens
            of cycle 40. Pass ``False`` to read the file verbatim.

    Returns:
        A ``(lines, n_soft_hyphens_removed)`` pair, the lines in file order.

    Raises:
        FileNotFoundError: If the cycle has no export. Raised rather than
            returning ``[]``, so a typo in a cycle number cannot quietly
            produce a corpus that is missing a fifth of itself.
    """
    path = parentheticals_path(cycle, data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"no parentheticals export for cycle {cycle}: {path}")
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if limit is not None:
        lines = lines[:limit]
    if not normalise:
        return lines, 0
    n_removed = 0
    normalised = []
    for line in lines:
        line, removed = strip_soft_hyphens(line)
        n_removed += removed
        normalised.append(line)
    return normalised, n_removed


def strip_soft_hyphens(text: str) -> tuple[str, int]:
    """Remove U+00AD soft hyphens, rejoining the words they were splitting.

    Cycle 40 carries 1,255 of them and no other cycle carries any -- a trace of
    how that transcript was extracted. A soft hyphen is a rendering hint with
    no width and no linguistic content, but emtsv does not treat it as one: it
    replaces the character with U+FFFD and tokenises around it, so
    ``hát\u00adoldalán`` ("its back side") comes back as three tokens,
    ``hát``, ``\ufffd``, ``oldalán``. That produced 371 U+FFFD tokens in the
    first run of this analysis and split 885 lines' words into fragments.

    Removing the character is lossless -- the intended word is the one without
    it -- but it does change the counts, so it is versioned by
    :data:`NORMALISATION_VERSION` and the number removed is reported.

    Args:
        text: A line as the file holds it.

    Returns:
        A ``(text, n_removed)`` pair.

    Example:
        >>> strip_soft_hyphens("hát\u00adoldalán")
        ('hátoldalán', 1)
        >>> strip_soft_hyphens("Taps.")
        ('Taps.', 0)
    """
    n_removed = text.count(SOFT_HYPHEN)
    return (text.replace(SOFT_HYPHEN, ""), n_removed) if n_removed else (text, 0)


def collapse_consecutive(lines: Sequence[str]) -> list[str]:
    """Collapse each run of identical adjacent lines to a single line.

    The second reading of the corpus. It removes the block-level duplication
    visible at the head of cycle 42, where a four-line group appears three
    times verbatim, and it flattens the 15,840 consecutive ``Folyamatos
    sípolás.`` lines of cycle 41 to the number of distinct bursts. It also
    removes genuine repetition -- two separate rounds of applause recorded
    back to back become one -- which is exactly why it is reported alongside
    the raw counts and not instead of them.

    Args:
        lines: The lines, in file order.

    Returns:
        The lines with adjacent duplicates removed. Non-adjacent repeats are
        untouched.

    Example:
        >>> collapse_consecutive(["Taps.", "Taps.", "Derültség.", "Taps."])
        ['Taps.', 'Derültség.', 'Taps.']
        >>> collapse_consecutive([])
        []
    """
    return [line for i, line in enumerate(lines) if i == 0 or line != lines[i - 1]]


def line_stats(
    cycle: int, lines: Sequence[str], path: Path, *, sha256: str = ""
) -> LineStats:
    """Measure one cycle's lines.

    Args:
        cycle: The parliamentary cycle.
        lines: The lines, in file order.
        path: Where they were read from.
        sha256: Hex digest of the file, if already computed. Defaults to
            empty, meaning "not recorded".

    Returns:
        A :class:`LineStats`.

    Example:
        >>> stats = line_stats(43, ["Taps.", "Taps.", "Nagy taps."], Path("x"))
        >>> stats.n_lines, stats.n_words, stats.n_distinct
        (3, 4, 2)
        >>> stats.n_consecutive_repeats
        1
    """
    return LineStats(
        cycle=cycle,
        path=path,
        sha256=sha256,
        n_lines=len(lines),
        n_words=sum(len(line.split()) for line in lines),
        n_distinct=len(set(lines)),
        n_consecutive_repeats=len(lines) - len(collapse_consecutive(lines)),
    )


def file_sha256(path: Path) -> str:
    """Hash a file's bytes, for the reproducibility record.

    Args:
        path: The file to hash.

    Returns:
        The hex digest.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
