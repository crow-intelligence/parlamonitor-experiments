"""Remove parliamentary register vocabulary from the emotion dictionaries.

Putz Orsolya's collection is built for general Hungarian. This corpus is
parliamentary, and a handful of its entries mean something else here: ``vita``
is the name of a procedure, ``elfogadás`` is adopting a bill, ``fél`` is as
often "half" or "the other party" as it is "fears". Left in, ``jó``,
``támogatás``, ``kedves`` and ``segít`` alone made joy the dominant emotion in
62% of speeches.

Every removal below was checked against real usage in the corpus, not chosen by
frequency. The reason is recorded per word, the originals are backed up beside
the curated files, and the removals are written to a committed record so the
divergence from her collection is auditable and reversible.

This edits the working copies only. Her collection is untouched; re-running
with ``--restore`` puts the files back.

Usage::

    uv run python scripts/curate_emotion_lexicons.py
    uv run python scripts/curate_emotion_lexicons.py --restore
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from parlamonitor.lexicon import EKMAN_FILES

ROOT = Path(__file__).resolve().parents[1]
EMOTION_DIR = ROOT / "data" / "lexicons" / "emotion"

# word -> why it is not that emotion in this corpus. Checked in context.
REMOVE: dict[str, dict[str, str]] = {
    "anger": {
        "vita": "the name of the procedure — 'általános vita', 785 hits",
        "húz": "literal: 'magával húzni', 'munkásszállókat húztak fel'",
        "félreértés": "a misunderstanding is not an emotion",
    },
    "disgust": {
        "rossz": "ordinary evaluative adjective, 242 hits",
    },
    "fear": {
        "fél": (
            "three words at once: 'fears', 'half' ('fél ország') and 'party' "
            "('a másik fél'). One sentence in the corpus uses two of them"
        ),
        "kegyelem": "legal: 'kegyelmi kérvény', not fear",
    },
    "joy": {
        "jó": "ordinary evaluative adjective, 1,741 hits",
        "támogatás": "policy noun — a subsidy, 611 hits",
        "kedves": "the salutation 'Kedves Képviselőtársaim', 409 hits",
        "segít": "policy verb, 393 hits",
        "szabadság": "political abstraction, and also 'holiday'",
        "emberi": "'emberi jogok', 'emberi méltóság'",
        "szíves": "the formula 'szíves tájékoztatás'",
        "tisztességes": "fairness rather than joy; belongs to justice if anywhere",
        "nyer": "mostly the formal 'megerősítést nyert', and 'választást nyerni'",
    },
    "sadness": {
        "elfogadás": "adopting a bill — pure procedure, 186 hits",
        "szegény": "poverty as an economic category, not pity",
        "negatív": "technical usage",
    },
    "surprise": {
        "rendkívüli": "'rendkívüli ülés', 'rendkívüli jogrend' — administrative",
        "véletlen": "mostly the rhetorical 'nem véletlen, hogy'",
    },
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--emotion-dir", type=Path, default=EMOTION_DIR)
    parser.add_argument(
        "--restore", action="store_true", help="put the original files back"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    directory: Path = args.emotion_dir

    if args.restore:
        restored = 0
        for backup in sorted(directory.glob("*.orig")):
            shutil.copy2(backup, backup.with_suffix(""))
            restored += 1
        print(f"restored {restored} files from .orig backups")
        return 0

    record = {
        "curated_at": datetime.now(UTC).isoformat(),
        "source": "Putz Orsolya's collection (unmodified originals kept as *.orig)",
        "rationale": (
            "The lists are built for general Hungarian; this corpus is "
            "parliamentary. Every removal was checked against usage in the "
            "corpus, not chosen by frequency."
        ),
        "removed": {},
    }

    for emotion, removals in REMOVE.items():
        path = directory / EKMAN_FILES[emotion]
        backup = path.with_suffix(path.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(path, backup)

        original = backup.read_text(encoding="utf-8").splitlines()
        kept, dropped = [], []
        for line in original:
            entry = line.strip()
            if entry.casefold() in removals:
                dropped.append(entry)
            else:
                kept.append(line)

        path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        missing = sorted(set(removals) - {d.casefold() for d in dropped})
        record["removed"][emotion] = {
            "file": path.name,
            "before": len(original),
            "after": len(kept),
            "dropped": {w: removals[w] for w in sorted(removals) if w not in missing},
            "not_found": missing,
        }
        note = f" (not in file: {', '.join(missing)})" if missing else ""
        counts = f"{len(original):4d} -> {len(kept):4d}"
        print(f"  {emotion:9s} {counts}  -{len(dropped)}{note}")

    (directory / "curation.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(len(v["dropped"]) for v in record["removed"].values())
    print(f"\n{total} entries removed; record in {directory / 'curation.json'}")
    print("Originals kept as *.orig — rerun with --restore to undo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
