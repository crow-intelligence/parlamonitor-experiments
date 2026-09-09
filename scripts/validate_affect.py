"""Convergent validity of the affect scores, computed from the corpus.

Short calibration probes establish which index carries which label. They do
not establish that a channel *works*: a four-sentence probe is nothing like a
twenty-minute speech, and this corpus showed the two tests disagreeing in both
directions. The Hungarian model's anger channel failed its probe and behaves
sensibly on real speeches; its fear channel passed the probe and turns out to
fire on two thirds of everything while correlating with nothing.

So reliability is decided here, on the scored corpus, by three checks:

* **Valence agreement.** An anger or sadness channel should correlate
  negatively with sentiment valence, a joy channel positively. A channel near
  zero is not tracking affect.
* **Cross-model agreement.** The two emotion models share three labels. Where
  they disagree entirely, at least one is wrong.
* **Saturation.** A channel with a very high mean and low variance is a
  default, not a discrimination.

Reads ``speech_affect.csv``; runs no models, so it is cheap to repeat.

Usage::

    uv run python scripts/validate_affect.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from parlamonitor.affect import UNRELIABLE_CHANNELS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "data" / "derived" / "metrics"

MIN_VALENCE_CORRELATION = 0.15
"""Below this absolute correlation with valence, a channel tracks nothing.

Deliberately lenient. Emotion and sentiment are related but not the same, so a
genuine channel need not correlate strongly -- but one at r = 0.04 across 1,693
documents is not measuring affect at all.
"""

MAX_MEAN_FOR_DISCRIMINATION = 0.55
"""Above this mean, a channel is firing on most of the corpus.

Combined with a failed valence check this identifies a saturated default. On
its own it is not disqualifying: a genuinely uniform corpus could produce it.
"""

EXPECTED_SIGN = {
    "anger": -1,
    "fear": -1,
    "disgust": -1,
    "sadness": -1,
    "joy": +1,
    "none": 0,
}
"""Which way each emotion should move with sentiment valence.

``none`` has no expected direction and is checked only for saturation.
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_DIR)
    return parser.parse_args(argv)


def emotion_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("emotion_")]


def validate(frame: pd.DataFrame) -> dict:
    """Score every emotion channel against the three checks."""
    report = {}
    for column in emotion_columns(frame):
        model, label = column.removeprefix("emotion_").split("_", 1)
        correlation = float(frame[column].corr(frame["sentiment_valence"]))
        mean = float(frame[column].mean())
        sd = float(frame[column].std())
        expected = EXPECTED_SIGN.get(label, 0)

        if expected == 0:
            valence_ok = None
        else:
            valence_ok = abs(correlation) >= MIN_VALENCE_CORRELATION and (
                correlation > 0
            ) == (expected > 0)
        saturated = mean > MAX_MEAN_FOR_DISCRIMINATION
        report[column] = {
            "model": model,
            "label": label,
            "mean": round(mean, 4),
            "sd": round(sd, 4),
            "valence_correlation": round(correlation, 4),
            "expected_sign": expected,
            "valence_check": valence_ok,
            "saturated": saturated,
            "reliable": valence_ok is not False and not (saturated and not valence_ok),
        }
    return report


def cross_model(frame: pd.DataFrame) -> dict:
    """Correlate the labels the two emotion models share."""
    shared = {}
    for label in ("anger", "fear", "sadness", "joy"):
        hu, xlm = f"emotion_hu_{label}", f"emotion_xlm_{label}"
        if hu in frame.columns and xlm in frame.columns:
            shared[label] = round(float(frame[hu].corr(frame[xlm])), 4)
    return shared


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    path = args.metrics_dir / "speech_affect.csv"
    frame = pd.read_csv(path)
    print(f"{len(frame):,} scored speeches from {path.name}")

    report = validate(frame)
    shared = cross_model(frame)

    print("\nchannel                  mean     sd    r(valence)  checks")
    for column, row in report.items():
        flags = []
        if row["valence_check"] is False:
            flags.append("no-valence-signal")
        if row["saturated"]:
            flags.append("saturated")
        verdict = "ok" if row["reliable"] else "UNRELIABLE"
        print(
            f"  {column:24s} {row['mean']:.3f}  {row['sd']:.3f}   "
            f"{row['valence_correlation']:+.3f}     {verdict}"
            + (f"  [{', '.join(flags)}]" if flags else "")
        )

    print("\ncross-model agreement on shared labels:")
    for label, r in shared.items():
        print(f"  {label:8s} r={r:+.3f}")

    flagged = {c for c, row in report.items() if not row["reliable"]}
    print(f"\nflagged unreliable: {sorted(flagged) or 'none'}")
    if flagged != set(UNRELIABLE_CHANNELS):
        print(
            f"  note: parlamonitor.affect.UNRELIABLE_CHANNELS lists "
            f"{sorted(UNRELIABLE_CHANNELS)}; update it to match."
        )

    out = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": str(path),
        "n_speeches": len(frame),
        "thresholds": {
            "min_valence_correlation": MIN_VALENCE_CORRELATION,
            "max_mean_for_discrimination": MAX_MEAN_FOR_DISCRIMINATION,
        },
        "channels": report,
        "cross_model_agreement": shared,
        "flagged_unreliable": sorted(flagged),
        "declared_unreliable": sorted(UNRELIABLE_CHANNELS),
        "agrees_with_declaration": flagged == set(UNRELIABLE_CHANNELS),
    }
    (args.metrics_dir / "affect_validation.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.metrics_dir / 'affect_validation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
