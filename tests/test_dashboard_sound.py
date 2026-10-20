"""The sonification's score: ``buildEvents`` in ``dashboard/sound.js``.

The sound itself is checked in the browser (an offline render must not be
silent); here, under node, the timing and the bookkeeping: every event falls
inside the piece, every virtue mention and every reaction is played exactly
once on the layer it belongs to, and the drone never holds two chords at once.
Skipped where node is not installed.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

ROOT = Path(__file__).resolve().parents[1]
SOUND_JS = ROOT / "dashboard" / "sound.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

RUNNER = """
import { buildEvents, CHORDS, LAYERS } from %s;
let input = "";
process.stdin.on("data", d => (input += d));
process.stdin.on("end", () => {
  const out = JSON.parse(input).map(c => {
    const day = c.day, speech = day.speeches[c.speech ?? 0];
    return buildEvents(c.mode, day, speech, c.opts);
  });
  process.stdout.write(JSON.stringify({ out, chords: CHORDS, layers: LAYERS }));
});
"""

VIRTUE_KEYS = [
    "prudence",
    "justice",
    "courage",
    "temperance",
    "truthfulness",
    "magnanimity",
]
LANES = [
    "support",
    "laugh_with",
    "applause",
    "heckle",
    "heckle_hostile",
    "laugh_at",
    "noise",
]


def run_js(cases):
    script = RUNNER % json.dumps(SOUND_JS.as_uri())
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def expected_layers(r):
    """What the reaction should sound on: its lanes, the bell, the closing."""
    if r["closing"]:
        return ["closing"]
    out = list(r["lanes"])
    if "bell" in r["kinds"]:
        out.append("bell")
    if not out and "applause" in r["kinds"]:
        out.append("applause")
    return out


@st.composite
def speeches(draw):
    n = draw(st.integers(1, 12))
    t0 = draw(st.floats(0, 20_000))
    dur = draw(st.floats(1, 1_500))
    times = sorted(draw(st.lists(st.floats(0, 1), max_size=6)))
    reactions = []
    for frac in times:
        lanes = draw(st.lists(st.sampled_from(LANES), max_size=2, unique=True))
        reactions.append(
            {
                "s": draw(st.integers(0, n - 1)),
                "t": t0 + frac * dur,
                "kinds": draw(
                    st.lists(
                        st.sampled_from(["applause", "bell", "laughter", "noise"]),
                        min_size=1,
                        max_size=2,
                        unique=True,
                    )
                ),
                "lanes": lanes,
                "closing": draw(st.booleans()),
                "w": draw(st.sampled_from([0.5, 1.0, 1.5, 2.0, 3.0])),
                "bench": draw(st.sampled_from([None, "government", "opposition"])),
            }
        )
    virtues = [
        [
            s := draw(st.integers(0, n - 1)),
            draw(st.sampled_from(VIRTUE_KEYS)),
            draw(st.sampled_from(["affirming", "accusing"])),
            t0 + dur * (s + 1) / n,
        ]
        for _ in range(draw(st.integers(0, 5)))
    ]
    topic = draw(st.one_of(st.none(), st.integers(0, 27)))
    return {
        "uid": f"u{draw(st.integers(0, 10**9))}",
        "t0": t0,
        "dur": dur,
        "topic": None if topic is None else {"id": topic, "name": f"t{topic}"},
        "sentences": ["x"] * n,
        "virtues": virtues,
        "reactions": reactions,
    }


days = st.lists(speeches(), min_size=1, max_size=6).map(lambda sp: {"speeches": sp})


def check_piece(piece, day, chords, layers):
    events, total = piece["events"], piece["total"]
    times = [e["time"] for e in events]
    assert times == sorted(times)
    assert all(0 <= t <= total + 1e-9 for t in times)
    assert {e["layer"] for e in events} <= set(layers) | {"_speech", "_sentence"}
    drones = [e for e in events if e["layer"] == "drone"]
    for a, b in zip(drones, drones[1:], strict=False):
        assert a["time"] + a["dur"] <= b["time"] + 1e-9, "two chords at once"
    for e in drones:
        assert e["chord"] == chords[e["topic"] % len(chords)]
    return events


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(days, min_size=1, max_size=8), st.floats(30, 600))
def test_a_day_plays_every_mention_and_reaction_once_inside_the_piece(batch, seconds):
    result = run_js(
        [{"mode": "day", "day": d, "opts": {"seconds": seconds}} for d in batch]
    )
    for day, piece in zip(batch, result["out"], strict=True):
        events = check_piece(piece, day, result["chords"], result["layers"])
        assert piece["total"] == pytest.approx(seconds)
        played = sorted(e["layer"] for e in events if e["layer"].startswith("virtue_"))
        wanted = sorted(
            f"virtue_{m[2]}" for sp in day["speeches"] for m in sp["virtues"]
        )
        assert played == wanted
        sounds = sorted(
            e["layer"]
            for e in events
            if not e["layer"].startswith(("virtue_", "_")) and e["layer"] != "drone"
        )
        reactions = [r for sp in day["speeches"] for r in sp["reactions"]]
        assert sounds == sorted(lay for r in reactions for lay in expected_layers(r))
        assert sum(e["layer"] == "_speech" for e in events) == len(day["speeches"])


@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(speeches(), min_size=1, max_size=8), st.floats(0.15, 1.0))
def test_a_speech_plays_one_sentence_per_beat(batch, beat):
    cases = [
        {"mode": "speech", "day": {"speeches": [sp]}, "opts": {"beat": beat}}
        for sp in batch
    ]
    result = run_js(cases)
    for sp, piece in zip(batch, result["out"], strict=True):
        events = check_piece(
            piece, {"speeches": [sp]}, result["chords"], result["layers"]
        )
        n = len(sp["sentences"])
        assert piece["total"] == pytest.approx(n * beat + 1.5)
        ticks = [e["time"] for e in events if e["layer"] == "_sentence"]
        assert ticks == pytest.approx([i * beat for i in range(n)])
        notes = sorted(e["time"] for e in events if e["layer"].startswith("virtue_"))
        assert notes == pytest.approx(sorted(m[0] * beat for m in sp["virtues"]))
        assert sum(e["layer"] == "drone" for e in events) == (sp["topic"] is not None)


def test_a_worked_day():
    """Two speeches, hand-timed: a 100 s sitting played in 10 s."""
    day = {
        "speeches": [
            {
                "uid": "a",
                "t0": 0,
                "dur": 50,
                "topic": {"id": 9, "name": "x"},
                "sentences": ["x", "y"],
                "virtues": [[0, "courage", "affirming", 25.0]],
                "reactions": [
                    {
                        "s": 1,
                        "t": 50.0,
                        "kinds": ["applause"],
                        "lanes": ["applause"],
                        "closing": False,
                        "w": 2.0,
                        "bench": "government",
                    }
                ],
            },
            {
                "uid": "b",
                "t0": 60,
                "dur": 40,
                "topic": None,
                "sentences": ["z"],
                "virtues": [[0, "truthfulness", "accusing", 100.0]],
                "reactions": [],
            },
        ]
    }
    [piece] = run_js([{"mode": "day", "day": day, "opts": {"seconds": 10}}])["out"]
    sound = [
        (e["time"], e["layer"], e.get("note"), e.get("pan"))
        for e in piece["events"]
        if not e["layer"].startswith("_")
    ]
    assert sound == [
        (0.0, "drone", None, None),
        (2.5, "virtue_affirming", "A5", None),
        (5.0, "applause", None, -0.6),
        (10.0, "virtue_accusing", "E3", None),
    ]
    [drone] = [e for e in piece["events"] if e["layer"] == "drone"]
    assert drone["dur"] == 6.0  # held to the next speech, which has no topic
    assert drone["chord"] == ["F2", "C3", "A3"]  # topic 9 -> chord 9 % 8 = 1
