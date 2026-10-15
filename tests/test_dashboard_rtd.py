"""The dashboard's in-browser RTD must give keyflux's numbers.

``dashboard/rtd.js`` re-implements rank-turbulence divergence so that any two
speakers can be compared without precomputing every pair. This runs it under
node on the same inputs as :func:`keyflux.rtd` and compares the divergence and
every type's ranks and contribution. Skipped where node is not installed.
"""

import json
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from keyflux import RankedList, rtd

ROOT = Path(__file__).resolve().parents[1]
RTD_JS = ROOT / "dashboard" / "rtd.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs node")

RUNNER = """
import { rtd } from %s;
let input = "";
process.stdin.on("data", d => (input += d));
process.stdin.on("end", () => {
  const out = JSON.parse(input).map(c => {
    const r = rtd(new Map(Object.entries(c.a)), new Map(Object.entries(c.b)), c.alpha);
    const types = r.types.map(t => [t.t, t.r1, t.r2, t.contrib, t.dir]);
    return { divergence: r.divergence, types };
  });
  process.stdout.write(JSON.stringify(out));
});
"""


def run_js(cases):
    script = RUNNER % json.dumps(RTD_JS.as_uri())
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def random_counts(rng, vocabulary, n_types, max_count):
    words = rng.sample(vocabulary, n_types)
    return {w: rng.randint(1, max_count) for w in words}


def cases():
    rng = random.Random(0)
    vocabulary = [f"w{i}" for i in range(400)] + ["árvíztűrő", "ő", "Ő", "z"]
    out = [
        # identical, disjoint, single types, heavy ties, the alpha limits
        {"a": {"x": 3, "y": 2}, "b": {"x": 3, "y": 2}, "alpha": 1 / 3},
        {"a": {"x": 3, "y": 2}, "b": {"p": 5, "q": 1}, "alpha": 1 / 3},
        {"a": {"x": 1}, "b": {"y": 1}, "alpha": 1.0},
        {
            "a": {w: 1 for w in vocabulary[:50]},
            "b": {w: 1 for w in vocabulary[25:90]},
            "alpha": 1 / 3,
        },
        {
            "a": {"a": 20, "e": 14, "c": 8, "b": 7, "f": 4, "g": 2, "d": 1},
            "b": {"b": 24, "a": 16, "e": 5, "d": 4, "c": 3, "f": 2, "g": 1},
            "alpha": 1.0,
        },
    ]
    for alpha in (0.0, 1 / 3, 1.0, 3.0):
        for _ in range(25):
            out.append(
                {
                    "a": random_counts(
                        rng, vocabulary, rng.randint(1, 300), rng.choice([3, 40, 2000])
                    ),
                    "b": random_counts(
                        rng, vocabulary, rng.randint(1, 300), rng.choice([3, 40, 2000])
                    ),
                    "alpha": alpha,
                }
            )
    return out


def test_browser_rtd_matches_keyflux():
    inputs = cases()
    outputs = run_js(inputs)
    assert len(outputs) == len(inputs)
    for case, js in zip(inputs, outputs, strict=True):
        ref = rtd(
            RankedList.from_counts(case["a"]),
            RankedList.from_counts(case["b"]),
            alpha=case["alpha"],
        )
        assert js["divergence"] == pytest.approx(ref.divergence, rel=1e-12, abs=1e-14)
        expected = {
            c.type: (
                c.rank1,
                c.rank2,
                c.contribution,
                {"system1": "left", "system2": "right"}.get(c.direction, "shared"),
            )
            for c in ref.contributions
        }
        got = {t: (r1, r2, contrib, d) for t, r1, r2, contrib, d in js["types"]}
        assert got.keys() == expected.keys()
        for t, (r1, r2, contrib, d) in got.items():
            e1, e2, econtrib, ed = expected[t]
            assert (r1, r2, d) == (e1, e2, ed), t
            assert contrib == pytest.approx(econtrib, rel=1e-9, abs=1e-15), t


def test_the_keyflux_reference_example():
    [js] = run_js([cases()[4]])
    assert round(js["divergence"], 6) == 0.459248
