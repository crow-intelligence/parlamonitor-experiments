import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from parlamonitor.loading import (
    iter_jsonl,
    load_manifest,
    load_qa,
    load_speeches,
    provenance,
)


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    return path


def test_iter_jsonl_preserves_order(tmp_path):
    records = [{"uid": f"43003-{i}"} for i in range(5)]
    path = write_jsonl(tmp_path / "toy.jsonl", records)
    assert list(iter_jsonl(path)) == records


def test_iter_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "toy.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(iter_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_iter_jsonl_names_the_bad_line(tmp_path):
    path = tmp_path / "toy.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\noops\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3"):
        list(iter_jsonl(path))


def test_iter_jsonl_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl(tmp_path / "absent.jsonl"))


def test_iter_jsonl_is_lazy(tmp_path):
    """A malformed tail must not stop the good head from being read."""
    path = tmp_path / "toy.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    stream = iter_jsonl(path)
    assert next(stream) == {"a": 1}


def test_loaders_use_the_given_directory(tmp_path):
    write_jsonl(tmp_path / "cycle43-speeches.jsonl", [{"uid": "s"}])
    write_jsonl(tmp_path / "cycle43-qa.jsonl", [{"exchange_id": "q"}])
    (tmp_path / "cycle43-manifest.json").write_text(
        json.dumps({"period_number": 43}), encoding="utf-8"
    )

    assert load_speeches(tmp_path) == [{"uid": "s"}]
    assert load_qa(tmp_path) == [{"exchange_id": "q"}]
    assert load_manifest(tmp_path)["period_number"] == 43


def test_provenance_reports_sizes_and_absences(tmp_path):
    manifest = {"period_number": 43, "db_data_updated_at": "2026-07-29T22:34:01Z"}
    (tmp_path / "cycle43-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    write_jsonl(tmp_path / "cycle43-speeches.jsonl", [{"uid": "s"}])
    # cycle43-qa.jsonl deliberately absent: missing is None, not zero.

    record = provenance(tmp_path)

    assert record["period_number"] == 43
    assert record["db_data_updated_at"] == "2026-07-29T22:34:01Z"
    assert record["manifest"] == manifest
    assert record["files"]["cycle43-speeches.jsonl"] > 0
    assert record["files"]["cycle43-qa.jsonl"] is None


def test_provenance_without_manifest_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        provenance(tmp_path)


@given(
    st.lists(
        st.dictionaries(
            st.text(min_size=1, max_size=8),
            st.one_of(st.integers(), st.text(max_size=32), st.none()),
            max_size=4,
        ),
        max_size=20,
    )
)
def test_jsonl_roundtrip(tmp_path_factory, records):
    """Anything JSON-encodable one-per-line comes back unchanged."""
    path = write_jsonl(tmp_path_factory.mktemp("jsonl") / "rt.jsonl", records)
    assert list(iter_jsonl(path)) == records
