"""The site sync mirrors exactly the published set, and refuses unsafe copies."""

import importlib.util
import sys
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "sync_site", Path(__file__).resolve().parents[1] / "scripts" / "sync_site.py"
)
sync_site = importlib.util.module_from_spec(SPEC)
sys.modules["sync_site"] = sync_site  # @dataclass resolves its module through here
SPEC.loader.exec_module(sync_site)


@pytest.fixture
def source(tmp_path):
    src = tmp_path / "dashboard"
    for rel, text in {
        "index.html": "<html></html>",
        "app.js": "export {}",
        "rtd.js": "export {}",
        "README.md": "readme",
        "assets/og.png": "png",
        "vendor/d3.js": "d3",
        "data/dashboard.json": "{}",
        "data/vocabulary.json": "{}",
        "data/lines.json": "{}",
        "data/scratch.csv": "not published",
    }.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text, encoding="utf-8")
    return src


def test_only_the_published_set_is_mirrored(source):
    files = sync_site.published_files(source)
    assert "data/scratch.csv" not in files
    assert {"index.html", "rtd.js", "vendor/d3.js", "data/vocabulary.json"} <= set(
        files
    )


def test_a_missing_data_file_is_refused(source):
    (source / "data" / "vocabulary.json").unlink()
    with pytest.raises(FileNotFoundError, match="build_dashboard_data"):
        sync_site.published_files(source)


def test_sync_adds_updates_and_deletes_until_identical(source, tmp_path):
    target = tmp_path / "site" / "projects" / "parlamonitor-dashboard"
    (target / "vendor").mkdir(parents=True)
    (target / "index.html").write_text("<html>hand edit</html>", encoding="utf-8")
    (target / "vendor" / "old.js").write_text("stale", encoding="utf-8")

    plan = sync_site.plan_sync(source, target)
    assert plan.changed == ["index.html"]
    assert plan.removed == ["vendor/old.js"]
    assert "rtd.js" in plan.added

    sync_site.apply(plan, source, target)
    assert sync_site.plan_sync(source, target).clean
    assert (target / "index.html").read_text(encoding="utf-8") == "<html></html>"


def test_a_home_directory_path_is_caught(source):
    (source / "data" / "dashboard.json").write_text(
        '{"source": "/home/someone/x"}', encoding="utf-8"
    )
    assert sync_site.leaked_paths(source, sync_site.published_files(source)) == [
        "data/dashboard.json"
    ]
