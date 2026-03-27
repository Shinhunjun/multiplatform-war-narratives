from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import snapshot_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_source_files(directory: Path, names: list[str]) -> list[Path]:
    """Create small dummy files and return their paths."""
    paths = []
    for name in names:
        p = directory / name
        p.write_text(f"dummy content for {name}")
        paths.append(p)
    return paths


SOURCE_NAMES = [
    "gdelt_scraped.csv",
    "url_lookup.csv",
    "url_filter_eval.csv",
    "url_filter_summary_counts.csv",
    "analysis_events.parquet",
    "analysis_url_content.parquet",
]


# ---------------------------------------------------------------------------
# fmt_size
# ---------------------------------------------------------------------------

def test_fmt_size_bytes() -> None:
    path = Path(__file__)
    # Just check it returns a string with a unit suffix.
    result = snapshot_data.fmt_size(path)
    assert any(unit in result for unit in ("B", "KB", "MB", "GB"))


# ---------------------------------------------------------------------------
# snapshot creation
# ---------------------------------------------------------------------------

def test_copies_all_six_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    snap_dir = tmp_path / "snapshots"

    source_files = make_source_files(src_dir, SOURCE_NAMES)

    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", source_files)
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_data, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["snapshot_data.py"]):
        snapshot_data.main()

    created = list(snap_dir.iterdir())
    assert len(created) == 1
    snap = created[0]

    for name in SOURCE_NAMES:
        assert (snap / name).exists(), f"{name} was not copied"


def test_snapshot_dir_name_uses_today_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    snap_dir = tmp_path / "snapshots"
    source_files = make_source_files(src_dir, SOURCE_NAMES)

    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", source_files)
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_data, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["snapshot_data.py"]):
        snapshot_data.main()

    snap_name = next(snap_dir.iterdir()).name
    assert snap_name.startswith("pre_weekly_")
    # Date portion is 8 digits.
    date_part = snap_name.replace("pre_weekly_", "")
    assert date_part.isdigit() and len(date_part) == 8


def test_label_appended_to_folder_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    snap_dir = tmp_path / "snapshots"
    source_files = make_source_files(src_dir, SOURCE_NAMES)

    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", source_files)
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_data, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["snapshot_data.py", "--label", "before_first_run"]):
        snapshot_data.main()

    snap_name = next(snap_dir.iterdir()).name
    assert "before_first_run" in snap_name


def test_duplicate_date_appends_counter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    snap_dir = tmp_path / "snapshots"
    source_files = make_source_files(src_dir, SOURCE_NAMES)

    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", source_files)
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_data, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["snapshot_data.py"]):
        snapshot_data.main()
    with patch("sys.argv", ["snapshot_data.py"]):
        snapshot_data.main()

    names = sorted(p.name for p in snap_dir.iterdir())
    assert len(names) == 2
    assert names[1].endswith("_2")


def test_manifest_contains_expected_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    snap_dir = tmp_path / "snapshots"
    source_files = make_source_files(src_dir, SOURCE_NAMES)

    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", source_files)
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(snapshot_data, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["snapshot_data.py"]):
        snapshot_data.main()

    snap = next(snap_dir.iterdir())
    manifest = json.loads((snap / "manifest.json").read_text())

    assert "created_at" in manifest
    assert "files" in manifest
    assert len(manifest["files"]) == len(SOURCE_NAMES)
    for entry in manifest["files"]:
        assert "source" in entry
        assert "snapshot" in entry
        assert "size" in entry


def test_aborts_if_source_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    # Point to a file that does not exist.
    missing = tmp_path / "nonexistent.csv"
    monkeypatch.setattr(snapshot_data, "SNAPSHOT_FILES", [missing])
    monkeypatch.setattr(snapshot_data, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(snapshot_data, "REPO_ROOT", tmp_path)

    with patch("sys.argv", ["snapshot_data.py"]):
        with pytest.raises(SystemExit) as exc_info:
            snapshot_data.main()

    assert exc_info.value.code == 1
    # No snapshot directory should have been created.
    assert not snap_dir.exists() or not any(snap_dir.iterdir())
