from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import revert_snapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_snapshot(
    snapshots_dir: Path,
    name: str,
    file_names: list[str],
    repo_root: Path,
    source_dir: Path,
) -> Path:
    """Create a realistic snapshot folder with a manifest and dummy files."""
    snap = snapshots_dir / name
    snap.mkdir(parents=True)

    files_meta = []
    for fname in file_names:
        f = snap / fname
        f.write_text(f"restored content of {fname}")
        files_meta.append(
            {
                "source": str((source_dir / fname).relative_to(repo_root)),
                "snapshot": str((snap / fname).relative_to(repo_root)),
                "size": "1.0 KB",
            }
        )

    manifest = {
        "created_at": "2026-03-25T10:00:00",
        "label": None,
        "files": files_meta,
    }
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return snap


SOURCE_NAMES = [
    "gdelt_scraped.csv",
    "url_lookup.csv",
]


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------

def test_list_shows_no_snapshots_when_dir_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)

    with patch("sys.argv", ["revert_snapshot.py", "--list"]):
        revert_snapshot.main()

    out = capsys.readouterr().out
    assert "No snapshots found" in out


def test_list_shows_available_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    snap_dir = tmp_path / "snapshots"
    src_dir = tmp_path / "data"
    src_dir.mkdir()

    make_snapshot(snap_dir, "pre_weekly_20260325", SOURCE_NAMES, tmp_path, src_dir)

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)

    with patch("sys.argv", ["revert_snapshot.py", "--list"]):
        revert_snapshot.main()

    out = capsys.readouterr().out
    assert "pre_weekly_20260325" in out
    assert "2026-03-25" in out


# ---------------------------------------------------------------------------
# revert
# ---------------------------------------------------------------------------

def test_restores_files_from_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    src_dir = tmp_path / "data"
    src_dir.mkdir()

    make_snapshot(snap_dir, "pre_weekly_20260325", SOURCE_NAMES, tmp_path, src_dir)

    # Create stale destination files that should be overwritten.
    for name in SOURCE_NAMES:
        (src_dir / name).write_text("stale content")

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(revert_snapshot, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["revert_snapshot.py", "--snapshot", "pre_weekly_20260325"]):
        revert_snapshot.main()

    for name in SOURCE_NAMES:
        content = (src_dir / name).read_text()
        assert content == f"restored content of {name}"


def test_dry_run_does_not_copy_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    src_dir = tmp_path / "data"
    src_dir.mkdir()

    make_snapshot(snap_dir, "pre_weekly_20260325", SOURCE_NAMES, tmp_path, src_dir)

    for name in SOURCE_NAMES:
        (src_dir / name).write_text("original content")

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(revert_snapshot, "DATA_ROOT", src_dir)

    with patch("sys.argv", [
        "revert_snapshot.py", "--snapshot", "pre_weekly_20260325", "--dry-run"
    ]):
        revert_snapshot.main()

    for name in SOURCE_NAMES:
        content = (src_dir / name).read_text()
        assert content == "original content", f"{name} was modified during dry run"


def test_exits_if_snapshot_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    snap_dir.mkdir()

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)

    with patch("sys.argv", ["revert_snapshot.py", "--snapshot", "pre_weekly_99991231"]):
        with pytest.raises(SystemExit) as exc_info:
            revert_snapshot.main()

    assert exc_info.value.code == 1


def test_exits_if_no_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    snap = snap_dir / "pre_weekly_20260325"
    snap.mkdir(parents=True)
    # No manifest.json written.

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)

    with patch("sys.argv", ["revert_snapshot.py", "--snapshot", "pre_weekly_20260325"]):
        with pytest.raises(SystemExit) as exc_info:
            revert_snapshot.main()

    assert exc_info.value.code == 1


def test_exits_if_snapshot_files_missing_from_snapshot_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snap_dir = tmp_path / "snapshots"
    src_dir = tmp_path / "data"
    src_dir.mkdir()

    snap = make_snapshot(snap_dir, "pre_weekly_20260325", SOURCE_NAMES, tmp_path, src_dir)

    # Delete one of the snapshot files to simulate corruption.
    (snap / SOURCE_NAMES[0]).unlink()

    monkeypatch.setattr(revert_snapshot, "SNAPSHOTS_DIR", snap_dir)
    monkeypatch.setattr(revert_snapshot, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(revert_snapshot, "DATA_ROOT", src_dir)

    with patch("sys.argv", ["revert_snapshot.py", "--snapshot", "pre_weekly_20260325"]):
        with pytest.raises(SystemExit) as exc_info:
            revert_snapshot.main()

    assert exc_info.value.code == 1
