"""
snapshot_data.py

Creates a dated snapshot of the six canonical data files that the weekly
update pipeline modifies in place.  Run this before each weekly update.

Usage:
    python gdelt/tools/snapshot_data.py
    python gdelt/tools/snapshot_data.py --label my_note

Snapshot folder:
    gdelt/data/snapshots/pre_weekly_YYYYMMDD/
    (a suffix is appended if a folder for today already exists)
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GDELT_ROOT = REPO_ROOT / "gdelt"
DATA_ROOT = GDELT_ROOT / "data"
SNAPSHOTS_DIR = DATA_ROOT / "snapshots"

SNAPSHOT_FILES = [
    DATA_ROOT / "gdelt_scraped.csv",
    DATA_ROOT / "preprocessing" / "url_lookup.csv",
    DATA_ROOT / "preprocessing" / "url_filter_eval.csv",
    DATA_ROOT / "preprocessing" / "url_filter_summary_counts.csv",
    DATA_ROOT / "analysis_ready" / "analysis_events.parquet",
    DATA_ROOT / "analysis_ready" / "analysis_url_content.parquet",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_snapshot_dir(label: str | None) -> Path:
    today = datetime.now().strftime("%Y%m%d")
    base_name = f"pre_weekly_{today}"
    if label:
        base_name = f"{base_name}_{label}"
    candidate = SNAPSHOTS_DIR / base_name
    if not candidate.exists():
        return candidate
    # Append a counter if today's folder already exists.
    for n in range(2, 100):
        candidate = SNAPSHOTS_DIR / f"{base_name}_{n}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("Could not find a unique snapshot directory name.")


def fmt_size(path: Path) -> str:
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Snapshot canonical GDELT data files.")
    parser.add_argument(
        "--label",
        metavar="TEXT",
        default=None,
        help="Optional short label appended to the snapshot folder name.",
    )
    args = parser.parse_args()

    # Verify source files exist before touching anything.
    missing = [f for f in SNAPSHOT_FILES if not f.exists()]
    if missing:
        print("ERROR: The following source files do not exist:")
        for f in missing:
            print(f"  {f.relative_to(REPO_ROOT)}")
        print("Snapshot aborted — no files were copied.")
        sys.exit(1)

    snapshot_dir = choose_snapshot_dir(args.label)
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    print(f"Snapshot directory: {snapshot_dir.relative_to(REPO_ROOT)}")
    print()

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": args.label,
        "files": [],
    }

    for src in SNAPSHOT_FILES:
        dst = snapshot_dir / src.name
        shutil.copy2(src, dst)
        entry = {
            "source": str(src.relative_to(REPO_ROOT)),
            "snapshot": str(dst.relative_to(REPO_ROOT)),
            "size": fmt_size(src),
        }
        manifest["files"].append(entry)
        print(f"  copied  {src.relative_to(DATA_ROOT)}  ({entry['size']})")

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print()
    print(f"Manifest written: {manifest_path.relative_to(REPO_ROOT)}")
    print("Snapshot complete.")


if __name__ == "__main__":
    main()
