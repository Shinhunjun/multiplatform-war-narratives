"""
revert_snapshot.py

Restores the six canonical data files from a previously created snapshot.

Usage:
    # List available snapshots:
    python gdelt/tools/revert_snapshot.py --list

    # Preview what would be restored (no changes made):
    python gdelt/tools/revert_snapshot.py --snapshot pre_weekly_20260325 --dry-run

    # Restore from a snapshot:
    python gdelt/tools/revert_snapshot.py --snapshot pre_weekly_20260325
"""

import argparse
import json
import shutil
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
GDELT_ROOT = REPO_ROOT / "gdelt"
DATA_ROOT = GDELT_ROOT / "data"
SNAPSHOTS_DIR = DATA_ROOT / "snapshots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_snapshots() -> None:
    if not SNAPSHOTS_DIR.exists():
        print("No snapshots directory found.")
        return

    snapshots = sorted(
        p for p in SNAPSHOTS_DIR.iterdir() if p.is_dir()
    )
    if not snapshots:
        print("No snapshots found.")
        return

    print(f"Available snapshots in {SNAPSHOTS_DIR.relative_to(REPO_ROOT)}:")
    print()
    for snap in snapshots:
        manifest_path = snap / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            created = manifest.get("created_at", "unknown")
            label = manifest.get("label") or ""
            label_str = f"  [{label}]" if label else ""
            print(f"  {snap.name}{label_str}  —  created {created}")
        else:
            print(f"  {snap.name}  (no manifest)")
    print()


def resolve_snapshot(name: str) -> Path:
    candidate = SNAPSHOTS_DIR / name
    if candidate.is_dir():
        return candidate
    # Also accept a full path.
    full = Path(name)
    if full.is_dir():
        return full
    print(f"ERROR: Snapshot not found: {name}")
    print(f"Run with --list to see available snapshots.")
    sys.exit(1)


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
    parser = argparse.ArgumentParser(
        description="Revert canonical GDELT data files from a snapshot."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available snapshots and exit.",
    )
    parser.add_argument(
        "--snapshot",
        metavar="NAME",
        default=None,
        help="Name of the snapshot folder to restore (e.g. pre_weekly_20260325).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be restored without copying any files.",
    )
    args = parser.parse_args()

    if args.list:
        list_snapshots()
        return

    if not args.snapshot:
        parser.print_help()
        sys.exit(1)

    snapshot_dir = resolve_snapshot(args.snapshot)
    manifest_path = snapshot_dir / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: No manifest.json found in {snapshot_dir}.")
        print("This snapshot may be incomplete or was not created by snapshot_data.py.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    created_at = manifest.get("created_at", "unknown")

    print(f"Snapshot:   {snapshot_dir.relative_to(REPO_ROOT)}")
    print(f"Created:    {created_at}")
    if args.dry_run:
        print("Mode:       DRY RUN — no files will be modified")
    print()

    # Build the restore plan from the manifest.
    plan = []
    for entry in manifest["files"]:
        src = snapshot_dir / Path(entry["snapshot"]).name
        dst = REPO_ROOT / entry["source"]
        plan.append((src, dst))

    # Verify snapshot files are present before touching anything.
    missing = [src for src, _ in plan if not src.exists()]
    if missing:
        print("ERROR: The following snapshot files are missing:")
        for f in missing:
            print(f"  {f.relative_to(REPO_ROOT)}")
        print("Revert aborted — no files were changed.")
        sys.exit(1)

    for src, dst in plan:
        size = fmt_size(src)
        label = "would restore" if args.dry_run else "restoring  "
        print(f"  {label}  {dst.relative_to(DATA_ROOT)}  ({size})")
        if not args.dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    print()
    if args.dry_run:
        print("Dry run complete — no files were modified.")
    else:
        print("Revert complete.")


if __name__ == "__main__":
    main()
