"""
Copy N sample images per cluster into a local staging folder so they can be
uploaded to Supabase manually via the dashboard UI.

Reads:
  ml/outputs/embeddings/clusters.json          {filename: cluster_id}
  ml/outputs/embeddings/cluster_summary.json   {cluster_id: {count, representative}}

Picks N frames per cluster (deterministic, evenly spaced across the sorted
cluster members) and copies them into:

  C:\\Users\\lmcka\\OneDrive\\Desktop\\images_to_upload\\cluster_<id>\\<filename>

Run from repo root:
  python scripts/stage_cluster_samples.py            # default 10 per cluster
  python scripts/stage_cluster_samples.py --per 15
  python scripts/stage_cluster_samples.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CLUSTERS_JSON = REPO_ROOT / "ml" / "outputs" / "embeddings" / "clusters.json"
SUMMARY_JSON = REPO_ROOT / "ml" / "outputs" / "embeddings" / "cluster_summary.json"

IMAGE_SOURCE_DIR = Path("C:/Users/lmcka/OneDrive/Desktop/deco/images")
DEST_ROOT = Path("C:/Users/lmcka/OneDrive/Desktop/images_to_upload")


def pick_samples(filenames: list[str], n: int) -> list[str]:
    """Pick N evenly-spaced filenames from a sorted list (deterministic)."""
    filenames = sorted(filenames)
    if len(filenames) <= n:
        return filenames
    step = len(filenames) / n
    return [filenames[int(i * step)] for i in range(n)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per", type=int, default=10, help="Samples per cluster")
    parser.add_argument("--dry-run", action="store_true", help="List copies without doing them")
    args = parser.parse_args()

    if not CLUSTERS_JSON.exists() or not SUMMARY_JSON.exists():
        print("ERROR: cluster JSON files not found in ml/outputs/embeddings/", file=sys.stderr)
        return 1
    if not IMAGE_SOURCE_DIR.exists():
        print(f"ERROR: image source dir not found: {IMAGE_SOURCE_DIR}", file=sys.stderr)
        return 1

    clusters_map: dict[str, int] = json.loads(CLUSTERS_JSON.read_text())
    summary: dict[str, dict] = json.loads(SUMMARY_JSON.read_text())

    by_cluster: dict[int, list[str]] = defaultdict(list)
    for fname, cid in clusters_map.items():
        by_cluster[int(cid)].append(fname)

    print(f"Found {len(by_cluster)} clusters, {sum(len(v) for v in by_cluster.values())} frames total")
    print(f"Sampling {args.per} per cluster -> {DEST_ROOT}")
    print()

    if not args.dry_run:
        DEST_ROOT.mkdir(parents=True, exist_ok=True)

    total_copied = 0
    total_skipped = 0
    total_missing = 0

    for cid in sorted(by_cluster.keys()):
        members = by_cluster[cid]
        picks = pick_samples(members, args.per)
        cluster_total = summary.get(str(cid), {}).get("count", len(members))
        cluster_dir = DEST_ROOT / f"cluster_{cid:02d}"
        print(f"Cluster {cid:>2}  ({cluster_total} frames)  -> picking {len(picks)}  -> {cluster_dir.name}/")

        if not args.dry_run:
            cluster_dir.mkdir(parents=True, exist_ok=True)

        for fname in picks:
            src = IMAGE_SOURCE_DIR / fname
            dst = cluster_dir / fname

            if not src.exists():
                print(f"   MISS  {fname}  (not on disk)")
                total_missing += 1
                continue

            if args.dry_run:
                print(f"   PLAN  {dst}")
                continue

            if dst.exists():
                print(f"   SKIP  {fname}  (already staged)")
                total_skipped += 1
                continue

            shutil.copy2(src, dst)
            print(f"   OK    {fname}")
            total_copied += 1

    print()
    print(f"Copied: {total_copied}   Skipped: {total_skipped}   Missing: {total_missing}")
    print(f"Staged at: {DEST_ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
