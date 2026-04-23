"""
Stamp DINOv3 cluster IDs onto AnalysisImage rows.

Reads:
  ml/outputs/embeddings/clusters.json   {filename: cluster_id}

For every AnalysisImage in the DB it tries to match by:
  1. original_filename (exact)
  2. basename of image_url path (exact)

and writes the cluster_id. Reports matched / unmatched / unchanged counts.

Usage:
  python manage.py assign_clusters
  python manage.py assign_clusters --dry-run
  python manage.py assign_clusters --reset    # clear cluster_id first
"""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

from django.core.management.base import BaseCommand
from django.db import transaction

from annotator.models import AnalysisImage


def _url_basename(url: str) -> str:
    if not url:
        return ""
    return os.path.basename(urlparse(url).path)

REPO_ROOT = Path(__file__).resolve().parents[3]
CLUSTERS_JSON = REPO_ROOT / "ml" / "outputs" / "embeddings" / "clusters.json"


class Command(BaseCommand):
    help = "Assign DINOv3 cluster_id to AnalysisImage rows from clusters.json"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true", help="Report without writing")
        parser.add_argument("--reset", action="store_true", help="Clear existing cluster_ids first")

    def handle(self, *args, **opts):
        if not CLUSTERS_JSON.exists():
            self.stderr.write(self.style.ERROR(f"Missing {CLUSTERS_JSON}"))
            return

        clusters_map: dict[str, int] = json.loads(CLUSTERS_JSON.read_text())
        self.stdout.write(f"Loaded {len(clusters_map)} filename->cluster mappings")

        total = AnalysisImage.objects.count()
        self.stdout.write(f"AnalysisImage rows in DB: {total}")
        if total == 0:
            self.stdout.write(self.style.WARNING(
                "No AnalysisImage rows yet. Ingest the frames first "
                "(e.g. via bulk_import) and re-run this command."
            ))
            return

        matched = unmatched = unchanged = 0
        unmatched_examples = []

        if opts["reset"] and not opts["dry_run"]:
            AnalysisImage.objects.update(cluster_id=None)
            self.stdout.write("Reset all cluster_id values to NULL")

        with transaction.atomic():
            for img in AnalysisImage.objects.all().iterator():
                # Try original_filename first, then basename of image_url path
                key = img.original_filename or ""
                cid = clusters_map.get(key)
                url_base = _url_basename(img.image_url)
                if cid is None and url_base:
                    cid = clusters_map.get(url_base)

                if cid is None:
                    unmatched += 1
                    if len(unmatched_examples) < 5:
                        unmatched_examples.append(img.original_filename or url_base)
                    continue

                if img.cluster_id == int(cid):
                    unchanged += 1
                    matched += 1
                    continue

                if not opts["dry_run"]:
                    img.cluster_id = int(cid)
                    img.save(update_fields=["cluster_id"])
                matched += 1

            if opts["dry_run"]:
                transaction.set_rollback(True)

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS(f"Matched:   {matched}"))
        self.stdout.write(f"  unchanged: {unchanged}")
        self.stdout.write(f"  updated:   {matched - unchanged}")
        self.stdout.write(self.style.WARNING(f"Unmatched: {unmatched}"))
        if unmatched_examples:
            self.stdout.write("  examples of unmatched filenames:")
            for ex in unmatched_examples:
                self.stdout.write(f"    {ex}")
        if opts["dry_run"]:
            self.stdout.write(self.style.NOTICE("(dry-run — no changes written)"))
