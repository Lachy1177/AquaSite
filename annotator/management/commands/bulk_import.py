"""Bulk-import images from a local directory into the Django database.

Automatically assigns DINOv3 cluster IDs if clusters.json exists.

Usage:
    python manage.py bulk_import "C:/Users/lmcka/OneDrive/Desktop/deco/images" --limit 100
    python manage.py bulk_import "C:/Users/lmcka/OneDrive/Desktop/deco/images"  # all images
    python manage.py bulk_import "C:/Users/lmcka/OneDrive/Desktop/deco/images" --cluster 3  # only cluster 3
"""

import json
import os
import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from annotator.models import AnalysisImage


class Command(BaseCommand):
    help = "Bulk-import images with DINOv3 cluster assignments."

    def add_arguments(self, parser):
        parser.add_argument("source_dir", type=str, help="Directory containing images")
        parser.add_argument(
            "--limit", type=int, default=None,
            help="Max number of images to import (default: all)",
        )
        parser.add_argument(
            "--cluster", type=int, default=None,
            help="Only import images from this cluster ID",
        )
        parser.add_argument(
            "--clusters_file", type=str, default=None,
            help="Path to clusters.json (default: ml/outputs/embeddings/clusters.json)",
        )
        parser.add_argument(
            "--extensions", type=str, default="jpg,jpeg,png,tiff",
            help="Comma-separated file extensions to include",
        )

    def handle(self, *args, **options):
        source_dir = Path(options["source_dir"])
        limit = options["limit"]
        cluster_filter = options["cluster"]
        extensions = set(options["extensions"].lower().split(","))

        if not source_dir.exists():
            self.stderr.write(f"Directory not found: {source_dir}")
            return

        # Load cluster assignments
        clusters_path = options["clusters_file"]
        if clusters_path is None:
            base = Path(settings.BASE_DIR) / "ml" / "outputs" / "embeddings" / "clusters.json"
            clusters_path = str(base)

        cluster_map = {}
        if os.path.exists(clusters_path):
            with open(clusters_path, "r") as f:
                cluster_map = json.load(f)
            self.stdout.write(f"Loaded cluster assignments for {len(cluster_map)} images")
        else:
            self.stdout.write(self.style.WARNING(
                f"No clusters.json found at {clusters_path}\n"
                "Run 'python -m ml.run cluster' first to group images.\n"
                "Importing without cluster assignments..."
            ))

        # Destination inside MEDIA_ROOT
        dest_dir = Path(settings.MEDIA_ROOT) / "analysis_images"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Find image files
        files = sorted([
            f for f in source_dir.iterdir()
            if f.is_file() and f.suffix.lstrip(".").lower() in extensions
        ])

        # Filter by cluster if requested
        if cluster_filter is not None and cluster_map:
            files = [f for f in files if cluster_map.get(f.name) == cluster_filter]
            self.stdout.write(f"Filtered to cluster {cluster_filter}: {len(files)} images")

        if limit:
            files = files[:limit]

        # Check which filenames already exist in DB
        existing = set(
            AnalysisImage.objects.values_list("original_filename", flat=True)
        )

        imported = 0
        skipped = 0

        for img_file in files:
            if img_file.name in existing:
                skipped += 1
                continue

            # Copy file to media directory
            dest_path = dest_dir / img_file.name
            if not dest_path.exists():
                shutil.copy2(str(img_file), str(dest_path))

            # Create DB record with cluster assignment
            relative_path = f"analysis_images/{img_file.name}"
            AnalysisImage.objects.create(
                image=relative_path,
                original_filename=img_file.name,
                status="bound",
                cluster_id=cluster_map.get(img_file.name),
            )

            imported += 1
            if imported % 100 == 0:
                self.stdout.write(f"  Imported {imported}...")

        self.stdout.write(self.style.SUCCESS(
            f"Done: {imported} imported, {skipped} skipped (already exist)"
        ))
