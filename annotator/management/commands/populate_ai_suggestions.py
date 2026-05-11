"""
Batch-run DINOv3 over every BoundingBox missing an `ai_suggested_label`
and persist the top-1 prediction + confidence.

Lets the review queue draw from existing data without waiting for a
classifier to click each box. Skips:
  - boxes already reviewed (is_ai_reviewed=True)
  - boxes already carrying a suggestion (unless --refresh is passed)
  - boxes on images with no image_url

Groups boxes by parent frame so each frame is downloaded once and the
classifier runs as a single batched forward pass per frame (matches
api_suggest_all's hot path).

Usage:
  python manage.py populate_ai_suggestions
  python manage.py populate_ai_suggestions --refresh
  python manage.py populate_ai_suggestions --limit 200
  python manage.py populate_ai_suggestions --dry-run
"""

from collections import defaultdict

from django.core.management.base import BaseCommand
from django.db import transaction

from annotator.models import BoundingBox


class Command(BaseCommand):
    help = "Populate BoundingBox.ai_suggested_label/confidence in bulk via DINOv3."

    def add_arguments(self, parser):
        parser.add_argument(
            "--refresh",
            action="store_true",
            help="Re-run inference on boxes that already have a suggestion "
                 "(still skips is_ai_reviewed=True rows).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Stop after processing this many boxes (useful for smoke tests).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be written without saving.",
        )

    def handle(self, *args, **opts):
        refresh = opts["refresh"]
        limit = opts["limit"]
        dry_run = opts["dry_run"]

        qs = (BoundingBox.objects
              .filter(is_ai_reviewed=False)
              .select_related("image"))
        if not refresh:
            qs = qs.filter(ai_suggested_label="")

        # Group by parent image so each frame is downloaded once.
        by_image = defaultdict(list)
        for box in qs:
            if not box.image.image_url:
                continue
            by_image[box.image_id].append(box)

        total_boxes = sum(len(v) for v in by_image.values())
        self.stdout.write(
            f"Found {total_boxes} candidate boxes across "
            f"{len(by_image)} frames (refresh={refresh}, dry_run={dry_run})."
        )
        if total_boxes == 0:
            return

        # Lazy import — ML deps are heavy and only needed when there's work.
        from ml.bridge import run_dino_inference_on_crops
        from urllib.error import URLError
        from urllib.request import urlopen

        processed = 0
        written = 0
        for image_id, boxes in by_image.items():
            if limit is not None and processed >= limit:
                break

            try:
                with urlopen(boxes[0].image.image_url, timeout=30) as resp:
                    img_bytes = resp.read()
            except URLError as exc:
                self.stdout.write(self.style.WARNING(
                    f"  skip image {image_id}: download failed: {exc}"
                ))
                continue

            crops = [(str(b.id), (b.x, b.y, b.width, b.height)) for b in boxes]
            try:
                results = run_dino_inference_on_crops(img_bytes, crops)
            except Exception as exc:
                self.stdout.write(self.style.ERROR(
                    f"  inference failed on image {image_id}: {exc}"
                ))
                continue

            if not results:
                self.stdout.write(self.style.ERROR(
                    "  no classifier loaded — aborting. "
                    "Ensure ml/outputs/checkpoints/classifier_best.pt is present."
                ))
                return

            box_by_id = {str(b.id): b for b in boxes}
            updates = []
            for bid, r in results:
                box = box_by_id.get(bid)
                if box is None:
                    continue
                label = r.get("label")
                conf = r.get("confidence")
                if not label:
                    continue
                box.ai_suggested_label = label
                box.ai_suggestion_confidence = conf
                updates.append(box)

            processed += len(boxes)
            if dry_run:
                for box in updates:
                    self.stdout.write(
                        f"  would write box {box.id}: "
                        f"{box.ai_suggested_label} ({box.ai_suggestion_confidence:.3f})"
                    )
            else:
                with transaction.atomic():
                    for box in updates:
                        box.save(update_fields=[
                            "ai_suggested_label", "ai_suggestion_confidence",
                        ])
                written += len(updates)

            self.stdout.write(
                f"  image {image_id}: {len(updates)} suggestions "
                f"({processed}/{total_boxes} boxes processed)"
            )

        verb = "would write" if dry_run else "wrote"
        self.stdout.write(self.style.SUCCESS(
            f"Done. {verb} {written if not dry_run else processed} suggestions."
        ))
