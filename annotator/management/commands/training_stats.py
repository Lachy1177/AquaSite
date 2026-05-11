"""
Print a breakdown of what BoundingBox rows would feed into the next training run.

Mirrors the filter `ml/data/dataset.py:load_box_records` applies
(`taxonomy_label__isnull=False` + `width * height >= min_box_fraction * 100^2`)
and slices the result by provenance fields the loader currently ignores:

  - human_classified  — classified_by is set (a human signed off)
  - specified         — is_finalized=True (researcher-approved)
  - agrees_with_dino  — taxonomy_label.name == ai_suggested_label
  - corrected         — labels differ → highest-signal training examples
  - rubber_stamp      — agrees with DINO AND no classified_by attribution

Usage:
  python manage.py training_stats
  python manage.py training_stats --by-class
  python manage.py training_stats --min-area 0.005
"""

from django.core.management.base import BaseCommand
from django.db.models import Count, F

from annotator.models import BoundingBox


class Command(BaseCommand):
    help = "Print a provenance breakdown of training-eligible BoundingBox rows."

    def add_arguments(self, parser):
        parser.add_argument(
            "--min-area",
            type=float,
            default=0.005,
            help="Min box area as a fraction of frame area "
                 "(default 0.005 = matches config.training.min_box_fraction).",
        )
        parser.add_argument(
            "--by-class",
            action="store_true",
            help="Also print per-taxonomy-label counts for the usable set.",
        )

    def handle(self, *args, **opts):
        min_area = opts["min_area"]
        min_area_pct_sq = min_area * 100 * 100  # width and height are in percent

        labelled = BoundingBox.objects.filter(taxonomy_label__isnull=False)
        usable = (labelled
                  .annotate(area=F("width") * F("height"))
                  .filter(area__gte=min_area_pct_sq))

        human_classified = usable.filter(classified_by__isnull=False)
        specified = usable.filter(is_finalized=True)

        with_dino = usable.exclude(ai_suggested_label="")
        agrees = with_dino.filter(taxonomy_label__name=F("ai_suggested_label"))
        corrected = with_dino.exclude(taxonomy_label__name=F("ai_suggested_label"))
        rubber = agrees.filter(classified_by__isnull=True)

        out = self.stdout.write
        out("Training set breakdown")
        out(f"  all labelled boxes:           {labelled.count()}")
        out(f"  after area filter (>= {min_area * 100:.2f}% area):")
        out(f"    usable boxes:               {usable.count()}")
        out(f"    human-classified:           {human_classified.count()}")
        out(f"    specified (is_finalized):   {specified.count()}")
        out("")
        out("  vs DINO suggestion (subset where ai_suggested_label is set):")
        out(f"    boxes with a DINO suggestion: {with_dino.count()}")
        out(f"    agrees with DINO:             {agrees.count()}")
        out(f"    corrected from DINO:          {corrected.count()}  (highest-signal)")
        out(f"    agrees AND classified_by NULL:{rubber.count()}  (possible rubber-stamp)")

        if opts["by_class"]:
            out("")
            out("Per-class counts (usable set):")
            rows = list(usable
                        .values("taxonomy_label__name", "taxonomy_label__category")
                        .annotate(n=Count("id"))
                        .order_by("-n"))
            if not rows:
                out("  (no labelled boxes)")
                return
            name_w = max(len(r["taxonomy_label__name"] or "") for r in rows)
            cat_w = max(len(r["taxonomy_label__category"] or "") for r in rows)
            for r in rows:
                name = r["taxonomy_label__name"] or "(unset)"
                cat = r["taxonomy_label__category"] or ""
                out(f"  {cat:<{cat_w}}  {name:<{name_w}}  {r['n']}")
