from django.db import models
from django.contrib.auth.models import User
import uuid
from django.db.models.signals import post_save
from django.dispatch import receiver

class TaxonomyLabel(models.Model):
    """Controlled vocabulary for classification."""
    CATEGORY_CHOICES = [
        ("fish", "Fish"),
        ("coral", "Coral"),
        ("invertebrate", "Invertebrate"),
        ("algae", "Algae"),
        ("other", "Other"),
    ]
    name = models.CharField(max_length=128, unique=True)
    category = models.CharField(max_length=32, choices=CATEGORY_CHOICES, default="other")
    description = models.TextField(blank=True)

    def __str__(self):
        return f"{self.category} / {self.name}"

    class Meta:
        ordering = ["category", "name"]


class AnalysisImage(models.Model):
    """An image submitted for analysis."""
    STATUS_CHOICES = [
        ("bound", "Bounding"),
        ("classify", "Classifying"),
        ("specify", "Specifying"),
        ("complete", "Complete"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image_url = models.CharField(max_length=500, blank=True, default='')
    original_filename = models.CharField(max_length=255, blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="uploaded_images")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default="bound")
    cluster_id = models.IntegerField(null=True, blank=True, help_text="DINOv3 embedding cluster assignment")
    notes = models.TextField(blank=True)

    def __str__(self):
        return f"{self.original_filename} [{self.status}]"

    def mark_complete_if_done(self):
        """Promote status to 'complete' once every box on this image has a label.

        A box counts as labelled if it has a human `taxonomy_label` OR a non-empty
        `ai_suggested_label`. Forward-only: never demotes a 'complete' image.
        Images with no boxes are left alone.
        """
        if self.status == "complete":
            return
        if not self.boxes.exists():
            return
        unlabeled = self.boxes.filter(
            ai_suggested_label="", taxonomy_label__isnull=True
        ).exists()
        if not unlabeled:
            self.status = "complete"
            self.save(update_fields=["status"])

    @property
    def thumbnail_url(self):
        """
        Supabase Storage image transform endpoint — returns a ~320px wide
        JPEG at 60% quality instead of the full-resolution original.
        Cuts per-image transfer from ~300 KB down to ~8–15 KB.
        Falls back to the original URL for non-Supabase hosts.
        """
        if not self.image_url or "/storage/v1/object/public/" not in self.image_url:
            return self.image_url
        return (
            self.image_url.replace("/storage/v1/object/public/", "/storage/v1/render/image/public/")
            + "?width=320&quality=60&resize=cover"
        )

    class Meta:
        ordering = ["-uploaded_at"]


class BoundingBox(models.Model):
    """A single bounding box on an AnalysisImage."""
    SOURCE_CHOICES = [
        ("dino", "DINOv3 (auto)"),
        ("manual", "Manual"),
        ("corrected", "Corrected"),
    ]
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    image = models.ForeignKey(AnalysisImage, on_delete=models.CASCADE, related_name="boxes")
    # Coordinates as percentages (0–100) of image width/height for resolution independence
    x = models.FloatField(help_text="Left edge as % of image width")
    y = models.FloatField(help_text="Top edge as % of image height")
    width = models.FloatField(help_text="Width as % of image width")
    height = models.FloatField(help_text="Height as % of image height")
    source = models.CharField(max_length=16, choices=SOURCE_CHOICES, default="dino")
    confidence = models.FloatField(null=True, blank=True, help_text="Model confidence 0–1")
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name="created_boxes")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Classification phase
    taxonomy_label = models.ForeignKey(
        TaxonomyLabel, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="boxes"
    )
    ai_suggested_label = models.CharField(max_length=128, blank=True)
    ai_suggestion_confidence = models.FloatField(null=True, blank=True)
    classified_by = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="classified_boxes"
    )
    classified_at = models.DateTimeField(null=True, blank=True)
    is_ai_reviewed = models.BooleanField(
        default=False,
        help_text="True once a human has explicitly confirmed or corrected "
                  "the DINOv3 suggestion in the review queue.",
    )

    # Specification phase (researcher identification)
    species_name = models.CharField(max_length=255, blank=True)
    researcher_notes = models.TextField(blank=True)
    specified_by = models.ForeignKey(
        User, on_delete=models.SET_NULL,
        null=True, blank=True, related_name="specified_boxes"
    )
    specified_at = models.DateTimeField(null=True, blank=True)
    is_finalized = models.BooleanField(default=False)

    def __str__(self):
        label = self.taxonomy_label.name if self.taxonomy_label else "unlabelled"
        return f"Box {self.id} on {self.image} [{label}]"

    def to_dict(self):
        """Serialise to JSON-safe dict for frontend use."""
        return {
            "id": str(self.id),
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "source": self.source,
            "confidence": self.confidence,
            "taxonomy_label": self.taxonomy_label.name if self.taxonomy_label else None,
            "taxonomy_category": self.taxonomy_label.category if self.taxonomy_label else None,
            "ai_suggested_label": self.ai_suggested_label,
            "ai_suggestion_confidence": self.ai_suggestion_confidence,
            "species_name": self.species_name,
            "researcher_notes": self.researcher_notes,
            "is_finalized": self.is_finalized,
            "is_ai_reviewed": self.is_ai_reviewed,
        }

    class Meta:
        ordering = ["created_at"]
    
class UserProfile(models.Model):
    ROLE_CHOICES = [
        ("admin", "Admin"),
        ("annotator", "Annotator"),
        ("classifier", "Classifier"),
        ("researcher", "Researcher"),
        ("viewer", "Viewer"),
    ]

    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default="viewer")

    def __str__(self):
        return f"{self.user.username} - {self.role}"
    

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    UserProfile.objects.get_or_create(user=instance)