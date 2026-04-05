from django.contrib import admin
from .models import AnalysisImage, BoundingBox, TaxonomyLabel, UserProfile


@admin.register(TaxonomyLabel)
class TaxonomyLabelAdmin(admin.ModelAdmin):
    list_display = ["name", "category", "description"]
    list_filter = ["category"]
    search_fields = ["name"]


class BoundingBoxInline(admin.TabularInline):
    model = BoundingBox
    extra = 0
    readonly_fields = ["id", "source", "confidence", "created_at", "updated_at"]
    fields = [
        "id", "x", "y", "width", "height", "source", "confidence",
        "taxonomy_label", "species_name", "is_finalized", "is_deleted",
    ]


@admin.register(AnalysisImage)
class AnalysisImageAdmin(admin.ModelAdmin):
    list_display = ["original_filename", "status", "uploaded_by", "uploaded_at"]
    list_filter = ["status"]
    search_fields = ["original_filename"]
    inlines = [BoundingBoxInline]


@admin.register(BoundingBox)
class BoundingBoxAdmin(admin.ModelAdmin):
    list_display = [
        "id", "image", "source", "confidence",
        "taxonomy_label", "is_finalized", "is_deleted",
    ]
    list_filter = ["source", "is_finalized", "is_deleted", "taxonomy_label__category"]
    search_fields = ["image__original_filename", "species_name"]


admin.site.register(UserProfile)