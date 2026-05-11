from django.urls import path
from . import views

app_name = "annotator"

urlpatterns = [
    # ── Gallery / upload ─────────────────────────────────────────────
    path("", views.index, name="index"),
    path("upload/", views.upload_image, name="upload"),

    # ── Workspace ────────────────────────────────────────────────────
    path("<uuid:image_id>/<str:tab>/", views.workspace, name="workspace"),

    # ── Queue navigation (NEW) ───────────────────────────────────────
    path("api/queue/next/", views.api_queue_next, name="api_queue_next"),
    path("api/<uuid:image_id>/advance/", views.api_advance_status, name="api_advance_status"),

    # ── Box CRUD ─────────────────────────────────────────────────────
    path("api/<uuid:image_id>/boxes/", views.api_boxes, name="api_boxes"),
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/", views.api_box_detail, name="api_box_detail"),

    # ── DINOv3 suggestions ───────────────────────────────────────────
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/suggest/", views.api_suggest_box, name="api_suggest_box"),
    path("api/<uuid:image_id>/suggest-all/", views.api_suggest_all, name="api_suggest_all"),

    # ── Classification & specification ───────────────────────────────
    path("api/<uuid:image_id>/classify_many/", views.api_classify_boxes, name="api_classify_many"),
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/classify/", views.api_classify_box, name="api_classify"),

    path("api/<uuid:image_id>/boxes/<uuid:box_id>/specify/", views.api_specify_box, name="api_specify"),

    # ── Export ───────────────────────────────────────────────────────
    path("api/<uuid:image_id>/export/", views.api_export, name="api_export"),

    # ── HITL review queue ────────────────────────────────────────────
    path("review/", views.review_dashboard, name="review"),
    path("api/review/queue/", views.api_review_queue, name="api_review_queue"),
    path("api/review/<uuid:box_id>/confirm/", views.api_review_confirm, name="api_review_confirm"),
    path("api/review/<uuid:box_id>/correct/", views.api_review_correct, name="api_review_correct"),

    # ── Admin ────────────────────────────────────────────────────────
    path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),
]
