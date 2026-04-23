from django.urls import path
from . import views

app_name = "annotator"

urlpatterns = [
    path("", views.index, name="index"),
    path("upload/", views.upload_image, name="upload"),
    path("<uuid:image_id>/<str:tab>/", views.workspace, name="workspace"),
    path("api/<uuid:image_id>/boxes/", views.api_boxes, name="api_boxes"),
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/", views.api_box_detail, name="api_box_detail"),
    path("api/<uuid:image_id>/dino/", views.api_dino_detect, name="api_dino"),
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/classify/", views.api_classify_box, name="api_classify"),
    path("api/<uuid:image_id>/boxes/<uuid:box_id>/specify/", views.api_specify_box, name="api_specify"),
    path("api/<uuid:image_id>/export/", views.api_export, name="api_export"),
    path("admin-dashboard/", views.admin_dashboard, name="admin_dashboard"),
    path("import-supabase/", views.import_from_supabase, name="import_supabase"),
]