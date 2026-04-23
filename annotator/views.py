import json
import uuid
from django.db.models import Count
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from .supabase_storage import upload_image_to_supabase
from django.http import HttpResponseForbidden
from .models import AnalysisImage, BoundingBox, TaxonomyLabel, UserProfile


# ---------------------------------------------------------------------------
# Upload / index
# ---------------------------------------------------------------------------

@login_required
def index(request):
    images = AnalysisImage.objects.annotate(box_count=Count('boxes')).order_by("-uploaded_at")

    cluster_id = request.GET.get("cluster")
    if cluster_id is not None and cluster_id != "":
        images = images.filter(cluster_id=int(cluster_id))

    clusters = (
        AnalysisImage.objects
        .exclude(cluster_id__isnull=True)
        .values_list("cluster_id", flat=True)
        .distinct()
        .order_by("cluster_id")
    )

    return render(request, "annotator/index.html", {
        "images": images,
        "clusters": list(clusters),
        "active_cluster": cluster_id,
    })

@login_required
@require_POST
def upload_image(request):
    role = get_user_role(request.user)
    if role not in ["annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    f = request.FILES.get("image")
    if not f:
        return JsonResponse({"error": "No file provided"}, status=400)

    image_id = uuid.uuid4()
    storage_path = f"analysis_images/{image_id}/{f.name}"

    try:
        public_url = upload_image_to_supabase(f, storage_path)
    except Exception as exc:
        return JsonResponse({"error": f"Storage upload failed: {exc}"}, status=500)

    img = AnalysisImage.objects.create(
        id=image_id,
        image_url=public_url,
        original_filename=f.name,
        uploaded_by=request.user if request.user.is_authenticated else None,
    )
    return JsonResponse({"id": str(img.id), "redirect": f"/annotator/{img.id}/bound/"})

# ---------------------------------------------------------------------------
# Main workspace view (renders shell; data loaded via AJAX)
# ---------------------------------------------------------------------------

@login_required
def workspace(request, image_id, tab="bound"):
    image = get_object_or_404(AnalysisImage, pk=image_id)
    taxonomy = list(TaxonomyLabel.objects.values("id", "name", "category"))
    return render(request, "annotator/workspace.html", {
        "image": image,
        "active_tab": tab,
        "taxonomy_json": json.dumps(taxonomy),
    })


# ---------------------------------------------------------------------------
# API: bounding boxes
# ---------------------------------------------------------------------------

def api_boxes(request, image_id):
    """GET: list boxes. POST: create box."""
    image = get_object_or_404(AnalysisImage, pk=image_id)
    role = get_user_role(request.user)

    if request.method == "GET":
        boxes = image.boxes.all()
        return JsonResponse({"boxes": [b.to_dict() for b in boxes]})

    if request.method == "POST":
        if role not in ["annotator", "admin"]:
            return JsonResponse({"error": "Permission denied"}, status=403)

        data = json.loads(request.body)
        box = BoundingBox.objects.create(
            image=image,
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            source="manual",
            created_by=request.user if request.user.is_authenticated else None,
        )
        return JsonResponse(box.to_dict(), status=201)

    return JsonResponse({"error": "Method not allowed"}, status=405)


def api_box_detail(request, image_id, box_id):
    """PATCH: update a box. DELETE: hard-delete."""
    image = get_object_or_404(AnalysisImage, pk=image_id)
    box = get_object_or_404(BoundingBox, pk=box_id, image=image)
    role = get_user_role(request.user)

    if role not in ["annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)
    
    if request.method == "PATCH":
        data = json.loads(request.body)
        for field in ("x", "y", "width", "height"):
            if field in data:
                setattr(box, field, data[field])
        if "source" in data:
            box.source = data["source"]
        box.save()
        return JsonResponse(box.to_dict())

    if request.method == "DELETE":
        box.delete()
        return JsonResponse({"deleted": str(box_id)})

    return JsonResponse({"error": "Method not allowed"}, status=405)


# ---------------------------------------------------------------------------
# API: classification
# ---------------------------------------------------------------------------
@require_POST
def api_classify_box(request, image_id, box_id):
    """Assign a taxonomy label to a box."""
    role = get_user_role(request.user)
    if role not in ["classifier", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    image = get_object_or_404(AnalysisImage, pk=image_id)
    box = get_object_or_404(BoundingBox, pk=box_id, image=image)
    data = json.loads(request.body)

    label_id = data.get("taxonomy_label_id")
    if label_id:
        label = get_object_or_404(TaxonomyLabel, pk=label_id)
        box.taxonomy_label = label
    box.classified_by = request.user if request.user.is_authenticated else None
    box.classified_at = timezone.now()
    box.save()
    return JsonResponse(box.to_dict())

# @require_POST
# def api_classify_box(request, image_id, box_id):
#     """Assign a taxonomy label to a box."""
#     image = get_object_or_404(AnalysisImage, pk=image_id)
#     box = get_object_or_404(BoundingBox, pk=box_id, image=image)
#     data = json.loads(request.body)

#     label_id = data.get("taxonomy_label_id")
#     if label_id:
#         label = get_object_or_404(TaxonomyLabel, pk=label_id)
#         box.taxonomy_label = label
#     box.classified_by = request.user if request.user.is_authenticated else None
#     box.classified_at = timezone.now()
#     box.save()
#     return JsonResponse(box.to_dict())


# ---------------------------------------------------------------------------
# API: DINOv3 per-box suggestion
# ---------------------------------------------------------------------------

def _fetch_frame_bytes(image):
    """Download the full frame from Supabase storage. Returns (bytes, error_response)."""
    from urllib.request import urlopen
    from urllib.error import URLError
    if not image.image_url:
        return None, JsonResponse({"error": "Image has no image_url"}, status=400)
    try:
        with urlopen(image.image_url, timeout=15) as resp:
            return resp.read(), None
    except URLError as exc:
        return None, JsonResponse({"error": f"Could not fetch image: {exc}"}, status=502)


@login_required
@require_POST
def api_suggest_box(request, image_id, box_id):
    """Classify a single box crop and return {label, confidence, top_k}. No DB write."""
    role = get_user_role(request.user)
    if role not in ["classifier", "annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    image = get_object_or_404(AnalysisImage, pk=image_id)
    box = get_object_or_404(BoundingBox, pk=box_id, image=image)

    img_bytes, err = _fetch_frame_bytes(image)
    if err is not None:
        return err

    try:
        from ml.bridge import run_dino_inference_on_crop
        result = run_dino_inference_on_crop(
            img_bytes, (box.x, box.y, box.width, box.height)
        )
    except Exception as e:
        return JsonResponse({"error": f"Inference failed: {e}"}, status=500)

    if result is None:
        return JsonResponse({
            "error": "No trained classifier found. Run `python -m ml.run train` first.",
        }, status=503)

    return JsonResponse({"box_id": str(box.id), **result})


@login_required
@require_POST
def api_suggest_all(request, image_id):
    """Batched classification for every unlabelled box on a frame."""
    role = get_user_role(request.user)
    if role not in ["classifier", "annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    image = get_object_or_404(AnalysisImage, pk=image_id)
    boxes = list(image.boxes.filter(taxonomy_label__isnull=True))
    if not boxes:
        return JsonResponse({"suggestions": [], "message": "No unclassified boxes"})

    img_bytes, err = _fetch_frame_bytes(image)
    if err is not None:
        return err

    try:
        from ml.bridge import run_dino_inference_on_crops
        crops = [(str(b.id), (b.x, b.y, b.width, b.height)) for b in boxes]
        results = run_dino_inference_on_crops(img_bytes, crops)
    except Exception as e:
        return JsonResponse({"error": f"Inference failed: {e}"}, status=500)

    if not results:
        return JsonResponse({
            "suggestions": [],
            "error": "No trained classifier found.",
        }, status=503)

    return JsonResponse({
        "suggestions": [{"box_id": bid, **r} for bid, r in results],
    })


# ---------------------------------------------------------------------------
# API: specification (researcher finalisation)
# ---------------------------------------------------------------------------

@require_POST
def api_specify_box(request, image_id, box_id):
    role = get_user_role(request.user)
    if role not in ["researcher", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    image = get_object_or_404(AnalysisImage, pk=image_id)
    box = get_object_or_404(BoundingBox, pk=box_id, image=image)
    data = json.loads(request.body)

    box.species_name = data.get("species_name", box.species_name)
    box.researcher_notes = data.get("researcher_notes", box.researcher_notes)
    box.is_finalized = data.get("is_finalized", box.is_finalized)
    box.specified_by = request.user if request.user.is_authenticated else None
    box.specified_at = timezone.now()
    box.save()
    return JsonResponse(box.to_dict())

# @require_POST
# def api_specify_box(request, image_id, box_id):
#     image = get_object_or_404(AnalysisImage, pk=image_id)
#     box = get_object_or_404(BoundingBox, pk=box_id, image=image)
#     data = json.loads(request.body)

#     box.species_name = data.get("species_name", box.species_name)
#     box.researcher_notes = data.get("researcher_notes", box.researcher_notes)
#     box.is_finalized = data.get("is_finalized", box.is_finalized)
#     box.specified_by = request.user if request.user.is_authenticated else None
#     box.specified_at = timezone.now()
#     box.save()
#     return JsonResponse(box.to_dict())


# ---------------------------------------------------------------------------
# API: export finalised boxes for an image
# ---------------------------------------------------------------------------

def api_export(request, image_id):
    """
    Returns all finalised bounding boxes as JSON.
    Downstream ML pipelines / researchers can consume this.
    """
    role = get_user_role(request.user)
    if role not in ["annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)
    image = get_object_or_404(AnalysisImage, pk=image_id)
    boxes = image.boxes.all()
    payload = {
        "image_id": str(image.id),
        "image_filename": image.original_filename,
        "status": image.status,
        "boxes": [b.to_dict() for b in boxes],
        # Group finalized boxes by category for routing to specialist researchers
        "by_category": {},
    }
    for box in boxes.filter(is_finalized=True):
        cat = box.taxonomy_label.category if box.taxonomy_label else "other"
        payload["by_category"].setdefault(cat, []).append(box.to_dict())

    return JsonResponse(payload)

    
# ---------------------------------------------------------------------------
#  Admin Dashboard
# ---------------------------------------------------------------------------
def admin_dashboard(request):
    role = get_user_role(request.user)
    if role != "admin":
        return HttpResponseForbidden("Not allowed")
    
    return render(request, "annotator/admin_dashboard.html")

def get_user_role(user):
    if not user.is_authenticated:
        return None
    try:
        return user.userprofile.role
    except UserProfile.DoesNotExist:
        return None