import json
import uuid
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.views.decorators.http import require_POST, require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.core.files.storage import default_storage
from django.http import HttpResponseForbidden
from .models import AnalysisImage, BoundingBox, TaxonomyLabel, UserProfile


# ---------------------------------------------------------------------------
# Upload / index
# ---------------------------------------------------------------------------

@login_required
def index(request):
    images = AnalysisImage.objects.all().order_by("-uploaded_at")
    return render(request, "annotator/index.html", {"images": images})

@login_required
@require_POST
def upload_image(request):
    role = get_user_role(request.user)
    if role not in ["annotator", "admin"]:
        return JsonResponse({"error": "Permission denied"}, status=403)

    f = request.FILES.get("image")
    if not f:
        return JsonResponse({"error": "No file provided"}, status=400)

    img = AnalysisImage.objects.create(
        image=f,
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

# def api_boxes(request, image_id):
#     """GET: list boxes. POST: create box."""
#     image = get_object_or_404(AnalysisImage, pk=image_id)

#     if request.method == "GET":
#         boxes = image.boxes.filter(is_deleted=False)
#         return JsonResponse({"boxes": [b.to_dict() for b in boxes]})

#     if request.method == "POST":
#         data = json.loads(request.body)
#         box = BoundingBox.objects.create(
#             image=image,
#             x=data["x"],
#             y=data["y"],
#             width=data["width"],
#             height=data["height"],
#             source="manual",
#             created_by=request.user if request.user.is_authenticated else None,
#         )
#         return JsonResponse(box.to_dict(), status=201)

#     return JsonResponse({"error": "Method not allowed"}, status=405)

def api_boxes(request, image_id):
    """GET: list boxes. POST: create box."""
    image = get_object_or_404(AnalysisImage, pk=image_id)
    role = get_user_role(request.user)

    if request.method == "GET":
        boxes = image.boxes.filter(is_deleted=False)
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
    """PATCH: update a box. DELETE: soft-delete."""
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
        box.is_deleted = True
        box.save()
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
# API: DINOv3 auto-detect (stubbed)
# ---------------------------------------------------------------------------

@require_POST
def api_dino_detect(request, image_id):
    """
    Run DINOv3 on the image and return bounding boxes.

    REAL IMPLEMENTATION (uncomment when DINOv3 endpoint is available):
    -----------------------------------------------------------------------
    # import requests as req
    # image = get_object_or_404(AnalysisImage, pk=image_id)
    # DINO_ENDPOINT = "http://your-dino-service/detect"
    # with open(image.image.path, "rb") as f:
    #     resp = req.post(DINO_ENDPOINT, files={"image": f}, data={"threshold": 0.5})
    # raw_boxes = resp.json()["boxes"]  # expects [{x,y,w,h,confidence}, ...] 
    # created = []
    # for rb in raw_boxes:
    #     box = BoundingBox.objects.create(
    #         image=image, x=rb["x"], y=rb["y"],
    #         width=rb["w"], height=rb["h"],
    #         source="dino", confidence=rb["confidence"]
    #     )
    #     created.append(box.to_dict())
    # return JsonResponse({"boxes": created})
    -----------------------------------------------------------------------
    """
    image = get_object_or_404(AnalysisImage, pk=image_id)

    # STUB — generates plausible-looking boxes for UI development
    import random
    random.seed(str(image_id))
    stub_boxes = []
    for _ in range(random.randint(4, 8)):
        x = random.uniform(5, 65)
        y = random.uniform(5, 60)
        w = random.uniform(8, 28)
        h = random.uniform(8, 28)
        box = BoundingBox.objects.create(
            image=image, x=x, y=y, width=w, height=h,
            source="dino", confidence=round(random.uniform(0.55, 0.95), 2),
            created_by=None,
        )
        stub_boxes.append(box.to_dict())

    return JsonResponse({"boxes": stub_boxes})


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
    boxes = image.boxes.filter(is_deleted=False)
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