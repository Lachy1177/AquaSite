from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from annotator.models import AnalysisImage, BoundingBox, UserProfile
from django.core.files.uploadedfile import SimpleUploadedFile
from unittest.mock import patch
import json


def make_user(username, role):
    user = User.objects.create_user(username=username, password="testpass123")
    UserProfile.objects.filter(user=user).update(role=role)
    return user


class RolePermissionTests(TestCase):
    def setUp(self):
        self.client = Client()

        self.annotator = make_user("annotator1", "annotator")
        self.viewer = make_user("viewer1", "viewer")
        self.admin = make_user("admin1", "admin")

        self.image = AnalysisImage.objects.create(
            original_filename="test.jpg"
        )

    def test_upload_requires_login(self):
        url = reverse("annotator:upload")

        image = SimpleUploadedFile(
            "test.jpg",
            b"file_content",
            content_type="image/jpeg"
        )

        response = self.client.post(url, {"image": image})

        self.assertIn(response.status_code, [302, 403])

    @patch(
        "annotator.views.upload_image_to_supabase",
        return_value="https://example.supabase.co/storage/v1/object/public/analysis-images/test.jpg"
    )
    def test_annotator_can_upload_image(self, mock_upload):
        self.client.login(username="annotator1", password="testpass123")

        url = reverse("annotator:upload")

        image = SimpleUploadedFile(
            "test.jpg",
            b"file_content",
            content_type="image/jpeg"
        )

        response = self.client.post(url, {"image": image})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(AnalysisImage.objects.count(), 2)

    def test_viewer_cannot_create_bounding_box(self):
        self.client.login(username="viewer1", password="testpass123")

        url = reverse("annotator:api_boxes", args=[self.image.id])

        data = {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40
        }

        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )

        self.assertIn(response.status_code, [302, 403])
        self.assertEqual(BoundingBox.objects.count(), 0)

    def test_annotator_can_create_bounding_box(self):
        self.client.login(username="annotator1", password="testpass123")

        url = reverse("annotator:api_boxes", args=[self.image.id])

        data = {
            "x": 10,
            "y": 20,
            "width": 30,
            "height": 40
        }

        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(BoundingBox.objects.count(), 1)

    def test_create_bounding_box_missing_data(self):
        self.client.login(username="annotator1", password="testpass123")

        url = reverse("annotator:api_boxes", args=[self.image.id])

        data = {
            "x": 10,
            "y": 20
        }

        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(BoundingBox.objects.count(), 0)

    def test_get_boxes_requires_login(self):
        self.client.logout()

        url = reverse("annotator:api_boxes", args=[self.image.id])

        response = self.client.get(url)

        self.assertIn(response.status_code, [302, 403])

    def test_admin_can_access_admin_dashboard(self):
        self.client.login(username="admin1", password="testpass123")

        url = reverse("annotator:admin_dashboard")

        response = self.client.get(url)

        self.assertEqual(response.status_code, 200)