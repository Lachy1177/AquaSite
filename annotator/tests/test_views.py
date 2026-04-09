from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User
from annotator.models import AnalysisImage, BoundingBox, UserProfile
from django.core.files.uploadedfile import SimpleUploadedFile
import json


def make_user(username, role):
    user = User.objects.create_user(username=username, password='testpass123')
    UserProfile.objects.filter(user=user).update(role=role)
    return user


class UploadImageTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('annotator1', 'annotator')
        self.client.login(username='annotator1', password='testpass123')

    def test_upload_image_success(self):
        image = SimpleUploadedFile(
            "test.jpg",
            b"file_content",
            content_type="image/jpeg"
        )
        url = reverse("annotator:upload")
        response = self.client.post(url, {"image": image})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(AnalysisImage.objects.count(), 1)

    def test_upload_image_no_file(self):
        url = reverse("annotator:upload")
        response = self.client.post(url, {})
        self.assertEqual(response.status_code, 400)


class BoundingBoxTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = make_user('annotator1', 'annotator')
        self.client.login(username='annotator1', password='testpass123')
        self.image = AnalysisImage.objects.create(
            original_filename="test.jpg"
        )

    def test_create_box(self):
        url = reverse("annotator:api_boxes", args=[self.image.id])
        data = {"x": 10, "y": 20, "width": 30, "height": 40}
        response = self.client.post(
            url,
            data=json.dumps(data),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 201)
        self.assertEqual(BoundingBox.objects.count(), 1)

    def test_get_boxes(self):
        BoundingBox.objects.create(
            image=self.image, x=1, y=1, width=10, height=10
        )
        url = reverse("annotator:api_boxes", args=[self.image.id])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertIn("boxes", response.json())

    def test_delete_box(self):
        box = BoundingBox.objects.create(
            image=self.image, x=1, y=1, width=10, height=10
        )
        url = reverse("annotator:api_box_detail", args=[self.image.id, box.id])
        response = self.client.delete(url)
        self.assertEqual(response.status_code, 200)
        box.refresh_from_db()
        self.assertTrue(box.is_deleted)