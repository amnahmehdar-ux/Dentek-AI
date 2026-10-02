from django.contrib.auth.models import User
from django.test import SimpleTestCase, TestCase, RequestFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from rest_framework import status
from rest_framework.test import APITestCase, APIClient
from rest_framework.authtoken.models import Token
from unittest.mock import patch, MagicMock
import json

from .models import Patient, XRay
from .views import (
    clean_json_text,
    generate_dental_recommendation,
    make_json_safe,
    test_groq,
)


# =========================================================
#  CLEAN JSON TEXT TESTS
# =========================================================
class CleanJsonTextTest(SimpleTestCase):

    def test_normal_text(self):
        text = "hello world"
        result = clean_json_text(text)

        self.assertEqual(result, "hello world")

    def test_remove_json_prefix(self):
        text = "```json\n{\"key\": \"value\"}"
        result = clean_json_text(text)

        self.assertEqual(result, "{\"key\": \"value\"}")

    def test_remove_triple_backticks(self):
        text = "```\n{\"key\": \"value\"}"
        result = clean_json_text(text)

        self.assertEqual(result, "{\"key\": \"value\"}")

    def test_remove_both_sides(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        result = clean_json_text(text)

        self.assertEqual(result, "{\"key\": \"value\"}")

    def test_non_string(self):
        result = clean_json_text(123)

        self.assertEqual(result, 123)


# =========================================================
#  GENERATE DENTAL RECOMMENDATION TESTS
# =========================================================
class GenerateDentalRecommendationTest(SimpleTestCase):

    @patch("api.views.client.chat.completions.create")
    def test_generate_recommendation_success_json(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = """
        {
            "summary": "Possible dental findings detected.",
            "urgency": "moderate",
            "recommendation_text": "Review the X-ray clinically.",
            "next_steps": ["Clinical examination", "Further radiographic review"]
        }
        """
        mock_create.return_value = mock_response

        report = {
            "summary": "Lesion and impaction detected",
            "overall_label": "abnormal",
            "total_lesions": 1,
            "total_impacted": 1,
        }

        result = generate_dental_recommendation(
            report,
            findings=[{"label": "finding"}],
            impacted_findings=[{"label": "impacted", "confidence": 0.9}],
            lesion_findings=[{"pred_label": "lesion", "confidence": 0.8}],
        )

        self.assertEqual(result["summary"], "Possible dental findings detected.")
        self.assertEqual(result["urgency"], "moderate")
        self.assertEqual(result["recommendation_text"], "Review the X-ray clinically.")
        self.assertEqual(len(result["next_steps"]), 2)

    @patch("api.views.client.chat.completions.create")
    def test_generate_recommendation_plain_text_fallback(self, mock_create):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Please review the X-ray clinically."
        mock_create.return_value = mock_response

        result = generate_dental_recommendation(
            report={},
            findings=[],
            impacted_findings=[],
            lesion_findings=[],
        )

        self.assertEqual(result["summary"], "AI recommendation generated as plain text.")
        self.assertEqual(result["urgency"], "moderate")
        self.assertEqual(result["recommendation_text"], "Please review the X-ray clinically.")
        self.assertEqual(result["next_steps"], [])

    @patch("api.views.client.chat.completions.create")
    def test_generate_recommendation_error(self, mock_create):
        mock_create.side_effect = Exception("AI service error")

        result = generate_dental_recommendation(
            report={},
            findings=[],
            impacted_findings=[],
            lesion_findings=[],
        )

        self.assertEqual(result["summary"], "Recommendation could not be generated.")
        self.assertEqual(result["urgency"], "unknown")
        self.assertEqual(result["recommendation_text"], "No recommendation available.")
        self.assertIn("error", result)


# =========================================================
#  LOGIN VIEW TESTS
# =========================================================
class LoginViewTest(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser",
            password="123456",
            email="test@test.com",
        )

    def test_login_success(self):
        data = {
            "username": "testuser",
            "password": "123456",
        }

        response = self.client.post("/api/login/", data)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["success"])

    def test_login_failure(self):
        data = {
            "username": "testuser",
            "password": "wrongpassword",
        }

        response = self.client.post("/api/login/", data)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)
        self.assertFalse(response.data["success"])


# =========================================================
#  PROFILE VIEW TESTS
# =========================================================
class ProfileViewTest(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser",
            password="123456",
            email="test@test.com",
        )
        self.token = Token.objects.create(user=self.user)

    def test_get_profile(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token.key)

        response = self.client.get("/api/profile/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["username"], "testuser")
        self.assertEqual(response.data["email"], "test@test.com")

    def test_update_profile(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token.key)

        data = {
            "username": "newname",
            "email": "new@test.com",
        }

        response = self.client.put("/api/profile/", data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["message"], "Updated successfully")

    def test_profile_unauthorized(self):
        response = self.client.get("/api/profile/")

        self.assertEqual(response.status_code, 401)


# =========================================================
#  CREATE PATIENT TESTS
# =========================================================
class CreatePatientTest(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="testuser",
            password="123456",
        )
        self.token = Token.objects.create(user=self.user)

    def test_create_patient_success(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token.key)

        data = {
            "patient_id": "P001",
            "name": "Ahmed",
            "birthDate": "2000-01-01",
            "phone": "123456789",
            "email": "ahmed@test.com",
        }

        response = self.client.post("/api/patients/create/", data)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.data["patient_id"], "P001")

    def test_create_patient_missing_id(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token.key)

        data = {
            "name": "Ahmed",
        }

        response = self.client.post("/api/patients/create/", data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["error"], "patient_id is required")

    def test_create_patient_duplicate(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token.key)

        Patient.objects.create(
            user=self.user,
            patient_id="P001",
            name="Old",
        )

        data = {
            "patient_id": "P001",
            "name": "New",
        }

        response = self.client.post("/api/patients/create/", data)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.data["error"], "Patient ID already exists")

    def test_create_patient_unauthorized(self):
        data = {
            "patient_id": "P002",
            "name": "Test",
        }

        response = self.client.post("/api/patients/create/", data)

        self.assertEqual(response.status_code, 401)


# =========================================================
#  LIST PATIENTS TESTS
# =========================================================
class ListPatientsTest(APITestCase):

    def setUp(self):
        self.user1 = User.objects.create_user(
            username="user1",
            password="123456",
        )
        self.user2 = User.objects.create_user(
            username="user2",
            password="123456",
        )
        self.token1 = Token.objects.create(user=self.user1)

    def test_list_patients(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token1.key)

        Patient.objects.create(user=self.user1, patient_id="P1", name="A")
        Patient.objects.create(user=self.user1, patient_id="P2", name="B")

        response = self.client.get("/api/patients/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 2)

    def test_list_patients_empty(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token1.key)

        response = self.client.get("/api/patients/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 0)

    def test_list_patients_unauthorized(self):
        response = self.client.get("/api/patients/")

        self.assertEqual(response.status_code, 401)

    def test_list_patients_only_user_data(self):
        self.client.credentials(HTTP_AUTHORIZATION="Token " + self.token1.key)

        Patient.objects.create(user=self.user1, patient_id="P1", name="Mine")
        Patient.objects.create(user=self.user2, patient_id="P2", name="Other")

        response = self.client.get("/api/patients/")

        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["patient_id"], "P1")


# =========================================================
#  GROQ AND JSON HELPER TESTS
# =========================================================
class TestGroqAndJsonHelpers(TestCase):

    def setUp(self):
        self.factory = RequestFactory()

    def test_make_json_safe_dict_and_list(self):
        data = {
            "name": "Dentek",
            "values": [1, 2, 3],
            "nested": {
                "status": True,
            },
        }

        result = make_json_safe(data)

        self.assertEqual(result["name"], "Dentek")
        self.assertEqual(result["values"], [1, 2, 3])
        self.assertEqual(result["nested"]["status"], True)

    def test_make_json_safe_tuple(self):
        data = ("lesion", "normal")

        result = make_json_safe(data)

        self.assertEqual(result, ["lesion", "normal"])

    def test_make_json_safe_numpy_values(self):
        import numpy as np

        data = {
            "int_value": np.int64(5),
            "float_value": np.float32(0.95),
            "array_value": np.array([1, 2, 3]),
            "bool_value": np.bool_(True),
        }

        result = make_json_safe(data)

        self.assertIsInstance(result["int_value"], int)
        self.assertIsInstance(result["float_value"], float)
        self.assertEqual(result["array_value"], [1, 2, 3])
        self.assertIsInstance(result["bool_value"], bool)

    @patch("api.views.GROQ_API_KEY", None)
    @patch("api.views.client", None)
    def test_groq_missing_api_key(self):
        request = self.factory.get("/api/test-groq/")

        response = test_groq(request)
        data = json.loads(response.content)

        self.assertEqual(response.status_code, 500)
        self.assertFalse(data["success"])
        self.assertIn("GROQ_API_KEY is missing", data["error"])

    @patch("api.views.GROQ_API_KEY", "fake-key")
    @patch("api.views.client")
    def test_groq_success(self, mock_client):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Hello"
        mock_client.chat.completions.create.return_value = mock_response

        request = self.factory.get("/api/test-groq/")

        response = test_groq(request)
        data = json.loads(response.content)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(data["success"])
        self.assertEqual(data["response"], "Hello")

    @patch("api.views.GROQ_API_KEY", "fake-key")
    @patch("api.views.client")
    def test_groq_exception(self, mock_client):
        mock_client.chat.completions.create.side_effect = Exception("Groq error")

        request = self.factory.get("/api/test-groq/")

        response = test_groq(request)
        data = json.loads(response.content)

        self.assertEqual(response.status_code, 500)
        self.assertFalse(data["success"])
        self.assertEqual(data["error"], "Groq error")


# =========================================================
#  PATIENT AND XRAY TESTS
# =========================================================
class PatientXRayUnitTests(APITestCase):

    def setUp(self):
        self.user = User.objects.create_user(
            username="dr_joud",
            password="password123",
        )
        self.client.force_authenticate(user=self.user)

        self.patient = Patient.objects.create(
            user=self.user,
            name="Ahmed Test",
            patient_id="PAT-001",
        )

        self.upload_url = "/api/xrays/upload/"
        self.list_url = "/api/xrays/"

    def test_get_patient_success(self):
        url = f"/api/patients/{self.patient.id}/"
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["name"], "Ahmed Test")

    def test_get_patient_not_found(self):
        url = "/api/patients/9999/"
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_upload_xray_success(self):
        image = SimpleUploadedFile(
            "tooth_xray.jpg",
            b"fake_data",
            content_type="image/jpeg",
        )
        data = {
            "patient_id": self.patient.id,
            "image": image,
        }

        response = self.client.post(self.upload_url, data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertIn("image_url", response.data)

    def test_upload_xray_invalid_patient(self):
        image = SimpleUploadedFile("tooth.jpg", b"data")
        data = {
            "patient_id": 9999,
            "image": image,
        }

        response = self.client.post(self.upload_url, data, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_list_xrays_success(self):
        XRay.objects.create(
            patient=self.patient,
            image=SimpleUploadedFile("img.jpg", b"abc"),
        )

        response = self.client.get(self.list_url, {"patient_id": self.patient.id})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertTrue(len(response.data) > 0)

    def test_list_xrays_missing_patient_id(self):
        response = self.client.get(self.list_url)

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)


# =========================================================
#  ANALYZE XRAY VIEW TESTS
# =========================================================
class TestAnalyzeXRayView(APITestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="dr_test",
            password="password123",
        )
        self.client.force_authenticate(user=self.user)
        self.url = lambda xray_id: f"/api/xrays/{xray_id}/analyze/"

    @patch("api.views.generate_dental_recommendation")
    @patch("api.views.run_full_analysis")
    @patch("os.path.exists")
    @patch("api.views.XRay.objects.select_related")
    def test_success_flow(self, mock_select, mock_exists, mock_run, mock_rec):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_image = MagicMock()
        mock_image.path = "/media/xray.png"
        mock_image.url = "http://test/xray.png"

        mock_xray = MagicMock()
        mock_xray.id = 1
        mock_xray.image = mock_image
        mock_xray.patient = mock_patient
        mock_xray.save = MagicMock()

        mock_select.return_value.get.return_value = mock_xray
        mock_exists.return_value = True

        mock_run.return_value = {
            "report": {"status": "ok"},
            "findings": ["cavity"],
            "impacted_findings": [],
            "lesion_findings": [],
        }

        mock_rec.return_value = "Go dentist"

        response = self.client.post(self.url(1))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["xray_id"], 1)
        self.assertIn("recommendation", response.data)
        mock_xray.save.assert_called_once()

    @patch("api.views.XRay.objects.select_related")
    def test_xray_not_found(self, mock_select):
        mock_select.return_value.get.side_effect = XRay.DoesNotExist

        response = self.client.post(self.url(999))

        self.assertEqual(response.status_code, 404)

    @patch("api.views.XRay.objects.select_related")
    def test_no_image(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.image = None
        mock_xray.patient = mock_patient

        mock_select.return_value.get.return_value = mock_xray

        response = self.client.post(self.url(1))

        self.assertEqual(response.status_code, 400)

    @patch("os.path.exists")
    @patch("api.views.XRay.objects.select_related")
    def test_file_not_exists(self, mock_select, mock_exists):
        mock_image = MagicMock()
        mock_image.path = "/fake/path.png"

        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.image = mock_image
        mock_xray.patient = mock_patient

        mock_select.return_value.get.return_value = mock_xray
        mock_exists.return_value = False

        response = self.client.post(self.url(1))

        self.assertEqual(response.status_code, 404)


# =========================================================
#  GET XRAY ANALYSIS TESTS
# =========================================================
class TestGetXrayAnalysisView(APITestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="dr_test2",
            password="password123",
        )
        self.client.force_authenticate(user=self.user)
        self.url = lambda xray_id: f"/api/xrays/{xray_id}/analysis/"

    @patch("api.views.XRay.objects.select_related")
    def test_success(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.patient = mock_patient
        mock_xray.analysis_result = {
            "xray_id": 1,
            "findings": ["cavity"],
        }
        mock_xray.doctor_notes = "note"
        mock_xray.edited_report = "edited"

        mock_select.return_value.get.return_value = mock_xray

        response = self.client.get(self.url(1))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["findings"], ["cavity"])

    @patch("api.views.XRay.objects.select_related")
    def test_not_found(self, mock_select):
        mock_select.return_value.get.side_effect = XRay.DoesNotExist

        response = self.client.get(self.url(999))

        self.assertEqual(response.status_code, 404)

    @patch("api.views.XRay.objects.select_related")
    def test_missing_analysis(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.patient = mock_patient
        mock_xray.analysis_result = None

        mock_select.return_value.get.return_value = mock_xray

        response = self.client.get(self.url(1))

        self.assertEqual(response.status_code, 404)

    def test_unauthorized(self):
        self.client.force_authenticate(user=None)

        response = self.client.get(self.url(1))

        self.assertEqual(response.status_code, 401)


# =========================================================
#  LIST REPORTS TESTS
# =========================================================
class TestListReportsView(APITestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="doctor1",
            password="password123",
        )
        self.client.force_authenticate(user=self.user)
        self.url = "/api/reports/"

    @patch("api.views.XRay.objects.filter")
    def test_list_reports_success(self, mock_filter):
        mock_patient = MagicMock()
        mock_patient.id = 10
        mock_patient.patient_id = "PAT-001"
        mock_patient.name = "Ahmed"
        mock_patient.age = 30

        mock_xray = MagicMock()
        mock_xray.id = 1
        mock_xray.patient = mock_patient
        mock_xray.created_at.strftime.return_value = "2026-05-05 10:00"
        mock_xray.analysis_result = {
            "report": {
                "total_lesions": 2,
                "total_impacted": 1,
                "summary": "Test summary",
                "overall_label": "mild",
            },
            "recommendation": "Visit dentist",
        }

        mock_queryset = MagicMock()
        mock_queryset.select_related.return_value.order_by.return_value = [mock_xray]
        mock_filter.return_value = mock_queryset

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

        report = response.data[0]
        self.assertEqual(report["id"], 1)
        self.assertEqual(report["patient_id"], 10)
        self.assertEqual(report["findings"], 3)

    @patch("api.views.XRay.objects.filter")
    def test_list_reports_empty(self, mock_filter):
        mock_queryset = MagicMock()
        mock_queryset.select_related.return_value.order_by.return_value = []
        mock_filter.return_value = mock_queryset

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data, [])

    def test_list_reports_unauthorized(self):
        self.client.force_authenticate(user=None)

        response = self.client.get(self.url)

        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED)


# =========================================================
#  UPDATE REPORT TESTS
# =========================================================
class TestUpdateReportView(APITestCase):

    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username="doctor1",
            password="password123",
        )
        self.client.force_authenticate(user=self.user)
        self.url = lambda xray_id: f"/api/xrays/{xray_id}/update-report/"

    @patch("api.views.XRay.objects.select_related")
    def test_update_report_success(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.patient = mock_patient
        mock_xray.doctor_notes = "old note"
        mock_xray.edited_report = "old report"
        mock_xray.save = MagicMock()

        mock_select.return_value.get.return_value = mock_xray

        data = {
            "doctor_notes": "new note",
            "edited_report": "new report",
        }

        response = self.client.put(self.url(1), data, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["doctor_notes"], "new note")
        self.assertEqual(response.data["edited_report"], "new report")
        mock_xray.save.assert_called_once()

    @patch("api.views.XRay.objects.select_related")
    def test_update_report_not_found(self, mock_select):
        mock_select.return_value.get.side_effect = XRay.DoesNotExist

        data = {
            "doctor_notes": "note",
            "edited_report": "report",
        }

        response = self.client.put(self.url(999), data, format="json")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.data["error"], "XRay not found")

    @patch("api.views.XRay.objects.select_related")
    def test_update_only_doctor_notes(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.patient = mock_patient
        mock_xray.doctor_notes = "old"
        mock_xray.edited_report = "old"
        mock_xray.save = MagicMock()

        mock_select.return_value.get.return_value = mock_xray

        data = {
            "doctor_notes": "updated only notes",
        }

        response = self.client.put(self.url(1), data, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["doctor_notes"], "updated only notes")
        self.assertEqual(response.data["edited_report"], "old")

    @patch("api.views.XRay.objects.select_related")
    def test_update_only_edited_report(self, mock_select):
        mock_patient = MagicMock()
        mock_patient.user = self.user

        mock_xray = MagicMock()
        mock_xray.patient = mock_patient
        mock_xray.doctor_notes = "old"
        mock_xray.edited_report = "old"
        mock_xray.save = MagicMock()

        mock_select.return_value.get.return_value = mock_xray

        data = {
            "edited_report": "updated report only",
        }

        response = self.client.put(self.url(1), data, format="json")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["edited_report"], "updated report only")
        self.assertEqual(response.data["doctor_notes"], "old")

    def test_update_report_unauthorized(self):
        self.client.force_authenticate(user=None)

        data = {
            "doctor_notes": "test",
            "edited_report": "test",
        }

        response = self.client.put(self.url(1), data, format="json")

        self.assertEqual(response.status_code, 401)