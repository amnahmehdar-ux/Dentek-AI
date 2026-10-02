# 🦷 Dentek — AI-Assisted Dental Diagnostic System

Dentek is an end-to-end AI-powered web platform designed to assist dentists in detecting dental pathologies and abnormalities from panoramic X-rays. It serves as a clinical decision-support tool, helping streamline report generation and diagnosis workflows.

---

## 🌟 Key Features
- **Panoramic X-Ray Upload & Processing**: Seamless intake and preprocessing of high-resolution dental X-rays.
- **Tooth Detection & Cropping**: Computer vision pipeline to segment and localize individual teeth and anatomical areas.
- **AI Classification**: Deep learning inference for pathology detection, impacted teeth, and dental findings.
- **Structured Findings & Reports**: Automated generation of editable diagnostic reports for practitioners.
- **Doctor Verification Interface**: Interactive UI allowing dentists to review, edit findings, and export PDF reports.

---

## 🛠️ Tech Stack
- **Frontend**: React, Tailwind CSS, Lucide Icons, Vite
- **Backend**: Django REST Framework, Python
- **AI & Computer Vision**: PyTorch, YOLOv8 (OBB), OpenCV, ResNet
- **Database**: MariaDB / SQLite (dev)

---

## 🚀 Getting Started

### 1. Backend Setup
```bash
cd backend
python -m venv venv
# On Windows:
.\venv\Scripts\activate
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
