# AquaSite

**DECO3801 — Project 17: AI Rapid Sorting of Underwater Ecological Imagery**

---

## Getting Started

### 1. Clone & set up the environment

```bash
# Create and activate virtual environment
python -m venv venv

# Windows (cmd):
venv\Scripts\activate

# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in the Supabase credentials
```

### 3. Apply database migrations

```bash
python manage.py migrate
```

### 4. Run the development server

```bash
python manage.py runserver
# → http://localhost:8000/annotator/
```

> **Before committing:** if you've installed new packages, run `pip freeze > requirements.txt`

---

## User Accounts

### Creating an account

Accounts are created by an admin via the Django admin panel at `/admin/`.

1. Log in at `http://localhost:8000/admin/`
2. Go to **Users → Add User**, fill in username and password
3. Go to **Annotator → User Profiles**, find the new user and set their role

Available roles:

| Role | Permissions |
|------|-------------|
| `admin` | Full access |
| `annotator` | Upload images, draw bounding boxes |
| `classifier` | Assign taxonomy labels to boxes |
| `researcher` | Finalise species identification |
| `viewer` | Read-only access |

Alternatively, create a user from the Django shell:

```bash
python manage.py shell
```

```python
from django.contrib.auth.models import User
from annotator.models import UserProfile

user = User.objects.create_user(username='jane', password='securepassword123')
UserProfile.objects.filter(user=user).update(role='annotator')
```

### Modifying an account

**Change role via shell:**

```python
from annotator.models import UserProfile
UserProfile.objects.filter(user__username='jane').update(role='classifier')
```

**Change password via shell:**

```python
from django.contrib.auth.models import User
u = User.objects.get(username='jane')
u.set_password('newpassword123')
u.save()
```

**Or use the admin panel** at `/admin/auth/user/` to change any user's details, and `/admin/annotator/userprofile/` to change their role.

### Creating a superuser (admin CLI access)

```bash
python manage.py createsuperuser
# Then in the shell, set their role:
python manage.py shell
```
```python
from annotator.models import UserProfile
UserProfile.objects.filter(user__is_superuser=True).update(role='admin')
```

---

## Project Structure

```
AquaSite/                          ← project root (where manage.py lives)
│
├── manage.py                      ← Django CLI entry point
│
├── AquaSightWeb/                  ← project config package
│   ├── settings.py                ← Django settings (INSTALLED_APPS, DB, MEDIA, etc.)
│   ├── urls.py                    ← top-level router → hands /annotator/* to annotator app
│   └── wsgi.py                    ← production server entry point
│
├── annotator/                     ← main app; all annotation logic lives here
│   ├── models.py                  ← AnalysisImage, BoundingBox, TaxonomyLabel, UserProfile
│   ├── views.py                   ← page views + all API endpoints
│   ├── urls.py                    ← routes /annotator/* URLs to view functions
│   ├── admin.py                   ← registers models in Django's /admin panel
│   ├── apps.py                    ← app config
│   └── migrations/                ← auto-generated database migrations
│
├── templates/annotator/           ← HTML templates (Django finds these automatically)
│   ├── index.html                 ← upload page + image gallery  →  /annotator/
│   ├── workspace.html             ← annotation UI  →  /annotator/<id>/<tab>/
│   └── login.html                 ← login page  →  /login/
│
├── media/                         ← created automatically on first image upload
│   └── analysis_images/           ← uploaded images stored here
│
└── db.sqlite3                     ← local SQLite DB (CI only; production uses Supabase)
```

---

## URL Flow

```
GET /annotator/
  → AquaSightWeb/urls.py       hands off to annotator/urls.py
  → annotator/urls.py          matches "", calls views.index
  → views.index                renders index.html

GET /annotator/<uuid>/bound/
  → annotator/urls.py          calls views.workspace
  → workspace.html loads
  → JS fetches /annotator/api/<uuid>/boxes/
  → views.api_boxes            queries DB, returns JSON
```

---

## API Endpoints

All endpoints are under `/annotator/api/`.

| Method | Endpoint | Role required | Description |
|--------|----------|---------------|-------------|
| `GET` | `/<id>/boxes/` | any | List bounding boxes |
| `POST` | `/<id>/boxes/` | annotator, admin | Create a bounding box |
| `PATCH` | `/<id>/boxes/<box_id>/` | annotator, admin | Update a box |
| `DELETE` | `/<id>/boxes/<box_id>/` | annotator, admin | Soft-delete a box |
| `POST` | `/<id>/dino/` | any | Run DINOv3 auto-detection (stubbed) |
| `POST` | `/<id>/boxes/<box_id>/classify/` | classifier, admin | Assign taxonomy label |
| `POST` | `/<id>/boxes/<box_id>/specify/` | researcher, admin | Finalise species ID |
| `GET` | `/<id>/export/` | annotator, admin | Export finalised boxes as JSON |