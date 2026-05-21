# AquaSite

**DECO3801 — Project 17: AI Rapid Sorting of Underwater Ecological Imagery**

---

## Getting Started

Prerequisites (both platforms):

- **Python 3.10 or newer** (`python --version` to check)
- **Git**
- A copy of the project `.env` file (ask a maintainer; gitignored)

### Windows

```powershell
# 1. Clone the repo
git clone https://github.com/Lachy1177/AquaSite.git
cd AquaSite

# 2. Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1      # PowerShell
# venv\Scripts\activate           # cmd.exe — use this line instead if you're on cmd

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
copy .env.example .env
# then open .env in a text editor and fill in:
#   - Supabase credentials (SUPABASE_URL, SUPABASE_KEY, DB_*)
#   - HF_TOKEN  (see "DINOv3 Backbone Setup" below)

# 5. Apply database migrations
python manage.py migrate

# 6. Run the dev server
python manage.py runserver
# → http://localhost:8000/annotator/
```

### macOS / Linux

```bash
# 1. Clone the repo
git clone https://github.com/Lachy1177/AquaSite.git
cd AquaSite

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# then open .env in a text editor and fill in:
#   - Supabase credentials (SUPABASE_URL, SUPABASE_KEY, DB_*)
#   - HF_TOKEN  (see "DINOv3 Backbone Setup" below)

# 5. Apply database migrations
python manage.py migrate

# 6. Run the dev server
python manage.py runserver
# → http://localhost:8000/annotator/
```

> **Before committing:** if you've installed new packages, run `pip freeze > requirements.txt`.

---

## DINOv3 Backbone Setup (HuggingFace Token Required)

The ML pipeline uses Meta's DINOv3 ViT-B/16 as a frozen visual backbone (`facebook/dinov3-vitb16-pretrain-lvd1689m`). This is a **gated** model on HuggingFace — you need a free HuggingFace account, agreement to Meta's licence terms, and an access token before the model will download.

### 1. Get a HuggingFace account and access token

1. Create a free account at <https://huggingface.co/join>.
2. Visit the model page: <https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m>.
3. Click **"Agree and access repository"** at the top of the page. Without this, your token won't work for this model.
4. Generate an access token at <https://huggingface.co/settings/tokens> → **New token** → select **Read** scope → name it (e.g. `aquasite-dev`) → copy the `hf_…` string. You only see it once.

### 2. Add the token to your `.env`

Append to `.env`:

```
HF_TOKEN=hf_your_token_here
```

### 3. First-run download (~327 MB)

The first time any ML endpoint runs (e.g. clicking **Classify** in the workspace, or running training locally), the backbone downloads from HuggingFace into `~/.cache/huggingface/hub/` and is reused from there forever after.

| Platform | Cache location |
|---|---|
| Windows | `%USERPROFILE%\.cache\huggingface\hub\` |
| macOS / Linux | `~/.cache/huggingface/hub/` |

After the first download, `HF_TOKEN` is no longer required for inference — the model is loaded straight from the cache. The token is only needed if the cache is empty (e.g. fresh checkout on a new machine, deploying to a new EC2 instance, or the HF cache has been wiped).

### Working offline / air-gapped machines

If you want to use the model without any HuggingFace network access:

1. On a machine with internet, complete the first-run download above.
2. Copy the `models--facebook--dinov3-vitb16-pretrain-lvd1689m/` directory from `~/.cache/huggingface/hub/` to the same location on the offline machine.
3. Set `HF_HUB_OFFLINE=1` in `.env` to stop the library from trying to phone home on every load. The cached weights will still be picked up automatically.

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
AquaSite/                                  ← repo root (where manage.py lives)
│
├── manage.py                              ← Django CLI entry point
├── requirements.txt                       ← Python dependencies (Django + ML)
├── .env.example                           ← template for the gitignored .env
├── README.md                              ← this file
├── AI.txt                                 ← AI usage disclosure log
│
├── AquaSightWeb/                          ← Django project config package
│   ├── settings.py                        ← INSTALLED_APPS, DB, MEDIA, auth, etc.
│   ├── urls.py                            ← top-level router → hands /annotator/* to annotator app
│   └── wsgi.py                            ← production WSGI entry point (gunicorn)
│
├── annotator/                             ← main Django app — annotation + review UI + APIs
│   ├── models.py                          ← AnalysisImage, BoundingBox, TaxonomyLabel, UserProfile
│   ├── views.py                           ← page views + all /annotator/api/* endpoints
│   ├── urls.py                            ← routes /annotator/* URLs to view functions
│   ├── admin.py                           ← registers models in Django's /admin panel
│   ├── apps.py                            ← app config
│   ├── supabase_storage.py                ← upload helpers for the Supabase image bucket
│   ├── cluster_utils.py                   ← DINOv3 cluster-assignment helpers used at upload
│   ├── search_utils.py                    ← semantic-search term extraction for the gallery
│   ├── migrations/                        ← auto-generated DB migrations (incl. HITL fields)
│   ├── static/annotator/                  ← CSS / JS / SVG assets
│   ├── tests/                             ← Django test suite
│   ├── templates/annotator/               ← HTML templates
│   │   ├── landing.html                   ← public landing page  →  /
│   │   ├── login.html                     ← login page  →  /login/
│   │   ├── signup.html                    ← account signup  →  /signup/
│   │   ├── index.html                     ← image gallery  →  /annotator/
│   │   ├── index_partial.html             ← HTMX-style partial for paginated gallery
│   │   ├── workspace.html                 ← annotation UI  →  /annotator/<id>/<tab>/
│   │   ├── review.html                    ← HITL review queue  →  /annotator/review/
│   │   └── admin_dashboard.html           ← admin overview page
│   └── management/commands/               ← `python manage.py <name>` entry points
│       ├── bulk_import.py                 ← walk staged frames, upload + write AnalysisImage rows
│       ├── assign_clusters.py             ← (re)compute DINOv3 cluster_id for existing images
│       ├── populate_ai_suggestions.py     ← bulk-write ai_suggested_label across boxes
│       └── training_stats.py              ← provenance breakdown of training-eligible boxes
│
├── ml/                                    ← all model code and training pipeline
│   ├── config.yaml                        ← model + training + auto_box config
│   ├── run.py                             ← CLI entry: `python -m ml.run {train|eval|embed|cluster}`
│   ├── bridge.py                          ← Django ↔ ML bridge (cached backbone + classifier + inference fns)
│   ├── auto_box.py                        ← uniform NxN tiler for the "Auto-box" button
│   ├── models/
│   │   ├── backbone.py                    ← DINOv3 frozen ViT-B/16 wrapper (CLS + patch tokens)
│   │   └── classifier.py                  ← LinearClassifier + MLPClassifier + checkpoint I/O
│   ├── data/
│   │   ├── config.py                      ← pydantic schema for config.yaml + .env loading
│   │   ├── dataset.py                     ← AquaSiteImageDataset + AquaSiteBoxDataset + .split()
│   │   ├── transforms.py                  ← train + eval image transforms (ImageNet normalisation)
│   │   ├── local_loader.py                ← read images from a local directory
│   │   └── supabase_loader.py             ← read images from Supabase storage
│   ├── training/
│   │   ├── train.py                       ← train loop with class weighting + val-split early stopping
│   │   ├── evaluate.py                    ← macro-F1, per-class report, confusion matrix
│   │   ├── embed.py                       ← embed all images for clustering / search
│   │   └── cluster.py                     ← KMeans over DINOv3 embeddings
│   ├── search/
│   │   └── similarity.py                  ← cosine-similarity search over cached embeddings
│   └── outputs/                           ← gitignored — checkpoints, logs, cached embeddings
│       ├── checkpoints/                   ← classifier_{type}_best.pt + epoch_*.pt
│       ├── logs/                          ← per-run train metrics (JSONL)
│       └── embeddings/                    ← saved DINOv3 embeddings for the search/cluster paths
│
├── scripts/                               ← one-off scripts (run with `python scripts/<name>.py`)
│   ├── extract_frames.py                  ← pull frames out of dive-video MP4s at fixed FPS
│   ├── extract_bags.py                    ← extract images from rosbag files
│   ├── stage_cluster_samples.py           ← copy one representative image per cluster to a staging dir
│   ├── save_cluster_centroids.py          ← persist KMeans centroids for reuse
│   ├── predict_holdout.py                 ← run the classifier over the held-out frames folder
│   ├── test_similarity.py                 ← smoke-test the embedding similarity search
│   └── README.md                          ← per-script notes
│
├── extracted_frames/                      ← gitignored — staged frames from extract_frames.py
├── media/                                 ← gitignored — local image uploads (Supabase is canonical in prod)
└── db.sqlite3                             ← gitignored — local SQLite (dev/CI only; prod is Supabase Postgres)
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