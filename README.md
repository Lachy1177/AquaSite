# AquaSite
Deco3801 Project 17 AI Rapid Sorting of Underwater Ecological Imagery

(optional) To run, install venv, and make sure VScode is using it (python -m venv venv)

Run: pip install -r requirements.txt
To add new requirements (BEFORE COMMIT): pip freeze > requirements.txt

To run webapp: python manage.py runserver
                         ^ path of the manage.py file


Current project overview:
AquaSite/                          ← project root (where manage.py lives)
│
├── manage.py                      ← Django CLI, run everything from here
│
├── AquaSightWeb/                  ← project config package (same name as your project)
│   ├── settings.py                ← all Django settings (INSTALLED_APPS, DB, MEDIA, etc.)
│   ├── urls.py                    ← top-level router, sends /annotator/* to the annotator app
│   └── wsgi.py                    ← production server entry point (ignore for now)
│
├── annotator/                     ← the app all annotation logic lives here
│   ├── models.py                  ← database schema: AnalysisImage, BoundingBox, TaxonomyLabel
│   ├── views.py                   ← all page views + every API endpoint (boxes, DINO, classify, etc.)
│   ├── urls.py                    ← routes /annotator/* URLs to the right view functions
│   ├── admin.py                   ← registers models in Django's /admin panel
│   ├── apps.py                    ← app config (tells Django the app is called "annotator")
│   └── migrations/
│       └── __init__.py            ← empty, lets Django know this is a migrations package
│                                    (Django will add 0001_initial.py here after makemigrations)
│
├── AquaSightWeb/
│   └── templates/annotator/       ← HTML templates (Django finds these automatically)
│       ├── index.html             ← upload page + image gallery at /annotator/
│       └── workspace.html         ← the main Photoshop-style annotation UI at /annotator/<id>/<tab>/
│
├── media/                         ← created automatically when first image is uploaded
│   └── analysis_images/           ← uploaded images are stored here
│
├── db.sqlite3                     ← the database file (created after migrate)

The URL flow:
Browser hits /annotator/
  → AquaSightWeb/urls.py          receives it, strips nothing, hands to annotator/urls.py
  → annotator/urls.py             matches "", calls views.index
  → annotator/views.py            renders templates/annotator/index.html

Browser hits /annotator/<uuid>/bound/
  → annotator/urls.py             matches, calls views.workspace
  → workspace.html loads, JS hits /annotator/api/<uuid>/boxes/ to fetch box data
  → views.api_boxes               queries DB, returns JSON
