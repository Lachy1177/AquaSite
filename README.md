# AquaSite

Deco3801 Project 17 AI Rapid Sorting of Underwater Ecological Imagery

(optional) To run, install venv, and make sure VScode is using it (python -m venv venv)

Run: pip install -r requirements.txt
To add new requirements (BEFORE COMMIT): pip freeze > requirements.txt

To run webapp: python manage.py runserver
^ path of the manage.py file

## AquaSite

How to run currently

```bash
# Create and activate virtual environment
python -m venv venv

# Windows (cmd):
venv\Scripts\activate

# Windows (powershell):
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env

# Edit .env and fill it in with the Supabase credentials

# Start development server
python manage.py runserver
# → http://localhost:8000
```
