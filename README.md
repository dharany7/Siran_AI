# Siren AI 🚨

> Multi-Agent System for real-time emergency siren detection, ANPR, and response optimisation.

## Project Structure

```
siren-ai/
├── backend/          # FastAPI app, SQLAlchemy (SQLite)
│   ├── main.py       # App factory + lifespan
│   ├── config.py     # Pydantic settings (reads .env)
│   ├── database.py   # Engine, session, Base
│   ├── models.py     # ORM models
│   └── routers/
│       └── health.py # GET /health
├── agents/           # Multi-Agent System logic
│   ├── base_agent.py # Abstract base agent
│   └── coordinator.py
├── audio_ml/         # Siren classifier (librosa + sklearn)
│   └── classifier.py
├── anpr/             # Plate detection (OpenCV + EasyOCR)
│   └── plate_detector.py
├── security/         # Prompt-injection guard
│   └── prompt_guard.py
├── frontend/         # Plain HTML+JS dashboard
│   ├── index.html
│   ├── style.css
│   └── app.js
├── .env              # Local secrets (never commit!)
├── .env.example      # Template for new developers
├── requirements.txt
└── run.py            # `python run.py` to start the server
```

## Quick Start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
copy .env.example .env        # then edit .env with your keys

# 4. Run the server
python run.py
# or: uvicorn backend.main:app --reload
```

The API will be available at:

| Endpoint         | Description          |
|------------------|----------------------|
| `GET /health`    | Liveness check       |
| `GET /docs`      | Swagger UI           |
| `GET /redoc`     | ReDoc docs           |
| `GET /dashboard` | HTML dashboard       |

## Environment Variables

| Variable               | Description                       | Required |
|------------------------|-----------------------------------|----------|
| `GOOGLE_GEMINI_API_KEY`| Gemini API key                    | Yes      |
| `APP_SECRET_KEY`       | App signing secret                | Yes      |
| `DATABASE_URL`         | SQLAlchemy URL (default: SQLite)  | No       |
| `DEBUG`                | Enable debug mode                 | No       |

## Modules

### `audio_ml` — Siren Classifier
Train with labelled `.wav` files:
```bash
python -m audio_ml.classifier --train --data-dir data/audio
```
Predict a file:
```bash
python -m audio_ml.classifier --predict siren_clip.wav
```

### `anpr` — Plate Detection
```python
from anpr.plate_detector import PlateDetector
detector = PlateDetector()
result = detector.detect("car.jpg")
print(result.text, result.confidence)
```

### `security` — Prompt Guard
```python
from security.prompt_guard import PromptGuard
guard = PromptGuard()
safe, reason = guard.check(user_input)
```
