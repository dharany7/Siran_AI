"""add_is_routable_column.py — one-shot DB migration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.database import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_name='hospitals' AND column_name='is_routable'"
    ))
    exists = result.fetchone() is not None

    if not exists:
        conn.execute(text(
            "ALTER TABLE hospitals ADD COLUMN is_routable BOOLEAN NOT NULL DEFAULT TRUE"
        ))
        conn.commit()
        print("Column is_routable added to hospitals table.")
    else:
        print("Column is_routable already exists — no change needed.")
