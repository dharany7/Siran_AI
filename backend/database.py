"""
database.py — SQLAlchemy engine, session factory, and Base model.
All models should import Base from here and call init_db() on startup.

Driver: psycopg2-binary (sync).  Connection URL read from DATABASE_URL
in .env (postgresql://siren:siren@localhost:5432/siren_ai).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from backend.config import settings

engine = create_engine(
    settings.database_url,
    # pool_pre_ping keeps connections healthy after postgres restarts
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db() -> None:
    """Create all tables defined by SQLAlchemy models (idempotent)."""
    import backend.models  # noqa: F401  — registers all models on Base
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
