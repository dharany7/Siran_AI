"""
models.py — SQLAlchemy ORM models.
Add new models here; they will be picked up by init_db() automatically.

Tables
------
Existing (migrated from SQLite):
  siren_events, plate_detections, security_events

New (PostgreSQL-first):
  users, drivers, hospitals, dispatches
"""
import enum
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Enum as PgEnum,
    Float, ForeignKey, Integer, String, Text,
)
from sqlalchemy.orm import relationship

from backend.database import Base


# ── Existing tables ────────────────────────────────────────────────────────────

class SirenEvent(Base):
    """Records a detected siren audio event."""

    __tablename__ = "siren_events"

    id         = Column(Integer, primary_key=True, index=True)
    siren_type = Column(String(64),  nullable=False)   # e.g. "ambulance_dispatch", "fire"
    confidence = Column(String(64),  nullable=True)    # e.g. "0.93" or "1->2" for dispatch
    audio_file = Column(String(256), nullable=True)
    notes      = Column(Text,        nullable=True)    # full negotiation log JSON
    created_at = Column(DateTime, default=datetime.utcnow)


class PlateDetection(Base):
    """Records an ANPR plate-detection result."""

    __tablename__ = "plate_detections"

    id = Column(Integer, primary_key=True, index=True)
    plate_text = Column(String(32), nullable=False)
    image_file = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SecurityEvent(Base):
    """
    Records every call to the PromptGuard, whether SAFE or BLOCKED.

    Fields
    ------
    payload        : The raw text that was checked (truncated to 2 048 chars).
    verdict        : 'SAFE' or 'BLOCKED'.
    layer_blocked  : Which layer triggered the block (1, 2, or 3). NULL if SAFE.
    blocked_reason : Human-readable reason. NULL if SAFE.
    endpoint       : The API endpoint that invoked the guard (e.g. '/dispatch').
    created_at     : UTC timestamp.
    """

    __tablename__ = "security_events"

    id             = Column(Integer,     primary_key=True, index=True)
    payload        = Column(Text,        nullable=True)
    verdict        = Column(String(8),   nullable=False)   # 'SAFE' or 'BLOCKED'
    layer_blocked  = Column(Integer,     nullable=True)
    blocked_reason = Column(String(512), nullable=True)
    endpoint       = Column(String(128), nullable=True)
    created_at     = Column(DateTime,    default=datetime.utcnow)


# ── New tables ─────────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    driver = "driver"
    admin  = "admin"


class DispatchStatus(str, enum.Enum):
    pending    = "pending"
    en_route   = "en_route"
    arrived    = "arrived"
    cancelled  = "cancelled"


class User(Base):
    """Application user — owns login credentials."""

    __tablename__ = "users"

    id              = Column(Integer,                       primary_key=True, index=True)
    phone_number    = Column(String(20),  unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role            = Column(PgEnum(UserRole, name="userrole"), nullable=False, default=UserRole.driver)
    created_at      = Column(DateTime, default=datetime.utcnow)

    driver = relationship("Driver", back_populates="user", uselist=False)


class Driver(Base):
    """Ambulance driver — linked 1-to-1 with a User account."""

    __tablename__ = "drivers"

    id              = Column(Integer, primary_key=True, index=True)
    user_id         = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, unique=True)
    license_number  = Column(String(64),  nullable=False)
    ambulance_plate = Column(String(20),  unique=True, nullable=False, index=True)
    is_online       = Column(Boolean, default=False, nullable=False)
    current_lat     = Column(Float,   nullable=True)
    current_lng     = Column(Float,   nullable=True)
    last_updated    = Column(DateTime, nullable=True)

    user      = relationship("User",     back_populates="driver")
    dispatches = relationship("Dispatch", back_populates="driver")


class Hospital(Base):
    """Destination hospital registered in the system."""

    __tablename__ = "hospitals"

    id          = Column(Integer,      primary_key=True, index=True)
    name        = Column(String(256),  nullable=False)
    lat         = Column(Float,        nullable=False)
    lng         = Column(Float,        nullable=False)
    address     = Column(Text,         nullable=True)
    is_routable = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",   # PostgreSQL literal — keeps existing rows non-null
        comment="True if the hospital's lat/lng can be snapped to a SUMO edge "
                "within 150 m. Set by validate_hospitals.py.",
    )

    dispatches = relationship("Dispatch", back_populates="hospital")


class Dispatch(Base):
    """
    A single ambulance dispatch event.
    hospital_id is nullable — SUMO-based dispatches use junction IDs,
    not hospital rows, but real-world dispatches will reference a hospital.
    """

    __tablename__ = "dispatches"

    id          = Column(Integer, primary_key=True, index=True)
    driver_id   = Column(Integer, ForeignKey("drivers.id",   ondelete="SET NULL"), nullable=True)
    hospital_id = Column(Integer, ForeignKey("hospitals.id", ondelete="SET NULL"), nullable=True)
    start_lat   = Column(Float,  nullable=True)
    start_lng   = Column(Float,  nullable=True)
    status      = Column(
        PgEnum(DispatchStatus, name="dispatchstatus"),
        nullable=False,
        default=DispatchStatus.pending,
    )
    created_at   = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    driver   = relationship("Driver",   back_populates="dispatches")
    hospital = relationship("Hospital", back_populates="dispatches")
