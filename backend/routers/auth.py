"""
backend/routers/auth.py — /auth/signup and /auth/login endpoints.

POST /auth/signup
    Body : { phone_number, password, license_number, ambulance_plate }
    Creates a User row (role=driver) and a linked Driver row.
    Returns : { user_id, driver_id, phone_number, ambulance_plate }

POST /auth/login
    Body : { phone_number, password }
    Verifies bcrypt hash; returns a signed JWT access token.
    Returns : { access_token, token_type: "bearer" }
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth import hash_password, verify_password, create_access_token
from backend.database import get_db
from backend.models import Driver, User, UserRole

router = APIRouter(prefix="/auth", tags=["Auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    phone_number:    str = Field(..., example="+919876543210")
    password:        str = Field(..., min_length=6)
    license_number:  str = Field(..., example="TN-0123456789")
    ambulance_plate: str = Field(..., example="TN09AX4471")


class SignupResponse(BaseModel):
    user_id:         int
    driver_id:       int
    phone_number:    str
    ambulance_plate: str
    message:         str


class LoginRequest(BaseModel):
    phone_number: str = Field(..., example="+919876543210")
    password:     str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/signup",
    response_model=SignupResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new driver account",
)
def signup(body: SignupRequest, db: Session = Depends(get_db)) -> SignupResponse:
    # Uniqueness checks
    if db.query(User).filter(User.phone_number == body.phone_number).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="A user with this phone number already exists.",
        )
    if db.query(Driver).filter(Driver.ambulance_plate == body.ambulance_plate).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This ambulance plate is already registered.",
        )

    # Create User
    user = User(
        phone_number=body.phone_number,
        hashed_password=hash_password(body.password),
        role=UserRole.driver,
    )
    db.add(user)
    db.flush()  # get user.id without committing

    # Create Driver
    driver = Driver(
        user_id=user.id,
        license_number=body.license_number,
        ambulance_plate=body.ambulance_plate.upper(),
        is_online=False,
    )
    db.add(driver)
    db.commit()
    db.refresh(driver)

    return SignupResponse(
        user_id=user.id,
        driver_id=driver.id,
        phone_number=user.phone_number,
        ambulance_plate=driver.ambulance_plate,
        message="Driver account created successfully.",
    )


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Authenticate and receive a JWT access token",
)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> TokenResponse:
    user = db.query(User).filter(User.phone_number == body.phone_number).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect phone number or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(data={"sub": str(user.id)})
    return TokenResponse(access_token=token, token_type="bearer")


# ── /auth/me ──────────────────────────────────────────────────────────────────

from backend.auth import get_current_driver  # noqa: E402 (local import to avoid circular)


@router.get(
    "/me",
    summary="Get current driver profile",
    tags=["Auth"],
)
def me(
    db: Session = Depends(get_db),
    current_driver=Depends(get_current_driver),
):
    user = db.query(User).filter(User.id == current_driver.user_id).first()
    return {
        "driver_id":      current_driver.id,
        "user_id":        current_driver.user_id,
        "phone_number":   user.phone_number if user else None,
        "license_number": current_driver.license_number,
        "ambulance_plate": current_driver.ambulance_plate,
        "is_online":      current_driver.is_online,
        "current_lat":    current_driver.current_lat,
        "current_lng":    current_driver.current_lng,
    }


class OnlineToggleRequest(BaseModel):
    is_online: bool
    lat: float | None = None
    lng: float | None = None


@router.patch(
    "/me/online",
    summary="Toggle driver online / offline status",
    tags=["Auth"],
)
def set_online(
    body: OnlineToggleRequest,
    db: Session = Depends(get_db),
    current_driver=Depends(get_current_driver),
):
    current_driver.is_online = body.is_online
    if body.lat is not None:
        current_driver.current_lat = body.lat
    if body.lng is not None:
        current_driver.current_lng = body.lng
    db.commit()
    return {"is_online": current_driver.is_online, "driver_id": current_driver.id}

