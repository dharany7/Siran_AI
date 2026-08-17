"""
backend/auth.py — JWT utilities and FastAPI auth dependencies.

Password hashing : bcrypt (called directly — passlib 1.7.4 is incompatible
                    with bcrypt >= 4 due to an internal self-test bug)
Token signing    : python-jose (HS256)

Public API
----------
hash_password(plain)           -> hashed str
verify_password(plain, hashed) -> bool
create_access_token(data)       -> JWT str
get_current_driver(token, db)   -> Driver ORM row  (FastAPI Depends)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db

# ── OAuth2 scheme (tokenUrl tells /docs where to POST credentials) ────────────

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ── Password hashing ──────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=settings.jwt_expire_days)
    )
    payload["exp"] = expire
    return jwt.encode(
        payload,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_current_driver(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    """
    Validate the bearer token and return the authenticated Driver row.
    Raises HTTP 401 on any failure (missing token, invalid signature, expired,
    driver not found).
    """
    from backend.models import Driver, User  # local import avoids circular

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id: Optional[int] = payload.get("sub")
        if user_id is None:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exc

    driver = db.query(Driver).filter(Driver.user_id == user.id).first()
    if driver is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User has no driver profile",
        )
    return driver
