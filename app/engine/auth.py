import os
from datetime import datetime, timedelta, timezone
from typing import Annotated
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

import jwt
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.security import (
    OAuth2PasswordBearer,
    SecurityScopes,
)
from jwt.exceptions import InvalidTokenError
from pwdlib import PasswordHash
from pydantic import BaseModel, ValidationError

from app.schemas import TokenData, User, UserCreate
from app.engine.data_manager import get_user, upload_new_user
from app.models import UserData

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

password_hash = PasswordHash.recommended()

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token",
    scopes={
        # Authentication and User managment
        "auth:create_user": "Create a new user account in the system.",
        "auth:login": "Authenticate and recive access credentials.",
        "user:read": "View user information and user table entries.",
        "user:update": "modify user details such as role or status",
        "user:delete": "remove a user account from the system",
        # Market and Stock Data
        "market:read": "Read market data. Restricted",
        "stock:read": "Read stock data. Restricted",
        "market:delete_ticker_set": "Delete all market rows for a specific stock.",
        "stock:delete_ticker_set": "Delte all stock rows for a specific stock.",
        "stock:download": "Download historical data for a stock.",
        "market:download": "Download historical data for a S&P500 and NASDAQ.",
        # Table Viewing
        "table:view:market": "View the entire market table.",
        "table:view:stocks": "View the entire stock table.",
        "table:view:xgboost": "View the entire XGBoost model table.",
        "table:view:users": "View the entire users table.",
        # Model & Prediction
        "model:predict": "Generate a prediction using a model.",
        "model:train": "Train or retrain a model.",
        "model:read_metadata": "View model info, paramaters, and metadata. Restricted",
        "model:publish": "Promote a tained to active/production use."
        # System-Level
        "system:admin",
    },
)


def create_user(user: UserCreate):
    hashed_password = password_hash.hash(user.password)

    db_user = UserData(
        username=user.username,
        hashed_password=hashed_password,
        email=user.email,
        full_name=user.full_name,
        disabled=False,
    )

    upload_new_user(db_user)

    return db_user


def verify_password(plain_password, hashed_password):
    return password_hash.verify(plain_password, hashed_password)


def authenticate_user(username: str, password: str):
    user = get_user(username)
    if user is None:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    security_scopes: SecurityScopes, token: Annotated[str, Depends(oauth2_scheme)]
):
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": authenticate_value},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")

        if username is None:
            raise credentials_exception

        scope: str = payload.get("scope", "")
        token_scopes = scope.split(" ")
        token_data = TokenData(scopes=token_scopes, username=username)
    except (InvalidTokenError, ValidationError):
        raise credentials_exception

    user = get_user(username=token_data.username)

    if user is None:
        raise credentials_exception

    for scope in security_scopes.scopes:
        if scope not in token_data.scopes:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not enough permissions",
                headers={"WWW-Authenticate": authenticate_value},
            )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Security(get_current_user)],
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def init_admin():
    try:
        existing_admin = get_user("admin")
    except:
        existing_admin = False

    if not existing_admin:
        print("No admin found. Creating default admin...")

        hashed_password = password_hash.hash("default admin password")

        default_admin = UserData(
            username="admin",
            hashed_password=hashed_password,
            email="karol@jasniewicz.com",
            full_name="Default Administrator",
            disabled=False,
            scopes=[
                "auth:create_user",
                "auth:login",
                "user:read",
                "user:update",
                "user:delete",
                "model:predict",
                "model:train",
                "model:publish",
                "market:read",
                "stock:read",
                "market:delete_ticker_set",
                "stock:delete_ticker_set",
                "table:view:market",
                "table:view:stocks",
                "table:view:xgboost",
                "table:view:users",
                "model:predict",
                "model:train",
                "model:read_metadata",
                "model:publish",
                "stock:download",
                "market:download",
                "system:admin",
            ],
        )

        upload_new_user(default_admin)
        print("Default admin created!")
