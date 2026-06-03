from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import wraps

import jwt
from flask import current_app, jsonify, request

from .constants import PRIVILEGED_DOCUMENT_ROLES
from .extensions import db
from .models import User


def issue_token(user: User, minutes: int = 60 * 8) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "username": user.username,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=minutes)).timestamp()),
    }
    return jwt.encode(payload, current_app.config["JWT_SECRET"], algorithm="HS256")


def decode_token(token: str) -> dict:
    return jwt.decode(token, current_app.config["JWT_SECRET"], algorithms=["HS256"])


def current_user() -> User | None:
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
    elif request.args.get("token"):
        token = request.args.get("token", "").strip()

    if not token:
        return None

    try:
        payload = decode_token(token)
    except jwt.PyJWTError:
        return None

    return db.session.get(User, payload.get("sub"))


def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user or not user.status:
            return jsonify({"error": "Unauthorized"}), 401
        return fn(user, *args, **kwargs)

    return wrapper


def privileged_required(fn):
    @wraps(fn)
    def wrapper(user: User, *args, **kwargs):
        if user.role not in PRIVILEGED_DOCUMENT_ROLES:
            return jsonify({"error": "Forbidden"}), 403
        return fn(user, *args, **kwargs)

    return wrapper
