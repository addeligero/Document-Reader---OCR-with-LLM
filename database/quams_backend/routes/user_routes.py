from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request
from werkzeug.security import generate_password_hash

from ..auth import login_required
from ..constants import PRIVILEGED_USER_ROLES
from ..extensions import db
from ..models import User
from ..serializers import user_json


users_bp = Blueprint("users", __name__, url_prefix="/api/users")


def require_user_admin(user: User):
    if user.role not in PRIVILEGED_USER_ROLES:
        return jsonify({"error": "Forbidden"}), 403
    return None


@users_bp.get("")
@login_required
def list_users(user: User):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    rows = User.query.order_by(User.created_at.desc()).all()
    return jsonify([user_json(row) for row in rows])


@users_bp.post("")
@login_required
def create_user(user: User):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    payload = request.get_json(force=True)
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", "")).strip()
    f_name = str(payload.get("f_name", "")).strip()
    l_name = str(payload.get("l_name", "")).strip()

    if not username or not password or not f_name or not l_name:
        return jsonify({"error": "username, password, f_name, and l_name are required"}), 400

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 409

    new_user = User(
        username=username,
        password_hash=generate_password_hash(password),
        f_name=f_name,
        l_name=l_name,
        email=payload.get("email"),
        role=payload.get("role") or "user",
        department=payload.get("department"),
        status=payload.get("status", True),
        is_taskforce=payload.get("is_taskforce", False),
        avatar=payload.get("avatar"),
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.session.add(new_user)
    db.session.commit()
    return jsonify(user_json(new_user)), 201


@users_bp.patch("/<user_id>")
@login_required
def update_user(user: User, user_id: str):
    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    payload = request.get_json(force=True)
    if user.id == target.id:
        allowed_self_fields = {"f_name", "l_name", "avatar"}
        disallowed_fields = set(payload) - allowed_self_fields
        if disallowed_fields:
            return jsonify({"error": "Forbidden"}), 403

        for field in allowed_self_fields:
            if field in payload:
                setattr(target, field, payload[field])

        target.updated_at = datetime.now(timezone.utc)
        db.session.commit()
        return jsonify(user_json(target))

    blocked = require_user_admin(user)
    if blocked:
        return blocked

    if "username" in payload:
        username = str(payload.get("username", "")).strip().lower()
        if not username:
            return jsonify({"error": "username cannot be empty"}), 400
        existing = User.query.filter(User.username == username, User.id != target.id).first()
        if existing:
            return jsonify({"error": "Username already exists"}), 409
        target.username = username

    for field in ["f_name", "l_name", "email", "department", "is_taskforce", "avatar"]:
        if field in payload:
            setattr(target, field, payload[field])

    target.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify(user_json(target))


@users_bp.delete("/<user_id>")
@login_required
def delete_user(user: User, user_id: str):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    db.session.delete(target)
    db.session.commit()
    return jsonify({"ok": True})


@users_bp.post("/<user_id>/reset-password")
@login_required
def reset_password(user: User, user_id: str):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    payload = request.get_json(force=True)
    password = str(payload.get("password", "")).strip()
    if not password:
        return jsonify({"error": "password is required"}), 400

    target.password_hash = generate_password_hash(password)
    target.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"ok": True})


@users_bp.patch("/<user_id>/activate")
@login_required
def activate_user(user: User, user_id: str):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    target.status = True
    target.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify(user_json(target))


@users_bp.patch("/<user_id>/deactivate")
@login_required
def deactivate_user(user: User, user_id: str):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    target.status = False
    target.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify(user_json(target))


@users_bp.patch("/<user_id>/role")
@login_required
def update_role(user: User, user_id: str):
    blocked = require_user_admin(user)
    if blocked:
        return blocked

    target = db.session.get(User, user_id)
    if not target:
        return jsonify({"error": "User not found"}), 404

    payload = request.get_json(force=True)
    role = str(payload.get("role", "")).strip()
    if not role:
        return jsonify({"error": "role is required"}), 400

    target.role = role
    target.updated_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify(user_json(target))
