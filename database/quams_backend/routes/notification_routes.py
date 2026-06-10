from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..auth import login_required
from ..constants import PRIVILEGED_USER_ROLES
from ..extensions import db
from ..models import Notification, User
from ..serializers import notification_json


notifications_bp = Blueprint("notifications", __name__, url_prefix="/api/notifications")


def can_manage_notification(user: User, notification: Notification) -> bool:
    return notification.user_id == user.id or user.role in PRIVILEGED_USER_ROLES


def require_user_admin(user: User):
    if user.role not in PRIVILEGED_USER_ROLES:
        return jsonify({"error": "Forbidden"}), 403
    return None


@notifications_bp.get("")
@login_required
def list_notifications(user: User):
    rows = (
        Notification.query.filter_by(user_id=user.id)
        .order_by(Notification.created_at.desc())
        .all()
    )
    return jsonify([notification_json(row) for row in rows])


@notifications_bp.post("")
@login_required
def create_notification(user: User):
    payload = request.get_json(force=True)
    user_id = str(payload.get("user_id", user.id)).strip()

    if user_id != user.id:
        blocked = require_user_admin(user)
        if blocked:
            return blocked

    title = str(payload.get("title", "")).strip()
    message = str(payload.get("message", "")).strip()
    if not title or not message:
        return jsonify({"error": "title and message are required"}), 400

    target_user = db.session.get(User, user_id)
    if not target_user:
        return jsonify({"error": "User not found"}), 404

    notification = Notification(
        user_id=target_user.id,
        title=title,
        message=message,
        type=payload.get("type") or "info",
        read=bool(payload.get("read", False)),
        link=payload.get("link"),
        metadata_json=payload.get("metadata") or {},
        created_at=datetime.now(timezone.utc),
    )
    db.session.add(notification)
    db.session.commit()
    return jsonify(notification_json(notification)), 201


@notifications_bp.patch("/<notification_id>/read")
@login_required
def mark_read(user: User, notification_id: str):
    notification = db.session.get(Notification, notification_id)
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    if not can_manage_notification(user, notification):
        return jsonify({"error": "Forbidden"}), 403

    notification.read = True
    db.session.commit()
    return jsonify(notification_json(notification))


@notifications_bp.patch("/read-all")
@login_required
def mark_all_read(user: User):
    rows = Notification.query.filter_by(user_id=user.id, read=False).all()
    for notification in rows:
        notification.read = True
    db.session.commit()
    return jsonify({"ok": True, "updated_count": len(rows)})


@notifications_bp.delete("/<notification_id>")
@login_required
def delete_notification(user: User, notification_id: str):
    notification = db.session.get(Notification, notification_id)
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    if not can_manage_notification(user, notification):
        return jsonify({"error": "Forbidden"}), 403

    db.session.delete(notification)
    db.session.commit()
    return jsonify({"ok": True})
