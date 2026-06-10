from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..auth import login_required, privileged_required
from ..extensions import db
from ..models import AppSetting, User


settings_bp = Blueprint("settings", __name__, url_prefix="/api/settings")


@settings_bp.get("")
@login_required
def get_settings(user: User):
    rows = AppSetting.query.all()
    return jsonify({row.key: row.value for row in rows})


@settings_bp.patch("")
@login_required
@privileged_required
def update_settings(user: User):
    payload = request.get_json(force=True)
    for key in ["two_factor_required", "session_timeout_minutes"]:
        if key in payload:
            setting = AppSetting.query.get(key) or AppSetting(key=key, value="")
            setting.value = str(payload[key]).lower() if isinstance(payload[key], bool) else str(payload[key])
            setting.updated_at = datetime.now(timezone.utc)
            db.session.add(setting)
    db.session.commit()
    return jsonify({"ok": True})
