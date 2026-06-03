from __future__ import annotations

from datetime import datetime, timezone

import pyotp
from flask import Blueprint, jsonify, request
from werkzeug.security import check_password_hash

from ..auth import issue_token, login_required
from ..constants import PRIVILEGED_MFA_ROLES
from ..extensions import db
from ..models import User
from ..serializers import user_json
from ..settings import setting_bool


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


@auth_bp.post("/login")
def login():
    payload = request.get_json(force=True)
    username = str(payload.get("username", "")).strip().lower()
    password = str(payload.get("password", ""))
    mfa_code = str(payload.get("mfa_code", "")).strip()

    user = User.query.filter_by(username=username).first()
    if not user or not check_password_hash(user.password_hash, password):
        return jsonify({"error": "Invalid username or password"}), 401
    if not user.status:
        return jsonify({"error": "Account is deactivated"}), 403

    needs_mfa = setting_bool("two_factor_required") and user.role in PRIVILEGED_MFA_ROLES
    if needs_mfa:
        if not user.mfa_secret:
            user.mfa_secret = pyotp.random_base32()
            db.session.commit()

        totp = pyotp.TOTP(user.mfa_secret)
        if not mfa_code:
            return jsonify(
                {
                    "mfa_required": True,
                    "setup_required": not user.mfa_enabled,
                    "secret": user.mfa_secret if not user.mfa_enabled else None,
                    "otpauth_uri": totp.provisioning_uri(name=user.username, issuer_name="QuAMS")
                    if not user.mfa_enabled
                    else None,
                }
            ), 202

        if not totp.verify(mfa_code, valid_window=1):
            return jsonify({"error": "Invalid MFA code"}), 401

        user.mfa_enabled = True

    user.last_sign_in_at = datetime.now(timezone.utc)
    db.session.commit()
    return jsonify({"token": issue_token(user), "user": user_json(user)})


@auth_bp.get("/me")
@login_required
def me(user: User):
    return jsonify({"user": user_json(user)})


@auth_bp.post("/logout")
@login_required
def logout(user: User):
    return jsonify({"ok": True})
