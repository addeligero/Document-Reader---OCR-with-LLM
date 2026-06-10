from __future__ import annotations

from flask import Flask

from .auth_routes import auth_bp
from .category_routes import categories_bp
from .compliance_routes import (
    compliance_accreditations_bp,
    compliance_categories_bp,
    compliance_items_bp,
    compliance_requirement_categories_bp,
)
from .document_routes import documents_bp
from .notification_routes import notifications_bp
from .settings_routes import settings_bp
from .user_routes import users_bp


def register_blueprints(app: Flask) -> None:
    app.register_blueprint(auth_bp)
    app.register_blueprint(settings_bp)
    app.register_blueprint(categories_bp)
    app.register_blueprint(documents_bp)
    app.register_blueprint(users_bp)
    app.register_blueprint(notifications_bp)
    app.register_blueprint(compliance_items_bp)
    app.register_blueprint(compliance_accreditations_bp)
    app.register_blueprint(compliance_categories_bp)
    app.register_blueprint(compliance_requirement_categories_bp)
