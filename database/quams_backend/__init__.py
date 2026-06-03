from __future__ import annotations

import os
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from .extensions import db
from .routes import register_blueprints
from .seed import seed_defaults


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", "sqlite:///quams.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["JWT_SECRET"] = os.getenv("JWT_SECRET", "dev-secret-change-me")
    app.config["UPLOAD_DIR"] = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()

    CORS(app, supports_credentials=True)
    db.init_app(app)
    register_blueprints(app)

    with app.app_context():
        db.create_all()
        seed_defaults()
        app.config["UPLOAD_DIR"].mkdir(parents=True, exist_ok=True)

    return app
