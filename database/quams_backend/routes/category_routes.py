from __future__ import annotations

from flask import Blueprint, jsonify

from ..auth import login_required
from ..models import Category, User


categories_bp = Blueprint("categories", __name__, url_prefix="/api/categories")


@categories_bp.get("")
@login_required
def categories(user: User):
    rows = Category.query.order_by(Category.id.asc()).all()
    return jsonify([{"id": row.id, "name": row.name} for row in rows])
