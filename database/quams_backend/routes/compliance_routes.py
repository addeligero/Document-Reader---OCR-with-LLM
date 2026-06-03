from __future__ import annotations

from datetime import datetime, timezone

from flask import Blueprint, jsonify, request

from ..auth import login_required
from ..extensions import db
from ..models import (
    Catergory,
    ComplianceAccreditation,
    ComplianceItem,
    ComplianceItemDocument,
    ComplianceRequirementCategory,
    Document,
    User,
)
from ..serializers import (
    compliance_accreditation_json,
    compliance_category_json,
    compliance_item_json,
    compliance_requirement_category_json,
)


compliance_items_bp = Blueprint("compliance_items", __name__, url_prefix="/api/compliance-items")
compliance_accreditations_bp = Blueprint(
    "compliance_accreditations",
    __name__,
    url_prefix="/api/compliance-accreditations",
)
compliance_categories_bp = Blueprint(
    "compliance_categories",
    __name__,
    url_prefix="/api/compliance-categories",
)
compliance_requirement_categories_bp = Blueprint(
    "compliance_requirement_categories",
    __name__,
    url_prefix="/api/compliance-requirement-categories",
)

VALID_COMPLIANCE_STATUSES = {"met", "pending", "not_met"}


def _now():
    return datetime.now(timezone.utc)


def _list_value(payload: dict, key: str) -> list:
    value = payload.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _document_ids_from_payload(payload: dict) -> list[str]:
    raw_items = (
        payload.get("supporting_documents")
        if "supporting_documents" in payload
        else payload.get("document_ids", payload.get("documents", []))
    )
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        raw_items = [raw_items]

    document_ids = []
    for item in raw_items:
        if isinstance(item, dict):
            doc_id = item.get("id") or item.get("document_id")
        else:
            doc_id = item
        if doc_id:
            document_ids.append(str(doc_id))
    return list(dict.fromkeys(document_ids))


def _replace_supporting_documents(item: ComplianceItem, document_ids: list[str]) -> tuple[bool, str | None]:
    documents = Document.query.filter(Document.id.in_(document_ids)).all() if document_ids else []
    found_ids = {doc.id for doc in documents}
    missing_ids = [doc_id for doc_id in document_ids if doc_id not in found_ids]
    if missing_ids:
        return False, f"Document not found: {', '.join(missing_ids)}"

    item.documents = [
        ComplianceItemDocument(compliance_item_id=item.id, document_id=document_id)
        for document_id in document_ids
    ]
    return True, None


def _apply_compliance_item_payload(item: ComplianceItem, payload: dict) -> tuple[bool, str | None]:
    if "accreditation" in payload:
        item.accreditation = str(payload.get("accreditation", "")).strip()
    if "requirements" in payload:
        item.requirements = _list_value(payload, "requirements")
    if "remarks" in payload:
        item.remarks = str(payload.get("remarks") or "")
    if "mandatory" in payload:
        item.mandatory = _list_value(payload, "mandatory")
    if "enhancement" in payload:
        item.enhancement = _list_value(payload, "enhancement")
    if "status" in payload:
        status = str(payload.get("status", "")).strip()
        if status not in VALID_COMPLIANCE_STATUSES:
            return False, "status must be one of: met, pending, not_met"
        item.status = status

    if not item.accreditation:
        return False, "accreditation is required"
    return True, None


@compliance_items_bp.get("")
@login_required
def list_compliance_items(user: User):
    rows = ComplianceItem.query.order_by(ComplianceItem.created_at.desc()).all()
    return jsonify([compliance_item_json(row) for row in rows])


@compliance_items_bp.post("")
@login_required
def create_compliance_item(user: User):
    payload = request.get_json(force=True)
    item = ComplianceItem(
        accreditation=str(payload.get("accreditation", "")).strip(),
        requirements=_list_value(payload, "requirements"),
        remarks=str(payload.get("remarks") or ""),
        mandatory=_list_value(payload, "mandatory"),
        enhancement=_list_value(payload, "enhancement"),
        status=str(payload.get("status") or "pending"),
        created_at=_now(),
        updated_at=_now(),
    )

    ok, error = _apply_compliance_item_payload(item, payload)
    if not ok:
        return jsonify({"error": error}), 400

    db.session.add(item)
    db.session.flush()

    ok, error = _replace_supporting_documents(item, _document_ids_from_payload(payload))
    if not ok:
        db.session.rollback()
        return jsonify({"error": error}), 404

    db.session.commit()
    return jsonify(compliance_item_json(item)), 201


@compliance_items_bp.patch("/<item_id>")
@login_required
def update_compliance_item(user: User, item_id: str):
    item = db.session.get(ComplianceItem, item_id)
    if not item:
        return jsonify({"error": "Compliance item not found"}), 404

    payload = request.get_json(force=True)
    ok, error = _apply_compliance_item_payload(item, payload)
    if not ok:
        return jsonify({"error": error}), 400

    if any(key in payload for key in ["supporting_documents", "document_ids", "documents"]):
        ok, error = _replace_supporting_documents(item, _document_ids_from_payload(payload))
        if not ok:
            db.session.rollback()
            return jsonify({"error": error}), 404

    item.updated_at = _now()
    db.session.commit()
    return jsonify(compliance_item_json(item))


@compliance_items_bp.patch("/<item_id>/status")
@login_required
def update_compliance_item_status(user: User, item_id: str):
    item = db.session.get(ComplianceItem, item_id)
    if not item:
        return jsonify({"error": "Compliance item not found"}), 404

    payload = request.get_json(force=True)
    status = str(payload.get("status", "")).strip()
    if status not in VALID_COMPLIANCE_STATUSES:
        return jsonify({"error": "status must be one of: met, pending, not_met"}), 400

    item.status = status
    item.updated_at = _now()
    db.session.commit()
    return jsonify(compliance_item_json(item))


@compliance_items_bp.delete("/<item_id>")
@login_required
def delete_compliance_item(user: User, item_id: str):
    item = db.session.get(ComplianceItem, item_id)
    if not item:
        return jsonify({"error": "Compliance item not found"}), 404

    db.session.delete(item)
    db.session.commit()
    return jsonify({"ok": True})


@compliance_accreditations_bp.get("")
@login_required
def list_compliance_accreditations(user: User):
    rows = ComplianceAccreditation.query.order_by(ComplianceAccreditation.name.asc()).all()
    return jsonify([compliance_accreditation_json(row) for row in rows])


@compliance_accreditations_bp.post("")
@login_required
def create_compliance_accreditation(user: User):
    payload = request.get_json(force=True)
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify({"error": "name is required"}), 400
    if ComplianceAccreditation.query.filter_by(name=name).first():
        return jsonify({"error": "Accreditation already exists"}), 409

    row = ComplianceAccreditation(
        name=name,
        requirements=_list_value(payload, "requirements"),
        created_at=_now(),
        updated_at=_now(),
    )
    db.session.add(row)
    db.session.commit()
    return jsonify(compliance_accreditation_json(row)), 201


@compliance_accreditations_bp.patch("/<path:name>")
@login_required
def update_compliance_accreditation(user: User, name: str):
    row = ComplianceAccreditation.query.filter_by(name=name).first()
    if not row:
        return jsonify({"error": "Accreditation not found"}), 404

    payload = request.get_json(force=True)
    if "name" in payload:
        new_name = str(payload.get("name", "")).strip()
        if not new_name:
            return jsonify({"error": "name cannot be empty"}), 400
        existing = ComplianceAccreditation.query.filter(
            ComplianceAccreditation.name == new_name,
            ComplianceAccreditation.id != row.id,
        ).first()
        if existing:
            return jsonify({"error": "Accreditation already exists"}), 409
        row.name = new_name
    if "requirements" in payload:
        row.requirements = _list_value(payload, "requirements")

    row.updated_at = _now()
    db.session.commit()
    return jsonify(compliance_accreditation_json(row))


@compliance_accreditations_bp.delete("/<path:name>")
@login_required
def delete_compliance_accreditation(user: User, name: str):
    row = ComplianceAccreditation.query.filter_by(name=name).first()
    if not row:
        return jsonify({"error": "Accreditation not found"}), 404

    db.session.delete(row)
    db.session.commit()
    return jsonify({"ok": True})


@compliance_categories_bp.get("")
@login_required
def list_compliance_categories(user: User):
    rows = Catergory.query.order_by(Catergory.id.asc()).all()
    return jsonify([compliance_category_json(row) for row in rows])


@compliance_categories_bp.post("")
@login_required
def create_compliance_category(user: User):
    payload = request.get_json(force=True)
    name = str(payload.get("name", "")).strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    category_id = payload.get("id")
    row = db.session.get(Catergory, int(category_id)) if category_id is not None else None
    if row:
        row.name = name
    else:
        existing = Catergory.query.filter_by(name=name).first()
        if existing:
            return jsonify({"error": "Category already exists"}), 409
        row = Catergory(id=int(category_id), name=name) if category_id is not None else Catergory(name=name)

    db.session.add(row)
    db.session.commit()
    return jsonify(compliance_category_json(row)), 201


@compliance_categories_bp.delete("/<int:category_id>")
@login_required
def delete_compliance_category(user: User, category_id: int):
    row = db.session.get(Catergory, category_id)
    if not row:
        return jsonify({"error": "Category not found"}), 404

    db.session.delete(row)
    db.session.commit()
    return jsonify({"ok": True})


@compliance_requirement_categories_bp.get("")
@login_required
def list_compliance_requirement_categories(user: User):
    rows = ComplianceRequirementCategory.query.order_by(
        ComplianceRequirementCategory.accreditation_name.asc(),
        ComplianceRequirementCategory.requirement_key.asc(),
    ).all()
    return jsonify([compliance_requirement_category_json(row) for row in rows])


@compliance_requirement_categories_bp.post("")
@login_required
def replace_compliance_requirement_categories(user: User):
    payload = request.get_json(force=True)
    accreditation_name = str(payload.get("accreditation_name", "")).strip()
    requirement_key = str(payload.get("requirement_key", "")).strip()
    category_ids = _list_value(payload, "category_ids")
    if "category_id" in payload and "category_ids" not in payload:
        category_ids = _list_value(payload, "category_id")

    if not accreditation_name or not requirement_key:
        return jsonify({"error": "accreditation_name and requirement_key are required"}), 400

    try:
        category_ids = [int(category_id) for category_id in category_ids]
    except (TypeError, ValueError):
        return jsonify({"error": "category_id/category_ids must be integers"}), 400
    categories = Catergory.query.filter(Catergory.id.in_(category_ids)).all() if category_ids else []
    found_ids = {category.id for category in categories}
    missing_ids = [str(category_id) for category_id in category_ids if category_id not in found_ids]
    if missing_ids:
        return jsonify({"error": f"Category not found: {', '.join(missing_ids)}"}), 404

    ComplianceRequirementCategory.query.filter_by(
        accreditation_name=accreditation_name,
        requirement_key=requirement_key,
    ).delete()
    rows = [
        ComplianceRequirementCategory(
            accreditation_name=accreditation_name,
            requirement_key=requirement_key,
            category_id=category_id,
        )
        for category_id in category_ids
    ]
    db.session.add_all(rows)
    db.session.commit()
    return jsonify([compliance_requirement_category_json(row) for row in rows]), 201


@compliance_requirement_categories_bp.delete("")
@login_required
def delete_compliance_requirement_categories(user: User):
    payload = request.get_json(silent=True) or request.args
    accreditation_name = str(payload.get("accreditation_name", "")).strip()
    requirement_key = str(payload.get("requirement_key", "")).strip()

    if not accreditation_name:
        return jsonify({"error": "accreditation_name is required"}), 400

    if not requirement_key:
        deleted_count = ComplianceRequirementCategory.query.filter_by(
            accreditation_name=accreditation_name,
        ).delete()
        db.session.commit()
        return jsonify({"ok": True, "deleted_count": deleted_count})

    deleted_count = ComplianceRequirementCategory.query.filter_by(
        accreditation_name=accreditation_name,
        requirement_key=requirement_key,
    ).delete()
    db.session.commit()
    return jsonify({"ok": True, "deleted_count": deleted_count})
