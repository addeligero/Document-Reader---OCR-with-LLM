from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request, send_file
from werkzeug.utils import secure_filename

from ..auth import login_required
from ..constants import ALLOWED_EXTENSIONS, PRIVILEGED_DOCUMENT_ROLES
from ..extensions import db
from ..models import Document, Notification, User
from ..ocr_service import normalize_text, run_ocr
from ..serializers import document_json


documents_bp = Blueprint("documents", __name__, url_prefix="/api/documents")


def can_edit_document(user: User, doc: Document) -> bool:
    return doc.user_id == user.id or user.role in PRIVILEGED_DOCUMENT_ROLES


def add_notification(
    user_id: str,
    title: str,
    message: str,
    notification_type: str = "info",
    link: str | None = None,
    metadata: dict | None = None,
) -> None:
    db.session.add(
        Notification(
            user_id=user_id,
            title=title,
            message=message,
            type=notification_type,
            link=link,
            metadata_json=metadata or {},
        )
    )


def notify_reviewers_pending_document(doc: Document) -> None:
    reviewers = User.query.filter(
        db.or_(User.role.in_(PRIVILEGED_DOCUMENT_ROLES), User.is_taskforce.is_(True)),
        User.status.is_(True),
    ).all()
    for reviewer in reviewers:
        add_notification(
            reviewer.id,
            "New document pending review",
            f'"{doc.file_name}" is ready for checking.',
            "info",
            "/classification",
            {"document_id": doc.id, "status": doc.status},
        )


def notify_document_status_change(actor: User, doc: Document, previous_status: str | None) -> None:
    if doc.status == previous_status or doc.status not in {"approved", "rejected"}:
        return

    notification_type = "success" if doc.status == "approved" else "warning"
    title = "Document approved" if doc.status == "approved" else "Document rejected"
    message = f'"{doc.file_name}" was {doc.status}.'
    if actor.id != doc.user_id:
        message += f" Reviewed by {actor.f_name} {actor.l_name}."

    add_notification(
        doc.user_id,
        title,
        message,
        notification_type,
        "/classification",
        {"document_id": doc.id, "status": doc.status},
    )


@documents_bp.get("")
@login_required
def list_documents(user: User):
    status = request.args.get("status")
    query = Document.query.order_by(Document.created_at.desc())
    if status:
        query = query.filter(Document.status == status)
    return jsonify([document_json(doc) for doc in query.all()])


@documents_bp.post("/upload")
@login_required
def upload_document(user: User):
    uploaded = request.files.get("file")
    if not uploaded:
        return jsonify({"error": "Missing file"}), 400

    original_name = secure_filename(uploaded.filename or "document")
    extension = Path(original_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Unsupported file type"}), 400

    existing_title = Document.query.filter_by(file_name=original_name).first()
    if existing_title:
        return jsonify({"error": "A document with this title already exists"}), 409

    relative_path = f"{user.id}/{uuid.uuid4()}-{original_name}"
    full_path = current_app.config["UPLOAD_DIR"] / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded.save(full_path)

    doc = Document(user_id=user.id, file_name=original_name, path=relative_path, status="processing")
    db.session.add(doc)
    db.session.flush()
    add_notification(
        user.id,
        "Document uploaded",
        f'"{original_name}" was uploaded and is now being processed for checking.',
        "info",
        "/upload",
        {"document_id": doc.id, "status": doc.status},
    )
    db.session.commit()

    try:
        ocr = run_ocr(full_path, original_name)
        extracted_text = ocr.get("text")
        normalized = normalize_text(extracted_text)
        if normalized:
            duplicates = Document.query.filter(Document.id != doc.id, Document.extracted_text.isnot(None)).all()
            if any(normalize_text(item.extracted_text) == normalized for item in duplicates):
                doc.status = "error"
                doc.updated_at = datetime.now(timezone.utc)
                add_notification(
                    user.id,
                    "Document upload failed",
                    f'"{doc.file_name}" has duplicate extracted text and cannot proceed to checking.',
                    "error",
                    "/upload",
                    {"document_id": doc.id, "status": doc.status},
                )
                db.session.commit()
                return jsonify({"error": "Duplicate content: extracted text already exists", "document": document_json(doc)}), 409

        doc.file_name = ocr.get("filename") or original_name
        doc.primary_category = ocr.get("primary_category")
        doc.secondary_category = ocr.get("secondary_category")
        doc.tags = ocr.get("tags") or []
        doc.extracted_text = extracted_text
        doc.status = "pending"
        doc.updated_at = datetime.now(timezone.utc)
        add_notification(
            user.id,
            "Document ready for checking",
            f'"{doc.file_name}" was processed successfully and is now pending review.',
            "success",
            "/upload",
            {"document_id": doc.id, "status": doc.status},
        )
        notify_reviewers_pending_document(doc)
        db.session.commit()
    except Exception as exc:
        doc.status = "error"
        doc.updated_at = datetime.now(timezone.utc)
        add_notification(
            user.id,
            "Document processing failed",
            f'"{doc.file_name}" failed during OCR/classification.',
            "error",
            "/upload",
            {"document_id": doc.id, "status": doc.status},
        )
        db.session.commit()
        return jsonify({"error": f"OCR processing failed: {exc}", "document": document_json(doc)}), 502

    return jsonify(document_json(doc)), 201


@documents_bp.patch("/<doc_id>")
@login_required
def update_document(user: User, doc_id: str):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    if not can_edit_document(user, doc):
        return jsonify({"error": "Forbidden"}), 403

    payload = request.get_json(force=True)
    previous_status = doc.status
    for field in ["file_name", "primary_category", "secondary_category", "tags", "status", "extracted_text"]:
        if field in payload:
            setattr(doc, field, payload[field])
    doc.updated_at = datetime.now(timezone.utc)
    notify_document_status_change(user, doc, previous_status)
    db.session.commit()
    return jsonify(document_json(doc))


@documents_bp.delete("/<doc_id>")
@login_required
def delete_document(user: User, doc_id: str):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404
    if not can_edit_document(user, doc):
        return jsonify({"error": "Forbidden"}), 403

    full_path = current_app.config["UPLOAD_DIR"] / doc.path
    if full_path.exists():
        full_path.unlink()
    db.session.delete(doc)
    db.session.commit()
    return jsonify({"ok": True})


@documents_bp.get("/<doc_id>/download")
@login_required
def download_document(user: User, doc_id: str):
    doc = db.session.get(Document, doc_id)
    if not doc:
        return jsonify({"error": "Document not found"}), 404

    full_path = current_app.config["UPLOAD_DIR"] / doc.path
    if not full_path.exists():
        return jsonify({"error": "File missing"}), 404
    inline = request.args.get("inline") in {"1", "true", "yes"}
    return send_file(full_path, as_attachment=not inline, download_name=doc.file_name)
