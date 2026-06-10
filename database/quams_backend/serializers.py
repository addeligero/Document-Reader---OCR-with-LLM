from __future__ import annotations

from .models import (
    Catergory,
    ComplianceAccreditation,
    ComplianceItem,
    ComplianceItemDocument,
    ComplianceRequirementCategory,
    Document,
    Notification,
    User,
)


def user_json(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "f_name": user.f_name,
        "l_name": user.l_name,
        "email": user.email,
        "role": user.role,
        "department": user.department,
        "status": user.status,
        "is_taskforce": user.is_taskforce,
        "avatar": user.avatar,
        "last_sign_in_at": user.last_sign_in_at.isoformat() if user.last_sign_in_at else None,
    }


def document_json(doc: Document) -> dict:
    return {
        "id": doc.id,
        "created_at": doc.created_at.isoformat(),
        "updated_at": doc.updated_at.isoformat(),
        "user_id": doc.user_id,
        "file_name": doc.file_name,
        "primary_category": doc.primary_category,
        "secondary_category": doc.secondary_category,
        "tags": doc.tags or [],
        "path": doc.path,
        "status": doc.status,
        "extracted_text": doc.extracted_text,
        "uploaded_by": f"{doc.user.f_name} {doc.user.l_name}".strip() if doc.user else "Unknown User",
    }


def notification_json(notification: Notification) -> dict:
    return {
        "id": notification.id,
        "created_at": notification.created_at.isoformat(),
        "user_id": notification.user_id,
        "title": notification.title,
        "message": notification.message,
        "type": notification.type,
        "read": notification.read,
        "link": notification.link,
        "metadata": notification.metadata_json or {},
    }


def compliance_accreditation_json(row: ComplianceAccreditation) -> dict:
    return {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "name": row.name,
        "requirements": row.requirements or [],
    }


def compliance_item_document_json(row: ComplianceItemDocument) -> dict:
    doc = row.document
    return {
        "id": row.document_id,
        "file_name": doc.file_name if doc else None,
        "primary_category": doc.primary_category if doc else None,
    }


def compliance_item_json(row: ComplianceItem, include_documents: bool = True) -> dict:
    payload = {
        "id": row.id,
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
        "accreditation": row.accreditation,
        "requirements": row.requirements or [],
        "remarks": row.remarks,
        "mandatory": row.mandatory or [],
        "enhancement": row.enhancement or [],
        "status": row.status,
    }
    if include_documents:
        payload["supporting_documents"] = [compliance_item_document_json(item) for item in row.documents]
    return payload


def compliance_requirement_category_json(row: ComplianceRequirementCategory) -> dict:
    return {
        "accreditation_name": row.accreditation_name,
        "requirement_key": row.requirement_key,
        "category_id": row.category_id,
    }


def compliance_category_json(row: Catergory) -> dict:
    return {"id": row.id, "name": row.name}
