from __future__ import annotations

import uuid
from datetime import datetime, timezone

from .extensions import db


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    f_name = db.Column(db.String(120), nullable=False)
    l_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(80), nullable=False, default="user")
    department = db.Column(db.String(120), nullable=True)
    status = db.Column(db.Boolean, nullable=False, default=True)
    is_taskforce = db.Column(db.Boolean, nullable=False, default=False)
    avatar = db.Column(db.String(500), nullable=True)
    mfa_secret = db.Column(db.String(64), nullable=True)
    mfa_enabled = db.Column(db.Boolean, nullable=False, default=False)
    last_sign_in_at = db.Column(db.DateTime(timezone=True), nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False)
    file_name = db.Column(db.String(255), nullable=False)
    primary_category = db.Column(db.String(120), nullable=True)
    secondary_category = db.Column(db.String(120), nullable=True)
    tags = db.Column(db.JSON, nullable=False, default=list)
    path = db.Column(db.String(500), nullable=False, unique=True)
    status = db.Column(db.String(40), nullable=False, default="pending")
    extracted_text = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    user = db.relationship("User")


class AppSetting(db.Model):
    __tablename__ = "app_settings"

    key = db.Column(db.String(120), primary_key=True)
    value = db.Column(db.String(255), nullable=False)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Category(db.Model):
    __tablename__ = "categories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Catergory(db.Model):
    __tablename__ = "catergories"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), unique=True, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)


class Notification(db.Model):
    __tablename__ = "notifications"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    user_id = db.Column(db.String(36), db.ForeignKey("users.id"), nullable=False, index=True)
    title = db.Column(db.String(255), nullable=False)
    message = db.Column(db.Text, nullable=False)
    type = db.Column(db.String(40), nullable=False, default="info")
    read = db.Column(db.Boolean, nullable=False, default=False)
    link = db.Column(db.String(500), nullable=True)
    metadata_json = db.Column("metadata", db.JSON, nullable=True)

    user = db.relationship("User")


class ComplianceAccreditation(db.Model):
    __tablename__ = "compliance_accreditations"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    name = db.Column(db.String(120), unique=True, nullable=False)
    requirements = db.Column(db.JSON, nullable=False, default=list)


class ComplianceItem(db.Model):
    __tablename__ = "compliance_items"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    accreditation = db.Column(db.String(120), nullable=False, index=True)
    requirements = db.Column(db.JSON, nullable=False, default=list)
    remarks = db.Column(db.Text, nullable=False, default="")
    mandatory = db.Column(db.JSON, nullable=False, default=list)
    enhancement = db.Column(db.JSON, nullable=False, default=list)
    status = db.Column(db.String(40), nullable=False, default="pending")

    documents = db.relationship(
        "ComplianceItemDocument",
        cascade="all, delete-orphan",
        back_populates="compliance_item",
    )


class ComplianceItemDocument(db.Model):
    __tablename__ = "compliance_item_documents"
    __table_args__ = (db.UniqueConstraint("compliance_item_id", "document_id"),)

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    compliance_item_id = db.Column(
        db.String(36),
        db.ForeignKey("compliance_items.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id = db.Column(
        db.String(36),
        db.ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    compliance_item = db.relationship("ComplianceItem", back_populates="documents")
    document = db.relationship("Document")


class ComplianceRequirementCategory(db.Model):
    __tablename__ = "compliance_requirement_categories"
    __table_args__ = (
        db.UniqueConstraint("accreditation_name", "requirement_key", "category_id"),
    )

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    accreditation_name = db.Column(db.String(120), nullable=False, index=True)
    requirement_key = db.Column(db.String(255), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("catergories.id", ondelete="CASCADE"), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)

    category = db.relationship("Catergory")
