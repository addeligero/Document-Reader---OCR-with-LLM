from __future__ import annotations

import os

from werkzeug.security import generate_password_hash

from .extensions import db
from .models import AppSetting, Category, Catergory, ComplianceAccreditation, User


DEFAULT_ACCREDITATIONS = {
    "AACCUP": [
        "Area 1 - Vision, Mission, Goals and Objectives",
        "Area 2 - Faculty",
        "Area 3 - Curriculum and Instruction",
        "Area 4 - Support to Students",
        "Area 5 - Research",
        "Area 6 - Extension and Community Involvement",
        "Area 7 - Library",
        "Area 8 - Physical Plant and Facilities",
        "Area 9 - Laboratories",
        "Area 10 - Administration",
    ],
    "PICAB": [
        "1.0 Background Information",
        "2.0 Institutional Summary",
        "3.0 Program Educational Objectives",
        "4.0 Program Outcomes (Student Outcomes)",
        "5.0 Curriculum",
        "6.0 Students",
        "7.0 Faculty",
        "8.0 Facilities",
        "9.0 Institutional Support",
        "10.0 Industry-Academe Linkage and Community Programs",
        "11.0 Program Improvement",
    ],
    "COE": [
        "Criterion 1 - Innovation Culture",
        "Criterion 2 - Staff Development Tradition",
        "Criterion 3 - Learner and Graduate Quality",
        "Criterion 4 - Culture of Research and Creativity",
        "Criterion 5 - International Outlook",
        "Criterion 6 - Service Orientation",
    ],
    "AUN-QA": [
        "Criterion 1 - University Information",
        "Criterion 2 - Programme Structure and Content",
        "Criterion 3 - Teaching and Learning Approach",
        "Criterion 4 - Academic Staff",
        "Criterion 5 - Academic Staff Support",
        "Criterion 6 - Student Support Services",
        "Criterion 7 - Facilities and Infrastructure",
        "Criterion 8 - Output and Outcomes",
    ],
    "ISO": [],
}


def seed_defaults() -> None:
    if not AppSetting.query.get("two_factor_required"):
        db.session.add(AppSetting(key="two_factor_required", value=os.getenv("TWO_FACTOR_REQUIRED", "false")))
    if not AppSetting.query.get("session_timeout_minutes"):
        db.session.add(AppSetting(key="session_timeout_minutes", value=os.getenv("SESSION_TIMEOUT_MINUTES", "30")))
    if not User.query.filter_by(username="admin").first():
        db.session.add(
            User(
                username="admin",
                password_hash=generate_password_hash("Quams123"),
                f_name="System",
                l_name="Admin",
                email="admin@quams.local",
                role="admin",
            )
        )
    default_categories = ["Administration", "Instruction", "Research", "Extension"]
    if Category.query.count() == 0:
        db.session.add_all([Category(name=name) for name in default_categories])
    if Catergory.query.count() == 0:
        db.session.add_all([Catergory(name=name) for name in default_categories])
    for name, requirements in DEFAULT_ACCREDITATIONS.items():
        if not ComplianceAccreditation.query.filter_by(name=name).first():
            db.session.add(ComplianceAccreditation(name=name, requirements=requirements))
    db.session.commit()
