"""
Hybrid NMF + SVM document classifier — drop-in replacement for llm_process.py.

Exposes the same `classify_document(text) -> dict` interface so that
app.py only needs to change its import line.
"""

import os
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from sklearn.svm import SVC
from scipy.sparse import hstack
import joblib

# ── categories (same as llm_process.py) ──
CATEGORIES = [
    "VMGO", "PEO", "PO", "Faculty", "Curriculum", "Instruction", "Students",
    "Research", "Extension", "Library", "Facilities", "Laboratories",
    "Administration", "Institutional Support", "Strategic Planning",
    "Special Orders", "DPCR", "IPCR", "Budget", "Activity Report",
    "Memorandum", "Minutes of Meeting", "Transmittal Letter", "Documentation",
    "Best Practice", "Audit", "Client Satisfactory", "Quality Objectives",
    "Risk Registers", "Trainings", "PES", "Faculty Advising",
    "Faculty Consultation", "Class Interventions", "Student Internship",
    "Approved Leave", "Daily Time Records (DTR)", "Faculty Fellowship Contracts",
    "Notarized Contracts", "Terms of Reference (TOR)", "Institutional Records",
    "Quality Assurance",
]

# ── load persisted models at import time (once) ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_tfidf: TfidfVectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
_nmf: NMF = joblib.load(os.path.join(MODEL_DIR, "nmf_model.joblib"))
_svm: SVC = joblib.load(os.path.join(MODEL_DIR, "svm_classifier.joblib"))

print("Hybrid NMF+SVM classifier loaded.")


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")


def _fallback_tags(text: str, top_n: int = 5) -> list[str]:
    """
    Frequency-based fallback when TF-IDF produces no tags (e.g. OCR text
    shares no vocabulary with the fitted vectorizer).
    """
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 3]
    if not tokens:
        return []
    return [w for w, _ in Counter(tokens).most_common(top_n)]


def _extract_tags(text: str, top_n: int = 5) -> list[str]:
    """
    Generate keyword tags from the TF-IDF vector of the document.
    Falls back to raw token frequency if the TF-IDF vector is empty.
    """
    vec = _tfidf.transform([text])
    feature_names = np.array(_tfidf.get_feature_names_out())
    scores = vec.toarray().flatten()
    top_indices = scores.argsort()[-top_n:][::-1]
    tags = [str(feature_names[i]) for i in top_indices if scores[i] > 0]
    if not tags:
        tags = _fallback_tags(text, top_n)
    return tags


def classify_document(text: str) -> dict:
    """
    Classify document text using the hybrid NMF+SVM model.

    Returns the same structure as the LLM version:
      {
        "primary_category":   str,
        "secondary_category": str,
        "tags":               list[str]
      }
    """
    try:
        # ── build feature vector ──
        X_tfidf = _tfidf.transform([text[:4000]])
        X_nmf = _nmf.transform(X_tfidf)
        X_combined = hstack([X_tfidf, X_nmf])

        # ── predict with probabilities ──
        probs = _svm.predict_proba(X_combined)[0]
        top2_idx = probs.argsort()[-2:][::-1]
        classes = _svm.classes_

        primary = str(classes[top2_idx[0]])
        secondary = str(classes[top2_idx[1]])

        tags = _extract_tags(text[:4000])

        return {
            "primary_category": primary,
            "secondary_category": secondary,
            "tags": tags,
        }

    except Exception as e:
        print("classify_document error:", e)
        return {
            "primary_category": None,
            "secondary_category": None,
            "tags": _fallback_tags(text[:4000]),
            "error": str(e),
        }
