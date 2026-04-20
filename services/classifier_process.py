"""
Hybrid NMF + SVM document classifier — first stage of the pipeline.

Exposes `svm_classify(text)` which returns ranked candidate categories
with confidence scores.  These are passed to the LLM (llm_process.py)
for final category selection and tagging.
"""

import os
from sklearn.feature_extraction.text import TfidfVectorizer
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


def svm_classify(text: str, top_n: int = 5) -> list[dict]:
    """
    Run the SVM classifier and return the top-N candidate categories
    with confidence scores.  This is the first stage of the pipeline —
    it narrows down the category search space for the LLM.

    Returns a list like:
      [{"category": "Research", "confidence": 0.42}, ...]
    sorted by descending confidence.
    """
    try:
        X_tfidf = _tfidf.transform([text[:4000]])
        X_nmf = _nmf.transform(X_tfidf)
        X_combined = hstack([X_tfidf, X_nmf])

        probs = _svm.predict_proba(X_combined)[0]
        top_idx = probs.argsort()[-top_n:][::-1]
        classes = _svm.classes_

        return [
            {"category": str(classes[i]), "confidence": round(float(probs[i]), 4)}
            for i in top_idx
            if probs[i] > 0
        ]

    except Exception as e:
        print("svm_classify error:", e)
        return []
