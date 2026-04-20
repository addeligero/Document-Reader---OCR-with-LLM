"""
Train a hybrid LDA + SVM document classifier.

Usage:
    python -m services.train_classifier --data labeled_docs.csv

The CSV must have two columns:
    text     – the document text (OCR output, etc.)
    category – the ground-truth label (must be one of CATEGORIES)

The script saves three artefacts to  models/ :
    tfidf_vectorizer.joblib   – fitted TF-IDF vectorizer
    lda_model.joblib          – fitted LDA topic model
    svm_classifier.joblib     – fitted SVM (trained on TF-IDF + LDA features)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import joblib

# ── same category list used everywhere ──
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

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train(csv_path: str, n_topics: int = 30):
    """Train and persist the hybrid LDA+SVM classifier."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── load data ──
    df = pd.read_csv(csv_path)
    assert "text" in df.columns and "category" in df.columns, \
        "CSV must contain 'text' and 'category' columns."

    # keep only rows whose category is in the allowed list
    df = df[df["category"].isin(CATEGORIES)].reset_index(drop=True)
    if df.empty:
        sys.exit("No rows with valid categories found.")

    texts = df["text"].fillna("").tolist()
    labels = df["category"].tolist()

    # ── 1. TF-IDF ──
    tfidf = TfidfVectorizer(
        max_features=10_000,
        sublinear_tf=True,
        stop_words="english",
    )
    X_tfidf = tfidf.fit_transform(texts)

    # ── 2. LDA on TF-IDF ──
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        learning_method="online",
        max_iter=20,
    )
    X_lda = lda.fit_transform(X_tfidf)

    # ── 3. Combine features ──
    X_combined = hstack([X_tfidf, X_lda])

    # ── 4. Train SVM ──
    svm = SVC(
        kernel="linear",
        C=1.0,
        probability=True,   # needed so we can get top-2 predictions
        random_state=42,
    )
    svm.fit(X_combined, labels)
    # ── quick cross-val report ──
    scores = cross_val_score(svm, X_combined, labels, cv=min(5, len(df)), scoring="accuracy")
    print(f"Cross-val accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # ── persist ──
    joblib.dump(tfidf, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(lda,   os.path.join(MODEL_DIR, "lda_model.joblib"))
    joblib.dump(svm,   os.path.join(MODEL_DIR, "svm_classifier.joblib"))
    print(f"Models saved to {os.path.abspath(MODEL_DIR)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDA+SVM doc classifier")
    parser.add_argument("--data", required=True, help="Path to labeled CSV")
    parser.add_argument("--topics", type=int, default=30, help="Number of LDA topics")
    args = parser.parse_args()
    train(args.data, n_topics=args.topics)
