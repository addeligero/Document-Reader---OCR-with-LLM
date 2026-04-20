"""
Hybrid NMF + SVM document classifier — first stage of the pipeline.

Exposes `svm_classify(text)` which returns ranked candidate categories
with confidence scores. These are passed to the LLM (llm_process.py)
for final category selection and tagging.
"""

import os
import re
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.svm import SVC

# ── preprocessing config (MUST match training: src/02_preprocessing.py + src/config.py) ──
LANGUAGE = "english"
MIN_TOKEN_LEN = 3
DOMAIN_STOP_WORDS = {
    "caraga", "csu", "ccis", "butuan", "ampayon", "city",
    "university", "college", "campus", "republic", "philippines",
    "office", "department",
    "iso", "page", "document", "file", "section", "annex",
    "reference", "number", "date", "revision", "copy", "form",
    "service", "name", "day", "time", "one", "two", "three",
    "state", "shall", "also", "per", "may", "based", "upon",
    "use", "used", "using", "well", "new", "make",
    "yang", "dan", "untuk", "dalam", "dapat", "dari", "dengan",
    "atau", "pada", "ini", "itu", "adalah", "akan", "oleh",
    "universitas", "mulia", "balikpapan", "icsintesa", "luniversttas",
}

for _res in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
    nltk.download(_res, quiet=True)

_stop_words = set(stopwords.words(LANGUAGE)) | DOMAIN_STOP_WORDS
_lemmatizer = WordNetLemmatizer()


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _preprocess(text: str) -> str:
    cleaned = _clean(text)
    tokens = [
        _lemmatizer.lemmatize(t)
        for t in cleaned.split()
        if t not in _stop_words and len(t) >= MIN_TOKEN_LEN
    ]
    return " ".join(tokens)


# ── load persisted models at import time (once) ──
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_tfidf: TfidfVectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vec.joblib"))
_nmf: NMF = joblib.load(os.path.join(MODEL_DIR, "nmf_model.joblib"))
_svm: SVC = joblib.load(os.path.join(MODEL_DIR, "svm_pipeline.joblib"))

print("Hybrid NMF+SVM classifier loaded.")


def svm_classify(text: str, top_n: int = 5) -> list[dict]:
    """
    Run the SVM classifier and return the top-N candidate categories
    with confidence scores. First stage of the pipeline — it narrows
    the category search space for the LLM.

    Returns: [{"category": "Research", "confidence": 0.42}, ...]
    """
    try:
        processed = _preprocess(text[:4000])
        if not processed:
            return []

        X_tfidf = _tfidf.transform([processed])
        if X_tfidf.nnz == 0:
            print("svm_classify: no vocabulary overlap with training data.")
            return []

        X_nmf = _nmf.transform(X_tfidf)              # shape (1, 40) — SVM's actual input
        scores = _svm.decision_function(X_nmf)[0]    # shape (n_classes,)
        classes = _svm.classes_

        # Normalize decision-function scores to [0, 1] via softmax so they read as confidences
        exp = np.exp(scores - scores.max())
        probs = exp / exp.sum()

        top_idx = probs.argsort()[-top_n:][::-1]
        return [
            {"category": str(classes[i]), "confidence": round(float(probs[i]), 4)}
            for i in top_idx
        ]

    except Exception as e:
        print("svm_classify error:", e)
        return []