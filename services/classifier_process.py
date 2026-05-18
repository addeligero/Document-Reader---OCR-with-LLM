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
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_TFIDF_PATHS = [
    os.path.join(MODELS_DIR, "nmf_models", "nmf_default_run_003", "tfidf_vec.joblib"),
    os.path.join(MODELS_DIR, "tfidf_vec.joblib"),
]
_NMF_PATHS = [
    os.path.join(MODELS_DIR, "nmf_models", "nmf_default_run_003", "nmf_model.joblib"),
    os.path.join(MODELS_DIR, "nmf_model.joblib"),
]
_SVM_PATHS = [
    os.path.join(MODELS_DIR, "svm_models", "nmf_svm_run_004", "svm_pipeline.joblib"),
    os.path.join(MODELS_DIR, "svm_pipeline.joblib"),
]


def _pick_existing(paths: list[str]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Model file not found. Tried: {', '.join(paths)}")


_TFIDF_PATH = _pick_existing(_TFIDF_PATHS)
_NMF_PATH = _pick_existing(_NMF_PATHS)
_SVM_PATH = _pick_existing(_SVM_PATHS)

_tfidf: TfidfVectorizer = joblib.load(_TFIDF_PATH)
_nmf: NMF = joblib.load(_NMF_PATH)
_svm: SVC = joblib.load(_SVM_PATH)

if hasattr(_nmf, "components_") and np.allclose(_nmf.components_, 0):
    print("Warning: NMF components are all zeros. Check the nmf_model.joblib file.")

print("Hybrid NMF+SVM classifier loaded.")


def _pipeline_has_step(model, step_type) -> bool:
    if not hasattr(model, "named_steps"):
        return False
    return any(isinstance(step, step_type) for step in model.named_steps.values())


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

        if _pipeline_has_step(_svm, TfidfVectorizer):
            scores = _svm.decision_function([processed])[0]
            classes = _svm.classes_
        else:
            X_tfidf = _tfidf.transform([processed])
            if X_tfidf.nnz == 0:
                print("svm_classify: no vocabulary overlap with training data.")
                return []

            if _pipeline_has_step(_svm, NMF):
                X_features = X_tfidf
            else:
                try:
                    X_features = _nmf.transform(X_tfidf)  # shape (1, n_topics)
                except ValueError as exc:
                    print(f"svm_classify: NMF transform failed: {exc}")
                    expected = getattr(_svm, "n_features_in_", None)
                    if expected == X_tfidf.shape[1]:
                        X_features = X_tfidf
                    else:
                        return []

            scores = _svm.decision_function(X_features)[0]
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
        import traceback

        traceback.print_exc()
        return []