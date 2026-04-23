"""
integrate.py
Standalone integration module for NMF+SVM document classification.

Use this file in your web system (Flask, Django, FastAPI, etc.) to classify
OCR-extracted text and generate candidate QA categories.

Required files in the same directory as models/:
  - models/nmf_models/nmf_default_run_003/nmf_model.joblib
  - models/nmf_models/nmf_default_run_003/tfidf_vec.joblib
  - models/svm_models/nmf_svm_run_004/svm_pipeline.joblib

Usage:
  >>> from integrate import DocumentClassifier
  >>> classifier = DocumentClassifier()
  >>> ocr_text = "The annual budget allocation for the college..."
  >>> candidates = classifier.classify(ocr_text, top_n=5)
  >>> print(candidates)
  [
    {"category": "Budget", "confidence": 0.92},
    {"category": "Administration", "confidence": 0.05},
    ...
  ]
"""

import os
import re
import joblib
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path


class DocumentClassifier:
    """
    Standalone NMF+SVM document classifier.

    Encapsulates:
    - Text preprocessing (clean, tokenize, lemmatize)
    - TF-IDF vectorization
    - NMF feature extraction
    - SVM classification

    No external dependencies beyond scikit-learn, gensim, nltk.
    """

    # Preprocessing configuration (MUST match training pipeline)
    LANGUAGE = "english"
    MIN_TOKEN_LEN = 3
    DOMAIN_STOP_WORDS = {
        # institutional / location
        "caraga", "csu", "ccis", "butuan", "ampayon", "city",
        "university", "college", "campus", "republic", "philippines",
        "office", "department",
        # document boilerplate
        "iso", "page", "document", "file", "section", "annex",
        "reference", "number", "date", "revision", "copy", "form",
        # generic filler
        "service", "name", "day", "time", "one", "two", "three",
        "state", "shall", "also", "per", "may", "based", "upon",
        "use", "used", "using", "well", "new", "make",
        # Indonesian / Malay
        "yang", "dan", "untuk", "dalam", "dapat", "dari", "dengan",
        "atau", "pada", "ini", "itu", "adalah", "akan", "oleh",
        "universitas", "mulia", "balikpapan", "icsintesa", "luniversttas",
    }

    def __init__(self, models_dir: str = None):
        """
        Initialize the classifier by loading the three joblib models.

        Parameters
        ----------
        models_dir : str, optional
            Path to the models/ folder. If None, assumes models/ is in the
            current working directory.

        Raises
        ------
        FileNotFoundError
            If any of the required .joblib files are not found.
        """
        # Ensure NLTK data is available
        for resource in ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]:
            nltk.download(resource, quiet=True)

        self._stop_words = set(stopwords.words(self.LANGUAGE)) | self.DOMAIN_STOP_WORDS
        self._lemmatizer = WordNetLemmatizer()

        # Resolve models directory
        if models_dir is None:
            models_dir = os.path.join(os.getcwd(), "models")
        self.models_dir = Path(models_dir)

        # Load the three models
        self._load_models()

    def _load_models(self):
        """Load TF-IDF vectorizer, NMF model, and SVM pipeline from joblib files."""
        tfidf_path = (
            self.models_dir  / "tfidf_vec.joblib"
        )
        nmf_path = (
            self.models_dir  / "nmf_model.joblib"
        )
        svm_path = (
            self.models_dir  / "svm_pipeline.joblib"
        )

        if not tfidf_path.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {tfidf_path}")
        if not nmf_path.exists():
            raise FileNotFoundError(f"NMF model not found: {nmf_path}")
        if not svm_path.exists():
            raise FileNotFoundError(f"SVM pipeline not found: {svm_path}")

        self.tfidf_vec = joblib.load(str(tfidf_path))
        self.nmf_model = joblib.load(str(nmf_path))
        self.svm_pipeline = joblib.load(str(svm_path))

        print(
            f"✅ DocumentClassifier loaded:\n"
            f"   - TF-IDF: {len(self.tfidf_vec.vocabulary_)} terms\n"
            f"   - NMF: {self.nmf_model.n_components} topics\n"
            f"   - SVM: {len(self.svm_pipeline.classes_)} classes"
        )

    def _clean(self, text: str) -> str:
        """
        Clean text: lowercase, remove URLs, emails, HTML, punctuation, digits.
        Mirrors src/02_preprocessing.py clean_text().
        """
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)  # URLs
        text = re.sub(r"\S+@\S+", "", text)  # emails
        text = re.sub(r"<.*?>", "", text)  # HTML tags
        text = re.sub(r"[^a-z\s]", " ", text)  # non-alpha
        text = re.sub(r"\s+", " ", text).strip()  # collapse spaces
        return text

    def _preprocess(self, text: str) -> str:
        """
        Preprocess text: clean → tokenize → lemmatize → remove stopwords.
        Returns space-joined lemmatized tokens.
        Mirrors src/02_preprocessing.py tokenize_and_lemmatize().
        """
        cleaned = self._clean(text)
        tokens = [
            self._lemmatizer.lemmatize(t)
            for t in cleaned.split()
            if t not in self._stop_words and len(t) >= self.MIN_TOKEN_LEN
        ]
        return " ".join(tokens)

    def classify(self, ocr_text: str, top_n: int = 5) -> list[dict]:
        """
        Classify OCR-extracted text and return top-N candidate categories
        with confidence scores.

        This is the main entry point for your web system.

        Parameters
        ----------
        ocr_text : str
            Raw text extracted from a document via OCR.
        top_n : int, default=5
            Number of top candidate categories to return.

        Returns
        -------
        list[dict]
            List of candidates sorted by descending confidence:
            [
              {"category": "Budget", "confidence": 0.92},
              {"category": "Administration", "confidence": 0.05},
              ...
            ]
            Returns empty list if classification fails (e.g., empty input).

        Example
        -------
        >>> classifier = DocumentClassifier()
        >>> candidates = classifier.classify(ocr_text)
        >>> for cand in candidates:
        ...     print(f"{cand['category']}: {cand['confidence']:.2%}")
        Budget: 92.00%
        Administration: 5.00%
        """
        try:
            # Step 1: Preprocess the OCR text
            processed = self._preprocess(ocr_text[:4000])  # Limit to first 4000 chars
            if not processed:
                print(
                    "⚠️  classify: empty text after preprocessing "
                    "(no valid tokens found)"
                )
                return []

            # Step 2: TF-IDF vectorization
            X_tfidf = self.tfidf_vec.transform([processed])
            if X_tfidf.nnz == 0:
                print(
                    "⚠️  classify: no vocabulary overlap with training data "
                    "(text may be too different from training corpus)"
                )
                return []

            # Step 3: NMF feature extraction
            # W has shape (1, n_topics) — the document-topic distribution
            X_nmf = self.nmf_model.transform(X_tfidf)

            # Step 4: SVM classification via decision_function
            # decision_function returns scores for each class
            scores = self.svm_pipeline.decision_function(X_nmf)[0]

            # Convert decision scores to confidence-like values via softmax
            # This makes them more interpretable (sum to 1, in [0, 1] range)
            exp = np.exp(scores - scores.max())  # numerical stability
            confidences = exp / exp.sum()

            # Step 5: Get top-N candidates
            classes = self.svm_pipeline.classes_
            top_indices = confidences.argsort()[-top_n:][::-1]

            candidates = [
                {"category": str(classes[i]), "confidence": round(float(confidences[i]), 4)}
                for i in top_indices
            ]

            return candidates

        except Exception as e:
            print(f"❌ classify error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_model_info(self) -> dict:
        """
        Return metadata about the loaded models.
        Useful for debugging and validation.
        """
        return {
            "nmf_topics": self.nmf_model.n_components,
            "svm_classes": len(self.svm_pipeline.classes_),
            "svm_class_list": list(self.svm_pipeline.classes_),
            "tfidf_vocab_size": len(self.tfidf_vec.vocabulary_),
            "svm_kernel": self.svm_pipeline.named_steps.get("svm", {}).kernel
            if hasattr(self.svm_pipeline, "named_steps")
            else "unknown",
        }


# ── Quick test if run directly ──
if __name__ == "__main__":
    import json

    print("\n" + "=" * 70)
    print("  DocumentClassifier Test")
    print("=" * 70)

    try:
        # Initialize
        classifier = DocumentClassifier()

        # Test document
        test_text = """
        The annual budget allocation for the college includes salary, operations,
        and maintenance expenses. Budget management is critical for institutional
        sustainability and resource allocation. The finance office prepares detailed
        budget reports and ensures compliance with accounting standards.
        """

        print(f"\n[Test] Budget Document")
        print(f"Input: {test_text[:100]}...")

        # Classify
        candidates = classifier.classify(test_text, top_n=5)

        print(f"\nTop 5 candidates:")
        for i, cand in enumerate(candidates, 1):
            print(
                f"  {i}. {cand['category']:.<40} {cand['confidence']:.4f} "
                f"({cand['confidence']*100:.1f}%)"
            )

        # Model info
        print(f"\nModel Info:")
        info = classifier.get_model_info()
        print(f"  NMF topics: {info['nmf_topics']}")
        print(f"  SVM classes: {info['svm_classes']}")
        print(f"  Classes: {', '.join(info['svm_class_list'][:5])}...")

        print("\n✅ Test passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

    print("=" * 70 + "\n")
