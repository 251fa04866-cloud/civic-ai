"""
CivicAI – Machine Learning Model Module
Implements TF-IDF text vectorization and Logistic Regression multi-class classification
for automated civic complaint category prediction with confidence scoring.
"""

import os
import re
import math
import numpy as np
import joblib
from pathlib import Path
from config import CATEGORIES, MODEL_PATH, VECTORIZER_PATH

# Standard English stop words list for civic complaints
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself",
    "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off",
    "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's",
    "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're",
    "you've", "your", "yours", "yourself", "yourselves", "please", "sir", "madam", "dear", "complaint"
}

def preprocess_text(text: str) -> str:
    """Clean and normalize raw complaint text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple whitespaces
    tokens = [w for w in text.split() if len(w) > 1 and w not in STOP_WORDS]
    return " ".join(tokens)


class CivicTfidfVectorizer:
    """
    TF-IDF Vectorizer with unigram and bigram extraction, sublinear TF scaling,
    and L2 feature vector normalization.
    """
    def __init__(self, min_df: int = 1, max_features: int = 5000, ngram_range=(1, 2)):
        self.min_df = min_df
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary_ = {}
        self.idf_diag_ = None
        self.feature_names_ = []

    def _extract_ngrams(self, text: str):
        words = text.split()
        ngrams = []
        n_words = len(words)
        if self.ngram_range[0] <= 1:
            ngrams.extend(words)
        if self.ngram_range[1] >= 2 and n_words >= 2:
            for i in range(n_words - 1):
                ngrams.append(f"{words[i]} {words[i+1]}")
        return ngrams

    def fit(self, raw_documents):
        """Fit vocabulary and document frequency from raw text corpus."""
        doc_count = len(raw_documents)
        df_counts = {}

        for doc in raw_documents:
            cleaned = preprocess_text(doc)
            unique_terms = set(self._extract_ngrams(cleaned))
            for term in unique_terms:
                df_counts[term] = df_counts.get(term, 0) + 1

        # Filter by min_df
        filtered_terms = [t for t, count in df_counts.items() if count >= self.min_df]
        # Sort terms by frequency and take top max_features
        filtered_terms.sort(key=lambda t: df_counts[t], reverse=True)
        if self.max_features and len(filtered_terms) > self.max_features:
            filtered_terms = filtered_terms[:self.max_features]

        # Assign indices
        filtered_terms.sort()
        self.vocabulary_ = {term: idx for idx, term in enumerate(filtered_terms)}
        self.feature_names_ = filtered_terms

        # Compute smooth IDF: log((1 + N) / (1 + df)) + 1
        n_features = len(self.vocabulary_)
        idf = np.zeros(n_features, dtype=np.float64)
        for term, idx in self.vocabulary_.items():
            df = df_counts.get(term, 1)
            idf[idx] = math.log((1.0 + doc_count) / (1.0 + df)) + 1.0

        self.idf_diag_ = idf
        return self

    def transform(self, raw_documents):
        """Transform text documents into normalized TF-IDF matrix."""
        n_docs = len(raw_documents)
        n_features = len(self.vocabulary_)
        X = np.zeros((n_docs, n_features), dtype=np.float64)

        for doc_idx, doc in enumerate(raw_documents):
            cleaned = preprocess_text(doc)
            ngrams = self._extract_ngrams(cleaned)
            term_freqs = {}
            for t in ngrams:
                if t in self.vocabulary_:
                    term_freqs[t] = term_freqs.get(t, 0) + 1

            for term, count in term_freqs.items():
                col_idx = self.vocabulary_[term]
                # Sublinear TF: 1 + log(count)
                tf = 1.0 + math.log(count)
                X[doc_idx, col_idx] = tf * self.idf_diag_[col_idx]

            # L2 normalize row
            norm = np.linalg.norm(X[doc_idx])
            if norm > 0:
                X[doc_idx] /= norm

        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


class CivicLogisticRegression:
    """
    Multi-Class Logistic Regression (Softmax Regression) with Cross-Entropy Loss,
    L2 Regularization, and Softmax Probability Calibration.
    """
    def __init__(self, lr: float = 0.5, l2_reg: float = 0.001, epochs: int = 150):
        self.lr = lr
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.classes_ = np.array(CATEGORIES)
        self.W = None  # Shape (n_features, n_classes)
        self.b = None  # Shape (n_classes,)

    def _softmax(self, z):
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: list):
        """Train weights using Mini-batch Gradient Descent with momentum."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        # One-hot encode targets
        Y = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, label in enumerate(y):
            idx = np.where(self.classes_ == label)[0]
            if len(idx) > 0:
                Y[i, idx[0]] = 1.0
            else:
                Y[i, 0] = 1.0

        # Initialize weights with Xavier/Glorot scaling
        np.random.seed(42)
        limit = math.sqrt(6.0 / (n_features + n_classes))
        self.W = np.random.uniform(-limit, limit, (n_features, n_classes))
        self.b = np.zeros(n_classes, dtype=np.float64)

        # Batch Gradient Descent with Nesterov Momentum
        v_W = np.zeros_like(self.W)
        v_b = np.zeros_like(self.b)
        momentum = 0.9

        for epoch in range(self.epochs):
            # Forward pass
            logits = np.dot(X, self.W) + self.b
            probs = self._softmax(logits)

            # Gradient computation
            error = probs - Y
            grad_W = (np.dot(X.T, error) / n_samples) + (self.l2_reg * self.W)
            grad_b = np.sum(error, axis=0) / n_samples

            # Momentum updates
            v_W = momentum * v_W - self.lr * grad_W
            v_b = momentum * v_b - self.lr * grad_b

            self.W += v_W
            self.b += v_b

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability distribution over 5 civic categories."""
        if self.W is None:
            raise RuntimeError("Model has not been trained yet.")
        logits = np.dot(X, self.W) + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels with highest probability."""
        probs = self.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        return self.classes_[pred_indices]


class CivicClassifierPipeline:
    """
    Combined Classifier Pipeline integrating Vectorizer and Logistic Regression.
    """
    def __init__(self):
        self.vectorizer = CivicTfidfVectorizer()
        self.model = CivicLogisticRegression()
        self.is_trained = False

    def train(self, texts: list, labels: list):
        """Train pipeline on complaint text corpus."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True
        return self

    def predict_one(self, title: str, description: str) -> dict:
        """
        Classifies a single civic complaint.
        Returns:
            - predicted_category: str
            - confidence_score: float (0.0 to 1.0)
            - confidence_pct: float (0 to 100)
            - class_probabilities: dict[category: str, prob: float]
        """
        combined_text = f"{title} {description}"
        X = self.vectorizer.transform([combined_text])
        probs = self.model.predict_proba(X)[0]

        best_idx = int(np.argmax(probs))
        predicted_category = self.model.classes_[best_idx]
        confidence = float(probs[best_idx])

        prob_dict = {
            self.model.classes_[i]: round(float(probs[i]), 4)
            for i in range(len(self.model.classes_))
        }

        # Sort probability dict in descending order
        prob_dict = dict(sorted(prob_dict.items(), key=lambda item: item[1], reverse=True))

        return {
            "category": predicted_category,
            "confidence": confidence,
            "confidence_pct": round(confidence * 100.0, 1),
            "probabilities": prob_dict
        }

    def save(self, model_file=MODEL_PATH, vec_file=VECTORIZER_PATH):
        """Persist trained pipeline to disk using joblib."""
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_file)
        joblib.dump(self.vectorizer, vec_file)

    def load(self, model_file=MODEL_PATH, vec_file=VECTORIZER_PATH) -> bool:
        """Load trained pipeline from disk."""
        if Path(model_file).exists() and Path(vec_file).exists():
            self.model = joblib.load(model_file)
            self.vectorizer = joblib.load(vec_file)
            self.is_trained = True
            return True
        return False


# Singleton Instance
_PIPELINE_INSTANCE = None

def get_classifier_pipeline() -> CivicClassifierPipeline:
    """Retrieve or initialize the active classifier pipeline."""
    global _PIPELINE_INSTANCE
    if _PIPELINE_INSTANCE is None:
        _PIPELINE_INSTANCE = CivicClassifierPipeline()
        if not _PIPELINE_INSTANCE.load():
            # If not yet trained and saved, trigger training
            from train_model import train_and_save_model
            _PIPELINE_INSTANCE = train_and_save_model()
    return _PIPELINE_INSTANCE
