"""
CivicAI – Duplicate Complaint Detection Engine
Utilizes TF-IDF text representations and Cosine Similarity to identify redundant or recurring complaints.
"""

import numpy as np
from config import DUPLICATE_SIMILARITY_THRESHOLD
from database import get_all_complaints
from ai_model import CivicTfidfVectorizer, preprocess_text

def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D or 2D feature vectors."""
    dot = np.dot(vec_a, vec_b.T)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))

class DuplicateDetector:
    """
    Scans submitted complaint against existing database complaints to detect duplicates.
    """
    def __init__(self, threshold: float = DUPLICATE_SIMILARITY_THRESHOLD):
        self.threshold = threshold

    def check_for_duplicates(
        self,
        new_title: str,
        new_description: str,
        new_location: str = "",
        exclude_complaint_id: str = None
    ) -> dict:
        """
        Compares new complaint text against all active or recent complaints in the database.
        
        Returns:
            {
                "is_duplicate": bool,
                "highest_similarity": float (0.0 to 1.0),
                "highest_similarity_pct": float (0.0 to 100.0),
                "matched_complaint": dict | None,
                "top_matches": list[dict]
            }
        """
        existing_complaints = get_all_complaints()
        if exclude_complaint_id:
            existing_complaints = [
                c for c in existing_complaints if c["complaint_id"] != exclude_complaint_id
            ]

        if not existing_complaints:
            return {
                "is_duplicate": False,
                "highest_similarity": 0.0,
                "highest_similarity_pct": 0.0,
                "matched_complaint": None,
                "top_matches": []
            }

        new_text = f"{new_title} {new_description}"
        if new_location:
            new_text += f" {new_location}"

        existing_texts = [
            f"{c['title']} {c['description']} {c['location']}" for c in existing_complaints
        ]

        # Vectorize new text alongside existing texts
        corpus = [new_text] + existing_texts
        vec = CivicTfidfVectorizer(ngram_range=(1, 2))
        X = vec.fit_transform(corpus)

        new_vec = X[0]
        existing_vecs = X[1:]

        # Calculate cosine similarity with each existing complaint
        matches = []
        for i, c in enumerate(existing_complaints):
            target_vec = existing_vecs[i]
            sim = compute_cosine_similarity(new_vec, target_vec)
            
            # Boost score slightly if exact same location / ward matches
            if new_location and c.get("location") and new_location.strip().lower() in c.get("location", "").lower():
                sim = min(1.0, sim + 0.05)

            if sim >= 0.30:  # Keep candidates above 30% for inspection
                matches.append({
                    "complaint_id": c["complaint_id"],
                    "title": c["title"],
                    "category": c["category_confirmed"],
                    "status": c["status"],
                    "location": c["location"],
                    "created_at": c["created_at"],
                    "similarity": round(sim, 4),
                    "similarity_pct": round(sim * 100.0, 1)
                })

        matches.sort(key=lambda m: m["similarity"], reverse=True)

        highest_sim = matches[0]["similarity"] if matches else 0.0
        is_dup = highest_sim >= self.threshold
        best_match = matches[0] if (matches and is_dup) else (matches[0] if matches else None)

        return {
            "is_duplicate": is_dup,
            "highest_similarity": highest_sim,
            "highest_similarity_pct": round(highest_sim * 100.0, 1),
            "matched_complaint": best_match,
            "top_matches": matches[:5]
        }
