"""
SEMANTIC CACHE
---------------
Job: before calling retrieve() + generate() (expensive), check whether
we've already answered a similar-enough question. If yes, return the
cached answer instantly and skip the LLM entirely.
"""

from typing import Union
from embeddings import TfidfEmbedder, NeuralEmbedder
import numpy as np


class SemanticCache:
    def __init__(
        self,
        embedder: Union[TfidfEmbedder, NeuralEmbedder],
        similarity_threshold: float = 0.2,
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.entries = []  # list of {"question": str, "vector": [...], "answer": str}

    def _cosine_similarity(self, vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def lookup(self, question: str):
        """Return a cached answer if a similar-enough question exists, else None."""
        if not self.entries:
            return None

        query_vector = self.embedder.embed_query(question)[0]

        best_score = -1
        best_entry = None
        for entry in self.entries:
            score = self._cosine_similarity(query_vector, entry["vector"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            return {
                "answer": best_entry["answer"],
                "matched_question": best_entry["question"],
                "score": best_score,
            }
        return None

    def store(self, question: str, answer: str):
        """Save a new question-answer pair in the cache."""
        vector = self.embedder.embed_query(question)[0]
        self.entries.append({"question": question, "vector": vector, "answer": answer})


if __name__ == "__main__":
    # Quick manual calibration check - run `python cache.py` directly
    # to sanity-test similarity scores without going through main.py
    import pickle

    with open("../chroma_db/embedder.pkl", "rb") as f:
        loaded_embedder = pickle.load(f)

    c = SemanticCache(loaded_embedder)

    pairs = [
        ("how many vacation days do I get?", "how much PTO do I have per year?"),
        ("how many vacation days do I get?", "what is the IT support SLA?"),
        ("how many vacation days do I get?", "how many vacation days do I get?"),
    ]

    for q1, q2 in pairs:
        v1 = loaded_embedder.embed_query(q1)[0]
        v2 = loaded_embedder.embed_query(q2)[0]
        score = c._cosine_similarity(v1, v2)
        print(f'{score:.3f}  |  "{q1}" vs "{q2}"')