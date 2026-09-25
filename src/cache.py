"""
SEMANTIC CACHE
---------------
Job: before calling generate() (expensive), check whether we've already
answered a similar-enough question. A hit now requires BOTH high
semantic similarity AND overlap in which document chunks were
retrieved - similarity alone was fooled by same-structure,
different-topic questions (see eval_runner.py results).
"""

from typing import Union
from embeddings import TfidfEmbedder, NeuralEmbedder
import numpy as np


class SemanticCache:
    def __init__(
        self,
        embedder: Union[TfidfEmbedder, NeuralEmbedder],
        similarity_threshold: float = 0.45,
        overlap_threshold: float = 0.5,
    ):
        self.embedder = embedder
        self.threshold = similarity_threshold
        self.overlap_threshold = overlap_threshold
        self.entries = []  # list of {"question", "vector", "answer", "chunk_ids"}

    def _cosine_similarity(self, vec_a, vec_b) -> float:
        a = np.array(vec_a)
        b = np.array(vec_b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def lookup(self, question: str, chunk_ids: set) -> dict | None:
        """
        A cache hit requires BOTH:
        1. High semantic similarity to a past question
        2. Meaningful overlap in which document chunks were retrieved
        """
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

        if best_score < self.threshold:
            return None

        overlap = chunk_ids & best_entry["chunk_ids"]
        overlap_ratio = len(overlap) / max(len(chunk_ids), 1)

        if overlap_ratio < self.overlap_threshold:
            return None

        return {
            "answer": best_entry["answer"],
            "matched_question": best_entry["question"],
            "score": best_score,
            "overlap_ratio": overlap_ratio,
        }

    def store(self, question: str, answer: str, chunk_ids: set):
        vector = self.embedder.embed_query(question)[0]
        self.entries.append({
            "question": question,
            "vector": vector,
            "answer": answer,
            "chunk_ids": chunk_ids,
        })


if __name__ == "__main__":
    # Quick manual calibration check
    import pickle

    with open("../chroma_db/embedder.pkl", "rb") as f:
        loaded_embedder = pickle.load(f)

    c = SemanticCache(loaded_embedder)

    pairs = [
        ("how many PTO days do I get?", "how much vacation do I have?"),
        ("how many PTO days do I get?", "how many days can I carry over?"),
        ("how many PTO days do I get?", "how many days can I work remotely?"),
        ("how many PTO days do I get?", "who is the CFO?"),
    ]

    for q1, q2 in pairs:
        v1 = loaded_embedder.embed_query(q1)[0]
        v2 = loaded_embedder.embed_query(q2)[0]
        score = c._cosine_similarity(v1, v2)
        print(f'{score:.3f}  |  "{q1}" vs "{q2}"')