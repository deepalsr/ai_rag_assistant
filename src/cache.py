from embeddings import TfidfEmbedder
import numpy as np


class SemanticCache:
    def __init__(self, embedder: TfidfEmbedder, similarity_threshold: float = 0.85):
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
                return {"answer": best_entry["answer"], "matched_question": best_entry["question"], "score": best_score}
        return None

    def store(self, question: str, answer: str):
        """Save a new question-answer pair in the cache."""
        vector = self.embedder.embed_query(question)[0]
        self.entries.append({"question": question, "vector": vector, "answer": answer})

if __name__ == "__main__":
    from embeddings import TfidfEmbedder
    import pickle

    with open("../chroma_db/embedder.pkl", "rb") as f:
        embedder = pickle.load(f)

    c = SemanticCache(embedder)
    v1 = embedder.embed_query("how many vacation days do I get?")[0]
    v2 = embedder.embed_query("how much PTO do I have per year?")[0]
    print("Similarity score:", c._cosine_similarity(v1, v2))