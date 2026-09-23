from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class TfidfEmbedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.is_fitted = False

    def fit(self, corpus: list[str]):
        self.vectorizer.fit(corpus)
        self.is_fitted = True

    def _embed(self, input: list[str]) -> list[list[float]]:
        if not self.is_fitted:
            raise RuntimeError("Embedder must be fit before use.")
        vectors = self.vectorizer.transform(input).toarray()
        if vectors.shape[1] < 384:
            pad = np.zeros((vectors.shape[0], 384 - vectors.shape[1]))
            vectors = np.hstack([vectors, pad])
        return vectors.tolist()

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        return self._embed(input)

    def embed_query(self, input) -> list[list[float]]:
        if isinstance(input, str):
            texts = [input]
        else:
            texts = input
        return self._embed(texts)

    def name(self) -> str:
        return "tfidf_custom"