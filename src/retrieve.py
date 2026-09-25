import os
import pickle
import chromadb
from embeddings import TfidfEmbedder, NeuralEmbedder
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDER_PATH = os.path.join(DB_DIR, "embedder.pkl")

client = chromadb.PersistentClient(path=DB_DIR)

with open(EMBEDDER_PATH, "rb") as f:
    embedder = pickle.load(f)

collection = client.get_collection(name="handbook", embedding_function=embedder)

def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """
    Embed the query, find the top_k nearest chunks, return them
    with their source and similarity distance.
    """
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "distance": distance,
        })
    return chunks


if __name__ == "__main__":
    test_results = retrieve("how many vacation days do I get?")
    for r in test_results:
        print(f"[{r['source']} | distance={r['distance']:.3f}]")
        print(r["text"][:150])
        print("---")