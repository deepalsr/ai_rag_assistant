import os
import pickle
import chromadb
from embeddings import TfidfEmbedder
from embeddings import NeuralEmbedder

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDER_PATH = os.path.join(DB_DIR, "embedder.pkl")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks (measured in characters).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def main():
    # 1. Read + chunk every .txt file in data/
    all_chunks = []
    all_ids = []
    all_metadata = []

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".txt"):
            continue

        with open(os.path.join(DATA_DIR, filename), "r") as f:
            text = f.read()

        file_chunks = chunk_text(text)
        print(f"{filename}: {len(file_chunks)} chunks")

        for i, chunk in enumerate(file_chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}-{i}")
            all_metadata.append({"source": filename, "chunk_index": i})

    print(f"Total chunks: {len(all_chunks)}")

    # 2. Fit the embedder on this corpus, save it to disk
    embedder = NeuralEmbedder()

    os.makedirs(DB_DIR, exist_ok=True)
    with open(EMBEDDER_PATH, "wb") as f:
        pickle.dump(embedder, f)
    print(f"Saved embedder to {EMBEDDER_PATH}")

    # 3. Store everything in the vector DB
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection("handbook")
    except Exception:
        pass

    collection = client.create_collection(name="handbook", embedding_function=embedder)
    collection.add(documents=all_chunks, ids=all_ids, metadatas=all_metadata)

    print(f"Stored {collection.count()} chunks in vector DB at {DB_DIR}")


if __name__ == "__main__":
    main()