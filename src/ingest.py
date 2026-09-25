import os
import pickle
import chromadb
from embeddings import TfidfEmbedder
from embeddings import NeuralEmbedder
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
EMBEDDER_PATH = os.path.join(DB_DIR, "embedder.pkl")

def chunk_text(text: str, max_chunk_size: int = 300, overlap_sentences: int = 1) -> list[str]:
    """
    Split text into chunks along SENTENCE boundaries (not raw character
    cuts), keeping chunks small enough that unrelated facts don't end up
    sharing a chunk. overlap_sentences carries the last N sentences of
    one chunk into the start of the next, so context isn't lost at the seam.
    """
    # Split on sentence-ending punctuation followed by whitespace.
    # Simple heuristic - not perfect (won't handle "Dr. Smith" correctly),
    # but good enough for our formal policy-text handbook.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)

        if current_length >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            # Keep the last N sentences as overlap into the next chunk
            current_chunk = current_chunk[-overlap_sentences:] if overlap_sentences else []
            current_length = sum(len(s) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

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