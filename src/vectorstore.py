import pickle
from pathlib import Path
import numpy as np
from config import VECTORSTORE_DIR, VECTOR_BACKEND, TOP_K

VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
FAISS_META_PATH  = VECTORSTORE_DIR / "metadata.pkl"
CHROMA_PATH      = str(VECTORSTORE_DIR / "chroma")


# =============================================================================
# FAISS
# =============================================================================

def build_faiss_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    faiss.write_index(index, str(FAISS_INDEX_PATH))

    with open(FAISS_META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"FAISS: {index.ntotal} vetores indexados")


def search_faiss(query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    import faiss

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            "Índice FAISS não encontrado. Rode o pipeline primeiro."
        )

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with open(FAISS_META_PATH, "rb") as f:
        chunks = pickle.load(f)

    scores, indices = index.search(
        query_vec.reshape(1, -1).astype(np.float32),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        result = dict(chunks[idx])
        result["score"] = float(score)
        results.append(result)

    return results


# =============================================================================
# CHROMADB
# =============================================================================

def build_chroma_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        client.delete_collection("juridico")
    except Exception:
        pass

    collection = client.create_collection(
        name="juridico",
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "page": str(c["page"]),
            "chunk_id": str(c["chunk_id"]),
        }
        for c in chunks
    ]

    batch_size = 500

    for i in range(0, len(chunks), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    print(f"ChromaDB: {collection.count()} vetores indexados")


def search_chroma(query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection("juridico")

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
    )

    output = []

    for i in range(len(results["ids"][0])):
        output.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": int(results["metadatas"][0][i]["page"]),
            "chunk_id": int(results["metadatas"][0][i]["chunk_id"]),
            "score": 1 - results["distances"][0][i],
        })

    return output


# =============================================================================
# INTERFACE UNIFICADA
# =============================================================================

def build_index(embeddings: np.ndarray, chunks: list[dict]) -> None:
    if VECTOR_BACKEND == "chroma":
        build_chroma_index(embeddings, chunks)
    else:
        build_faiss_index(embeddings, chunks)


def search(query_vec: np.ndarray, top_k: int = TOP_K) -> list[dict]:
    if VECTOR_BACKEND == "chroma":
        return search_chroma(query_vec, top_k)
    else:
        return search_faiss(query_vec, top_k)
