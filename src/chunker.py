from typing import Optional
from config import CHUNK_SIZE, CHUNK_OVERLAP


def tokens_to_words(token_count: int) -> int:
    return int(token_count / 1.3)


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[str]:

    chunk_size = chunk_size or CHUNK_SIZE
    overlap    = overlap or CHUNK_OVERLAP

    chunk_words   = tokens_to_words(chunk_size)
    overlap_words = tokens_to_words(overlap)

    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += chunk_words - overlap_words

    return [c for c in chunks if len(c.split()) >= 20]


def chunk_documents(raw_docs: list[dict]) -> list[dict]:
    chunked = []

    for doc in raw_docs:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            chunked.append({
                "text": chunk,
                "source": doc["source"],
                "page": doc["page"],
                "filepath": doc.get("filepath", ""),
                "chunk_id": i,
            })

    print(f"{len(chunked)} chunks gerados")
    return chunked
