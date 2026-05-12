from embedder import embed_query
from vectorstore import search
from config import TOP_K


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    query_vec = embed_query(query)
    results = search(query_vec, top_k)

    for r in results:
        r["citation"] = f"{r['source']} | Página {r['page']}"

    return results


def format_context(results: list[dict]) -> str:
    parts = []

    for i, r in enumerate(results, 1):
        parts.append(
            f"[FONTE {i}: {r['source']} | Página {r['page']}]\n{r['text']}"
        )

    return "\n\n".join(parts)
