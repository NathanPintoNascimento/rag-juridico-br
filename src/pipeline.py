import argparse
from loader import load_all_documents, load_demo_documents
from chunker import chunk_documents
from embedder import embed_texts
from vectorstore import build_index


def run_pipeline(use_demo: bool = False) -> None:
    print("Iniciando pipeline de indexação")

    if use_demo:
        raw_docs = load_demo_documents()
    else:
        try:
            raw_docs = load_all_documents()
        except Exception:
            print("Nenhum documento encontrado. Usando demo.")
            raw_docs = load_demo_documents()

    chunks = chunk_documents(raw_docs)

    texts = [c["text"] for c in chunks]

    embeddings = embed_texts(texts)

    build_index(embeddings, chunks)

    print(f"Pipeline concluído. {len(chunks)} chunks indexados.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Usar documentos de demonstração"
    )

    args = parser.parse_args()

    run_pipeline(use_demo=args.demo)
