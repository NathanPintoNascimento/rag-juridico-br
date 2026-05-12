import re
from pathlib import Path
from typing import Optional
import pdfplumber
import fitz
from tqdm import tqdm
from config import DATA_RAW


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_pymupdf(filepath: Path) -> list[dict]:
    docs = []
    with fitz.open(filepath) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if len(text) < 30:
                continue
            docs.append({
                "text": text,
                "source": filepath.name,
                "page": page_num,
                "filepath": str(filepath),
            })
    return docs


def load_pdf_pdfplumber(filepath: Path) -> list[dict]:
    docs = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            if len(text) < 30:
                continue
            docs.append({
                "text": text,
                "source": filepath.name,
                "page": page_num,
                "filepath": str(filepath),
            })
    return docs


def load_txt(filepath: Path) -> list[dict]:
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    text = clean_text(text)
    return [{
        "text": text,
        "source": filepath.name,
        "page": 1,
        "filepath": str(filepath),
    }]


def load_all_documents(
    directory: Optional[Path] = None,
    use_pdfplumber: bool = False
) -> list[dict]:

    directory = directory or DATA_RAW
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {directory}")

    files = list(directory.glob("*.pdf")) + list(directory.glob("*.txt"))

    if not files:
        raise ValueError(
            f"Nenhum PDF ou TXT encontrado em {directory}"
        )

    all_docs = []
    pdf_loader = load_pdf_pdfplumber if use_pdfplumber else load_pdf_pymupdf

    for filepath in tqdm(files, desc="Carregando documentos"):
        try:
            docs = pdf_loader(filepath) if filepath.suffix.lower() == ".pdf" else load_txt(filepath)
            all_docs.extend(docs)
        except Exception as e:
            print(f"Erro em {filepath.name}: {e}")

    return all_docs


DEMO_DOCS = [
    {
        "text": "Art. 5º Todos são iguais perante a lei, sem distinção de qualquer natureza.",
        "source": "Constituição Federal 1988 (demo)",
        "page": 1,
        "filepath": "demo",
    },
    {
        "text": "Art. 186. Aquele que, por ação ou omissão voluntária, negligência ou imprudência, violar direito e causar dano a outrem, comete ato ilícito.",
        "source": "Código Civil 2002 (demo)",
        "page": 1,
        "filepath": "demo",
    },
]


def load_demo_documents() -> list[dict]:
    print("Usando documentos de demonstração")
    return DEMO_DOCS
