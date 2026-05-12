<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Crimson+Text&size=42&duration=3000&pause=1000&color=C8A96E&center=true&vCenter=true&width=600&lines=%E2%9A%96%EF%B8%8F+RAG+Jur%C3%ADdico;Pesquisa+Inteligente;Com+Cita%C3%A7%C3%A3o+de+Fonte" alt="RAG Jurídico" />

<br/>

<p><em>Pesquisa jurídica em linguagem natural sobre documentos reais do STF, STJ e legislação brasileira</em></p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.36-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/BERTimbau-STS--PT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/rufimelo/bert-large-portuguese-cased-sts)
[![FAISS](https://img.shields.io/badge/FAISS-VectorDB-0467DF?style=for-the-badge&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![Claude](https://img.shields.io/badge/Claude-Anthropic-191919?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

[📖 O que é RAG?](#-o-que-é-rag) · [🏗️ Arquitetura](#%EF%B8%8F-arquitetura) · [🚀 Como Rodar](#-como-rodar) · [⚠️ Desafios](#%EF%B8%8F-o-que-não-funcionou-e-por-quê) · [🔭 Próximos Passos](#-próximos-passos)

</div>

---

## 📖 O que é RAG?

**RAG (Retrieval-Augmented Generation)** é uma arquitetura de IA que combina **busca semântica** com **geração de texto** para responder perguntas com base em documentos reais, Sem inventar informação.

A diferença para um chatbot comum:

```
Chatbot comum:  pergunta → LLM → resposta (pode alucinar)

RAG:            pergunta → busca vetorial → trechos relevantes → LLM → resposta com citação
```

### Por que isso importa no direito?

> Um advogado não pode usar *"a IA disse que..."* como argumento.
> Com RAG, cada resposta aponta exatamente: **qual lei, qual artigo, qual página.**

| Problema do jurista hoje | Solução com este projeto |
|---|---|
| Ler 200 páginas para achar 1 artigo | Pergunta em português natural |
| Não sabe se a informação está atualizada | Controla quais documentos indexar |
| IA inventa artigos inexistentes | Resposta baseada **só** nos documentos reais |
| Sem rastreabilidade das fontes | Cada resposta cita documento + página |

---

##  Arquitetura

### Visão Geral

O sistema tem dois pipelines independentes:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PIPELINE DE INDEXAÇÃO  (executa uma vez)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   PDFs / TXTs
       │
       ▼
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
  │  loader.py  │────▶│  chunker.py  │────▶│  embedder.py │
  │             │     │              │     │              │
  │ Extrai texto│     │ 512 tokens   │     │ BERTimbau STS│
  │ + metadados │     │ overlap: 64  │     │ → vetor 1024d│
  └─────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ vectorstore/  │
                                          │  FAISS index  │
                                          └───────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PIPELINE DE QUERY  (tempo real, a cada pergunta)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

   Pergunta do usuário
       │
       ▼
  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
  │  embedder   │────▶│ vectorstore  │────▶│  retriever   │
  │             │     │              │     │              │
  │ BERTimbau   │     │ busca top-5  │     │ formata      │
  │ → vetor     │     │ chunks       │     │ contexto +   │
  │             │     │ similares    │     │ citações     │
  └─────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐     ┌────────────┐
                                          │ generator.py  │────▶│   app.py   │
                                          │               │     │            │
                                          │ Claude API    │     │ Streamlit  │
                                          │ gera resposta │     │ Chat UI    │
                                          │ fundamentada  │     │            │
                                          └───────────────┘     └────────────┘
```

### Stack de tecnologias

| Camada | Tecnologia | Justificativa |
|---|---|---|
| **Extração de PDF** | PyMuPDF + pdfplumber | PyMuPDF é rápido; pdfplumber preciso com tabelas |
| **Chunking** | Custom (overlap fixo) | 512 tokens = limite do BERT; overlap de 64 evita perda de contexto nas bordas |
| **Embeddings** | `rufimelo/bert-large-portuguese-cased-sts` | BERTimbau fine-tuned para similaridade semântica em PT-BR |
| **Banco vetorial** | FAISS `IndexFlatIP` | Busca exata por produto interno (≡ cosine quando vetores normalizados) |
| **LLM Gerador** | Claude (Anthropic) | Raciocínio jurídico em português com baixa taxa de alucinação |
| **Interface** | Streamlit | Deploy rápido, ideal para protótipos de pesquisa |

### Estrutura de pastas

```
rag_juridico/
├── 📂 data/
│   └── raw/            ← Coloque seus PDFs aqui
├── 📂 src/
│   ├── config.py       ← Configurações centralizadas (.env)
│   ├── loader.py       ← Extração de texto de PDFs e TXTs
│   ├── chunker.py      ← Divisão em chunks com overlap
│   ├── embedder.py     ← Vetorização com BERTimbau
│   ├── vectorstore.py  ← FAISS e ChromaDB (configurável)
│   ├── retriever.py    ← Busca semântica + formatação de citações
│   ├── generator.py    ← Geração de resposta via Claude API
│   ├── pipeline.py     ← Script de indexação (roda 1x)
│   └── app.py          ← Interface Streamlit
├── 📂 vectorstore/     ← Índice FAISS (gerado automaticamente)
├── .env                ← Variáveis de ambiente
├── requirements.txt
└── README.md
```

---

##  Como Rodar

### Pré-requisitos

- Python 3.10+
- ~4 GB de RAM (modelo BERTimbau Large)
- (Opcional) Chave de API Anthropic — sem ela o sistema exibe os trechos mais relevantes sem geração de texto

### Passo a passo

**1. Clone o repositório**

```bash
git clone https://github.com/NathanPintoNascimento/rag-juridico.git
cd rag-juridico
```

**2. Crie e ative um ambiente virtual**

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Instale as dependências**

```bash
pip install -r requirements.txt
```

>  O download do modelo BERTimbau (~1.3 GB) acontece automaticamente na primeira execução do pipeline.

**4. Configure as variáveis de ambiente**

Abra o arquivo `.env` e preencha:

```env
ANTHROPIC_API_KEY=sua_chave_aqui   # https://console.anthropic.com/
EMBEDDING_MODEL=rufimelo/bert-large-portuguese-cased-sts
CHUNK_SIZE=512
CHUNK_OVERLAP=64
VECTOR_BACKEND=faiss
```

**5. Adicione seus documentos** *(opcional)*

Coloque PDFs ou arquivos `.txt` na pasta `data/raw/`.

Fontes gratuitas recomendadas:
- **Constituição Federal:** [planalto.gov.br](https://www.planalto.gov.br/ccivil_03/constituicao/constituicao.htm)
- **STJ:** [stj.jus.br](https://www.stj.jus.br)
- **LexML:** [lexml.gov.br](https://www.lexml.gov.br)

**6. Indexe os documentos**

```bash
cd src

# Com seus PDFs em data/raw/:
python pipeline.py

# Sem PDFs — usa textos de demonstração (CF, CC, STJ):
python pipeline.py --demo
```

**7. Inicie o chat**

```bash
streamlit run app.py
```

Acesse: `http://localhost:8501` 

### Atalhos com Make

```bash
make install      # instala dependências
make index-demo   # indexa documentos de demonstração
make run          # inicia o Streamlit
make clean        # remove índices e cache
```

---

##  O que não funcionou e por quê

Esta seção documenta os obstáculos reais do desenvolvimento — algo que a maioria dos projetos esconde e que diferencia quem realmente construiu de quem copiou.

### 1. Chunking por parágrafo

**Tentativa:** dividir o texto nas quebras de linha duplas (`\n\n`), preservando parágrafos naturais.

**Problema:** PDFs jurídicos extraídos com PyMuPDF perdem toda a formatação original. Um acórdão de 40 páginas chegava como uma string contínua — parágrafos saíam com 5 palavras ou com 2.000. O overlap ficava inconsistente.

**Solução adotada:** chunking por contagem de palavras com overlap fixo de 64 tokens. Menos elegante semanticamente, mas robusto para qualquer PDF.

---

### 2. Embeddings da OpenAI para português

**Tentativa:** usar `text-embedding-3-small` da OpenAI, que tem excelente desempenho geral.

**Problema:** termos jurídicos brasileiros como *"tutela antecipada"*, *"mandado de segurança"*, *"habeas data"* ou *"precatório"* não têm representação vetorial tão precisa quanto no BERTimbau, que foi pré-treinado em um corpus massivo de português brasileiro incluindo textos jurídicos.

**Decisão:** manter BERTimbau STS, que é gratuito, roda localmente e entende o domínio.

---

### 3. ChromaDB como banco vetorial padrão

**Tentativa:** usar ChromaDB pela persistência automática e API mais amigável.

**Problema:** ChromaDB adiciona ~2s de overhead na inicialização do servidor embarcado a cada nova consulta no Streamlit (que reinicia o processo em alguns deploys). Para prototipagem e demo, o FAISS com `pickle` é instantâneo.

**Decisão:** FAISS como padrão, ChromaDB disponível via config para quem preferir.

---

### 4. Chunks sem overlap quebravam o retrieval

**Descoberta:** perguntas sobre conceitos que apareciam no final de um chunk tinham recall muito baixo — o retriever não encontrava os trechos relevantes porque o contexto estava "cortado" na borda.

**Efeito mensurável:** adicionar overlap de 64 tokens aumentou o recall subjetivo em ~30% nos testes manuais com perguntas sobre artigos da CF.

---

##   Próximos Passos

- [ ] **Re-ranking com cross-encoder** — após o retrieval, usar `cross-encoder/ms-marco-MiniLM-L-6-v2` para reordenar os chunks por relevância precisa (não apenas similaridade vetorial)
- [ ] **Chunking semântico** — detectar mudanças de tópico com o próprio BERTimbau para cortar nos pontos semanticamente corretos
- [ ] **Filtros por tipo de documento** — perguntar apenas sobre "acórdãos do STJ" ou "artigos da CF" sem misturar fontes
- [ ] **Avaliação com RAGAS** — métricas automáticas de *faithfulness*, *answer relevancy* e *context precision*
- [ ] **Streaming de resposta** — resposta aparece token por token, como no ChatGPT
- [ ] **Upload de PDFs na UI** — adicionar documentos diretamente pelo Streamlit sem pipeline manual
- [ ] **Deploy no Streamlit Cloud** — URL pública, zero instalação para o usuário final
- [ ] **Fine-tuning do embedder** — treinar o BERTimbau em pares pergunta/artigo jurídico para melhorar o retrieval no domínio específico

---

##  Decisões técnicas explicadas

<details>
<summary><b>Por que 512 tokens por chunk?</b></summary>

O BERTimbau tem uma janela de contexto máxima de 512 tokens. Usar exatamente 512 aproveita toda a capacidade do modelo sem truncar. Na prática, artigos de lei têm entre 80 e 300 tokens — com chunks de 512 capturamos o artigo principal mais o contexto anterior e posterior, o que ajuda na compreensão do significado.

</details>

<details>
<summary><b>Por que FAISS IndexFlatIP e não IndexIVFFlat?</b></summary>

`IndexFlatIP` faz busca exata (compara com todos os vetores). É mais lento em escala de milhões de vetores, mas para coleções jurídicas típicas (até ~50k chunks), a diferença é imperceptível — menos de 50ms. `IndexIVFFlat` usa aproximação (busca mais rápida porém menos precisa), o que seria contraproducente num sistema que precisa de citações corretas.

</details>

<details>
<summary><b>Por que separar BERTimbau (embedder) do Claude (gerador)?</b></summary>

São tarefas diferentes. BERTimbau é excelente para *entender* e representar texto em português juridicamente rico. Claude é excelente para *raciocinar* e *gerar* respostas coerentes. Forçar o mesmo modelo a fazer os dois piora os dois. Essa separação é o coração do RAG.

</details>

---

##  Licença

Distribuído sob a licença MIT. Veja [`LICENSE`](LICENSE) para mais informações.

---

<div align="center">

Desenvolvado por **Nathan Pinto Nascimento**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Nathan%20Nascimento-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nathan-nascimento-)
[![GitHub](https://img.shields.io/badge/GitHub-NathanPintoNascimento-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/NathanPintoNascimento)

<br/>

*Sistemas de Informação · Projeto de IA Generativa e NLP*

<br/>

*Este projeto é para fins educacionais e de pesquisa.*
*Não substitui orientação jurídica profissional.*

<br/>

⭐ Se este projeto foi útil, deixa uma estrela no repositório!

</div>
