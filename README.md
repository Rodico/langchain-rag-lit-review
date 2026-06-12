# 📚 LangChain RAG Literature Review System

A production-ready **Retrieval-Augmented Generation (RAG)** system for academic literature review, built with LangChain, ChromaDB, and OpenAI. Ingest PDFs, query across papers, synthesize insights, and generate structured literature reviews.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

##  Features

- **Multi-format ingestion** — PDF, TXT, HTML, and arXiv URL support
- **Semantic search** — Dense vector retrieval via ChromaDB + OpenAI embeddings
- **Hybrid retrieval** — Combines BM25 sparse + dense vector search
- **Literature review generation** — Structured synthesis: themes, gaps, methodology comparison
- **Conversational Q&A** — Chat with your paper collection with memory
- **Metadata filtering** — Filter by author, year, journal, topic
- **Citation tracking** — Automatic citation extraction and formatting
- **Streamlit UI** — Browser-based interface for non-technical users
- **CLI interface** — Scriptable command-line tool for automation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
│              (Streamlit UI / CLI / API)                  │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────┐
│                  RAG Pipeline                            │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐   │
│  │  Ingestion  │  │  Retrieval   │  │  Generation   │   │
│  │  - PDF load │  │  - Dense     │  │  - Synthesis  │   │
│  │  - Chunking │  │  - Sparse    │  │  - Q&A        │   │
│  │  - Metadata │  │  - Rerank    │  │  - Citations  │   │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘   │
└─────────┼───────────────-┼──────────────────┼───────────┘
          │                │                  │
┌─────────▼────────────────▼──────────────────▼───────────┐
│                   Storage Layer                          │
│         ChromaDB (vectors) + JSON (metadata)             │
└──────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/langchain-rag-literature-review.git
cd langchain-rag-literature-review

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Ingest Papers

```bash
# Ingest a folder of PDFs
python -m src.cli ingest --path data/papers/

# Ingest a single PDF
python -m src.cli ingest --path paper.pdf

# Ingest from arXiv URL
python -m src.cli ingest --arxiv 2303.08774
```

### 4. Query

```bash
# Ask a question
python -m src.cli query "What are the main approaches to transformer attention?"

# Generate a literature review
python -m src.cli review --topic "large language models" --output review.md

# Start conversational session
python -m src.cli chat
```

### 5. Launch UI

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
langchain-rag-literature-review/
├── src/
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── document_loader.py    # PDF/TXT/HTML/arXiv loaders
│   │   ├── chunker.py            # Semantic + recursive chunking
│   │   └── metadata_extractor.py # Author, year, title extraction
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py       # ChromaDB wrapper
│   │   ├── hybrid_retriever.py   # BM25 + dense hybrid search
│   │   └── reranker.py           # Cross-encoder reranking
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── qa_chain.py           # Conversational Q&A
│   │   ├── review_generator.py   # Literature review synthesis
│   │   └── prompts.py            # All prompt templates
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Settings management
│       └── logging_utils.py      # Structured logging
├── data/
│   ├── papers/                   # Raw input PDFs
│   ├── processed/                # Chunked JSON documents
│   └── vectorstore/              # ChromaDB persistent storage
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_generation.py
├── notebooks/
│   └── exploration.ipynb         # Interactive exploration
├── docs/
│   └── architecture.md
├── config/
│   └── settings.yaml
├── scripts/
│   └── batch_ingest.sh
├── app.py                        # Streamlit UI
├── requirements.txt
├── .env.example
├── pyproject.toml
└── README.md
```

---

## ⚙️ Configuration

Edit `config/settings.yaml` or use environment variables:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o` | LLM model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `6` | Documents to retrieve |
| `VECTORSTORE_PATH` | `data/vectorstore` | ChromaDB storage path |
| `USE_RERANKER` | `false` | Enable cross-encoder reranking |

---

## 🧪 Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov-report=html
```

---

## 📖 Usage Examples

### Python API

```python
from src.ingestion.document_loader import DocumentLoader
from src.retrieval.vector_store import VectorStoreManager
from src.generation.qa_chain import LiteratureQAChain
from src.generation.review_generator import ReviewGenerator

# Ingest papers
loader = DocumentLoader()
docs = loader.load_directory("data/papers/")

vsm = VectorStoreManager()
vsm.add_documents(docs)

# Ask questions
qa = LiteratureQAChain(vsm.get_retriever())
answer = qa.ask("What datasets are used for evaluation?")
print(answer["answer"])
print(answer["sources"])

# Generate full review
generator = ReviewGenerator(vsm.get_retriever())
review = generator.generate(
    topic="transformer architectures",
    style="thematic",   # or "chronological", "methodological"
    max_papers=20
)
print(review)
```

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
