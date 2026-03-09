# Local RAG

A fully local Retrieval-Augmented Generation (RAG) CLI. No external APIs — everything runs on your own machine.

**Stack:** PDF parsing → multilingual embeddings → ChromaDB → Ollama LLM → streamed answers with source references.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running with the `pet-analyst` model available

---

## Setup

```bash
git clone <this-repo>
cd local-rag
./setup.sh
```

`setup.sh` creates a virtual environment and installs all dependencies. The embedding model (`intfloat/multilingual-e5-small`) is downloaded automatically on first run.

**Verify Ollama:**
```bash
ollama list   # pet-analyst should appear
ollama serve  # start if not already running
```

---

## Usage

1. Place your PDF files in the `docs/` folder.
2. Start the CLI:

```bash
source .venv/bin/activate
python rag.py
```

3. Type your question and press Enter. Answers stream in real time, followed by source references.

**Special commands:**

| Command | Effect |
|---|---|
| `reindex` | Rebuild the vector index (use after adding new PDFs) |
| `quit` / `exit` | Exit the CLI |

---

## How it works

```
docs/*.pdf
    └─ pypdf extracts text page by page
         └─ split into 800-character chunks (150-char overlap)
              └─ "passage: " + chunk → multilingual-e5-small → embedding
                   └─ stored in ChromaDB (cosine similarity)

Query
    └─ "query: " + query → multilingual-e5-small → embedding
         └─ top-4 nearest chunks retrieved from ChromaDB
              └─ chunks + question → prompt → Ollama (pet-analyst)
                   └─ streamed answer + source file + page number
```

---

## Configuration

All settings are constants at the top of `rag.py`:

| Constant | Default | Description |
|---|---|---|
| `DOCS_DIR` | `./docs` | Folder with PDF files |
| `CHROMA_DIR` | `./.chroma` | Persistent vector database path |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-small` | Local HuggingFace embedding model |
| `LLM_MODEL` | `pet-analyst` | Ollama model name |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between consecutive chunks |
| `TOP_K` | `4` | Number of chunks retrieved per query |

---

## Project structure

```
local-rag/
├── docs/          # Put your PDF files here
├── .chroma/       # Auto-created — vector database (gitignore this)
├── rag.py         # All RAG logic + CLI
├── requirements.txt
└── setup.sh
```

### Recommended `.gitignore`

```
.venv/
.chroma/
__pycache__/
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `sentence-transformers` | Local embeddings via HuggingFace |
| `chromadb` | Persistent local vector database |
| `pypdf` | PDF text extraction |
| `ollama` | Local LLM inference |
| `rich` | Terminal formatting and streaming output |
