"""
Lokal RAG med HuggingFace sentence-transformers, ChromaDB og Ollama.
"""

import os
import sys
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path(__file__).parent / "docs"
CHROMA_DIR = Path(__file__).parent / ".chroma"
COLLECTION_NAME = "rag_docs"
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
LLM_MODEL = "intel-analyst"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 4

console = Console()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


def load_pdfs(docs_dir: Path) -> list[dict]:
    """Extract text from all PDFs in docs_dir, return list of {source, page, text}."""
    documents = []
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[yellow]Ingen PDF-filer fundet i {docs_dir}[/yellow]")
        return documents

    for pdf_path in pdf_files:
        console.print(f"  Indlæser [cyan]{pdf_path.name}[/cyan]...")
        try:
            reader = PdfReader(str(pdf_path))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue
                for chunk in chunk_text(text):
                    documents.append({
                        "source": pdf_path.name,
                        "page": page_num,
                        "text": chunk,
                    })
        except Exception as e:
            console.print(f"[red]Fejl ved indlæsning af {pdf_path.name}: {e}[/red]")

    return documents


def build_index(force_rebuild: bool = False) -> tuple[chromadb.Collection, SentenceTransformer]:
    """Build or load the ChromaDB vector index."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing and not force_rebuild:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            console.print(f"[green]Indlæser eksisterende indeks ({count} chunks).[/green]")
            model = SentenceTransformer(EMBEDDING_MODEL)
            return collection, model

    # Delete existing collection if rebuilding
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)

    console.print("[bold]Bygger indeks fra PDF-filer...[/bold]")
    documents = load_pdfs(DOCS_DIR)

    if not documents:
        console.print("[red]Ingen dokumenter at indeksere. Læg PDF-filer i /docs mappen.[/red]")
        sys.exit(1)

    console.print(f"  Loader embedding-model [cyan]{EMBEDDING_MODEL}[/cyan]...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    console.print(f"  Genererer embeddings for {len(documents)} chunks...")
    texts = [d["text"] for d in documents]
    passages = [f"passage: {t}" for t in texts]
    embeddings = model.encode(passages, show_progress_bar=True, batch_size=32).tolist()

    collection = client.create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"doc_{i}" for i in range(len(documents))],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{"source": d["source"], "page": d["page"]} for d in documents],
    )

    console.print(f"[green]Indeks bygget: {len(documents)} chunks fra {len(set(d['source'] for d in documents))} filer.[/green]")
    return collection, model


def retrieve(query: str, collection: chromadb.Collection, model: SentenceTransformer) -> list[dict]:
    """Find the most relevant chunks for a query."""
    query_embedding = model.encode([f"query: {query}"]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "score": round(1 - dist, 4),
        })
    return chunks


def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Kilde {i}: {chunk['source']}, side {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return (
        f"Du er en hjælpsom assistent. Besvar spørgsmålet udelukkende baseret på den givne kontekst. "
        f"Hvis svaret ikke kan findes i konteksten, sig det klart.\n\n"
        f"Kontekst:\n{context}\n\n"
        f"Spørgsmål: {query}\n\n"
        f"Svar:"
    )


def ask(query: str, collection: chromadb.Collection, model: SentenceTransformer) -> None:
    """Retrieve relevant chunks and generate an answer via Ollama."""
    chunks = retrieve(query, collection, model)

    prompt = build_prompt(query, chunks)

    console.print("\n[bold blue]Svar:[/bold blue]")
    answer_parts = []
    try:
        stream = ollama.chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for part in stream:
            token = part["message"]["content"]
            answer_parts.append(token)
            console.print(token, end="", markup=False)
    except Exception as e:
        console.print(f"\n[red]Fejl ved kald til Ollama: {e}[/red]")
        console.print("[yellow]Kontroller at Ollama kører og at modellen 'intel-analyst' er tilgængelig.[/yellow]")
        return

    console.print("\n")

    # Print sources
    console.print("[bold]Kilder:[/bold]")
    seen = set()
    for chunk in chunks:
        key = (chunk["source"], chunk["page"])
        if key not in seen:
            seen.add(key)
            console.print(f"  - {chunk['source']}, side {chunk['page']} (relevans: {chunk['score']})")


def cli() -> None:
    console.print(Panel.fit(
        "[bold green]Lokal RAG[/bold green] — HuggingFace + ChromaDB + Ollama\n"
        "Skriv [bold]quit[/bold] eller [bold]exit[/bold] for at afslutte\n"
        "Skriv [bold]reindex[/bold] for at genopbygge indekset",
        title="RAG CLI",
    ))

    collection, embed_model = build_index()

    while True:
        try:
            query = Prompt.ask("\n[bold cyan]Spørgsmål[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Afslutter.[/yellow]")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            console.print("[yellow]Afslutter.[/yellow]")
            break
        if query.lower() == "reindex":
            collection, embed_model = build_index(force_rebuild=True)
            continue

        ask(query, collection, embed_model)


if __name__ == "__main__":
    cli()
