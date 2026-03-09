#!/usr/bin/env bash
set -e

echo "=== Lokal RAG Setup ==="

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Opretter virtuelt miljø..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "Installerer afhængigheder..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "=== Setup fuldfort ==="
echo "Laeg dine PDF-filer i mappen: docs/"
echo ""
echo "Kontroller at Ollama korer og at modellen 'pet-analyst' er tilgaengelig:"
echo "  ollama list"
echo ""
echo "Start RAG-systemet:"
echo "  source .venv/bin/activate"
echo "  python rag.py"
