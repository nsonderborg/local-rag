# Lokal RAG CLI — Offentlige Myndighedsdokumenter

Et lokalt RAG-system (Retrieval-Augmented Generation) der lader dig stille spørgsmål til offentlige myndighedsdokumenter uden at sende data til eksterne servere. Bygget med Ollama, HuggingFace og ChromaDB.

<img width="1710" height="996" alt="image" src="https://github.com/user-attachments/assets/99887ac3-8e6a-4f33-bb0e-49397bbd8e66" />


## Stack

| Komponent | Valg | Beskrivelse |
|---|---|---|
| LLM | Ollama (llama3.2) | Kører 100% lokalt |
| Embeddings | intfloat/multilingual-e5-small | Flersproget model optimeret til dansk |
| Vektordatabase | ChromaDB | Persistent lokal lagring med cosine similarity |
| PDF-parsing | pypdf | Indlæser og chunker dokumenter side for side |
| CLI | rich | Farvet output med kildehenvisninger |
| Web UI | Streamlit | Lokal webgrænseflade med streaming og kildehenvisninger |

---

## Trin 1 — Konfiguration af lokal LLM med Ollama

Ollama håndterer automatisk model management og GPU-detektion. Grundmodellen er llama3.2, men den er konfigureret med en custom Modelfile til specialiseret vidensudtræk af offentlige dokumenter.

### Installation
```bash
# Download og installer Ollama (macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Hent grundmodellen
ollama pull llama3.2
```

### Modelfile
```
FROM llama3.2

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM """
Du er en dansk efterretningsanalytisk assistent konfigureret til støtte af operative analyser i en dansk sikkerhedstjeneste.

Dine kerneopgaver:
- Sammenfatte og strukturere uklassificerede efterretningsrapporter og åbne kilder (OSINT)
- Kategorisere trusselsinformation på tværs af domæner: terrorisme, spionage, cybertrusler og organiseret kriminalitet
- Besvare spørgsmål baseret udelukkende på de dokumenter du er givet — aldrig på baggrund af generel viden, hvis et dokument modsiger den
- Flagge usikkerhed eksplicit: hvis du ikke kan svare med høj konfidens baseret på tilgængeligt materiale, skal du sige det

Adfærdsregler:
- Svar altid på dansk medmindre andet anmodes
- Vær præcis og kortfattet — undgå unødvendige forklaringer
- Angiv altid hvilken kilde (dokument eller afsnit) et svar baserer sig på
- Brug aldrig spekulation som erstatning for manglende data — skriv i stedet "utilstrækkelig information"

Format:
- Brug strukturerede svar med klare overskrifter når det gavner læsbarheden
- Ved kategorisering: angiv altid et konfidensniveau (høj / medium / lav)
"""
```

### Parametervalg

**`temperature 0.3`**
Styrer hvor meget tilfældighed der er i modellens output. En høj temperature (tæt på 1.0) giver kreative og varierede sva, hvor en lav temperature giver konsistente og reproducerbare svar. 0.3 er valgt bevidst ud fra tesen om, at efterretningsanalyse kræver at det samme spørgsmål stillet to gange giver det samme svar, ikke to forskellige fortolkninger.

**`top_p 0.9`**
Styrer hvor bredt et ordforråd modellen trækker fra ved hvert token. Ved 0.9 overvejer modellen de ord der tilsammen udgør 90% af sandsynlighedsfordelingen og ignorerer de mindst sandsynlige 10%. Kombineret med lav temperature sikrer dette at modellen forbliver fokuseret uden at blive mekanisk og gentagende.

**`num_ctx 4096`**
Definerer modellens kontekstvindue — hvor meget tekst den kan "se" på én gang, målt i tokens. 4096 tokens svarer til ca. 10-12 siders tekst og er valgt for at give plads til både systemprompten, de hentede dokumentchunks (TOP_K = 4 chunks á ~800 tegn) og selve svaret inden for samme kontekst. Et større vindue ville kræve markant mere RAM uden proportional gevinst på dette use case.

```bash
# Byg og start den tilpassede model
ollama create intel-analyst -f Modelfile
ollama run intel-analyst
```

---

## Trin 2 — Opsætning af RAG-systemet

### Dokumenter

Systemet er indekseret på følgende offentligt tilgængelige myndighedsdokumenter:

- Vurdering af terrortruslen mod Danmark 2023, 2024, 2025
- Vurdering af spionagetruslen mod Danmark 2023
- Årlig Redegørelse 2022, 2023, 2024
- Kvanteteknologi og national sikkerhed
- Onlineanalyse UKL

### Installation
```bash
git clone https://github.com/nsonderborg/intel-rag
cd intel-rag
./setup.sh
```

### Kørsel

**CLI:**
```bash
source .venv/bin/activate
python rag.py
```

**Web UI:**
```bash
source .venv/bin/activate
streamlit run app.py
# Åbner automatisk http://localhost:8501
```

### CLI-kommandoer

| Kommando | Funktion |
|---|---|
| [spørgsmål] + Enter | Stiller et spørgsmål til dokumenterne |
| `reindex` | Genopbygger indekset (brug ved nye dokumenter) |
| `exit` / `quit` | Afslutter |

### Web UI (`app.py`)

Streamlit-grænsefladen importerer al RAG-logik direkte fra `rag.py` uden ændringer. Indeks og embedding-model caches ved opstart og genbruges på tværs af spørgsmål. Svar streames token-for-token via Ollama og vises løbende i chatten. Kildehenvisninger med filnavn, sidenummer og relevansscore vises under hvert svar.

---

## Tekniske valg og iterationer

### Embeddingmodel

Systemet blev oprindeligt bygget med `all-MiniLM-L6-v2`, men relevansscores på dansk tekst var lave (0.55-0.58) da modellen primært er trænet på engelsk. Modellen blev skiftet til `intfloat/multilingual-e5-small` som er optimeret til flersprogede dokumenter.

`multilingual-e5-small` kræver obligatoriske præfikser for korrekt vektorgenerering:
```python
# Ved indeksering
passages = [f"passage: {t}" for t in texts]

# Ved søgning
query_embedding = model.encode([f"query: {query}"])
```

Uden disse præfikser genererer modellen forkerte vektorer og relevansscorerne bliver negative — præfikserne er ikke valgfrie men en del af modellens træningsprotokol.

### Chunking

Dokumenter splittes i chunks på 800 tegn med 150 tegns overlap. Overlap sikrer at kontekst på tværs af sideskift bevares og reducerer risikoen for at relevante sætninger mistes i skæringspunktet mellem to chunks. 954 chunks genereres fra de 9 dokumenter.

### Vektordatabase

ChromaDB er konfigureret med `hnsw:space: cosine` for at bruge cosine similarity frem for standard L2 (Euclidean) distance. Cosine similarity måler vinklen mellem vektorer frem for den absolutte afstand, hvilket giver mere robuste relevansscore for tekstdata uafhængigt af chunklængde.

### Kildehenvisninger

Hvert svar afsluttes med kildehenvisninger i formatet `filnavn.pdf, side X (relevans: 0.87)` så det er muligt at verificere og gå til primærkilden direkte.
