markdown
# RegWatch — Analyse automatique de conformité Solvabilité II

Outil RegTech basé sur une architecture RAG (Retrieval-Augmented Generation)
pour analyser automatiquement la conformité des rapports SFCR des entreprises
d'assurance vis-à-vis de Solvabilité II.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     PIPELINE REGWATCH                       │
├──────────────┬──────────────┬──────────────┬────────────────┤
│   Phase 1    │   Phase 2    │   Phase 3    │   Phase 4      │
│  Référentiel │  Ingestion   │  Embeddings  │  RAG Core      │
│  SII (71 ex) │  PDF→Chunks  │  ChromaDB    │  LLM scoring   │
├──────────────┴──────────────┴──────────────┴────────────────┤
│                     Phase 5                                 │
│              Moteur de scoring A/B/C/D                      │
├─────────────────────────────────────────────────────────────┤
│                     Phase 6                                 │
│              Interface Streamlit                            │
└─────────────────────────────────────────────────────────────┘
```

## Stack technologique

| Couche | Outil | Justification |
|---|---|---|
| Parsing PDF | PyMuPDF (fitz) | Le plus rapide, conserve la structure typographique |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respecte les frontières naturelles (§ > phrase > mot) |
| Embeddings | `intfloat/multilingual-e5-large` | Top-3 MTEB, natif FR/EN, entraîné pour le retrieval |
| Base vectorielle | ChromaDB | Persistant, filtrage par métadonnées, offline |
| LLM analyse |  Mistral-7B 
| Interface | Streamlit | 

---

## Installation

```bash
# 1. Cloner et créer l'environnement virtuel
git clone 
cd regwatch
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier l'installation
python tests/test_referentiel.py
python tests/test_ingestion.py
python tests/test_phase3.py
python tests/test_phase4.py
```

# ou notebook

```bash
cd notebooks
jupyter notebook demonstration_pipeline.ipynb
```

---

## Structure du projet

```
regwatch/
├── data/
│   ├── raw/                    ← PDFs SFCR à analyser
│   ├── processed/
│   │   └── exigences/          ← 11 fichiers JSON (T01-T11)
│   └── vectorstore/            ← Base ChromaDB persistante
│
├── src/
│   ├── ingestion/
│   │   ├── models.py           ← Dataclasses (ExigenceAtomique, ChunkDocument...)
│   │   ├── referentiel_solvabilite2.py  ← 3 piliers, 11 thèmes, 3 sources
│   │   ├── referentiel_loader.py        ← Chargeur et validateur
│   │   ├── pdf_parser.py       ← Parseur PyMuPDF avec détection sections A-E
│   │   ├── sfcr_chunker.py     ← Chunking différencié SFCR vs réglementaire
│   │   ├── text_utils.py       ← Nettoyage, tokens, préfixes e5
│   │   └── ingestion_pipeline.py  ← Orchestrateur PDF→chunks
│   │
│   └── rag/
│       ├── embedding_engine.py ← MultilingualE5 / LocalTfidf (interface abstraite)
│       ├── vector_store.py     ← Deux collections ChromaDB
│       ├── llm_engine.py       ← Anthropic / Ollama / RuleBased (interface abstraite)
│       └── rag_pipeline.py     ← Retrieval + scoring 40/60
│
├── scripts/
│   └── build_vectorstore.py   ← CLI d'initialisation ChromaDB
│
├── notebooks/
│   └── demonstration_pipeline.ipynb  ← Démonstration interactive
│
├── tests/
│   ├── test_referentiel.py    ← Phase 1 (9 tests)
│   ├── test_ingestion.py      ← Phase 2 (13 tests)
│   ├── test_phase3.py         ← Phase 3 (12 tests)
│   └── test_phase4.py         ← Phase 4 (17 tests)
│
└── README.md
```

---

## Tests

```bash
# Tous les tests (51 tests, ~30 secondes)
python -m pytest tests/ -v

# Par phase
python tests/test_referentiel.py   # Phase 1 : 9 tests
python tests/test_ingestion.py     # Phase 2 : 13 tests
python tests/test_phase3.py        # Phase 3 : 12 tests
python tests/test_phase4.py        # Phase 4 : 17 tests
```

---

