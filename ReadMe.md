# RegWatch — Analyse automatique de conformité Solvabilité II

Outil **RegTech** basé sur une architecture **RAG** pour analyser automatiquement
la conformité des rapports SFCR vis-à-vis de Solvabilité II.

## Architecture du projet

```
regwatch/
├── app.py                          ← Interface Streamlit
├── requirements.txt
│
├── data/
│   ├── raw/                        ← PDFs SFCR à analyser
│   ├── processed/
│   │   └── exigences/              ← 11 fichiers JSON (T01-T11), 71 exigences
│   ├── vectorstore/                ← Base ChromaDB persistante
│   └── rapports/                   ← Exports JSON/CSV des analyses
│
├── src/
│   ├── ingestion/                  ← Phase 1 & 2
│   │   ├── models.py
│   │   ├── referentiel_solvabilite2.py
│   │   ├── referentiel_loader.py
│   │   ├── pdf_parser.py
│   │   ├── sfcr_chunker.py
│   │   ├── text_utils.py
│   │   └── ingestion_pipeline.py
│   │
│   ├── rag/                        ← Phase 3 & 4
│   │   ├── embedding_engine.py
│   │   ├── embedding_manager.py
│   │   ├── vector_store.py
│   │   ├── llm_engine.py
│   │   └── rag_pipeline.py
│   │
│   ├── scoring/                    ← Phase 5
│   │   ├── scoring_engine.py
│   │   └── rapport_exporter.py
│   │
│   └── evaluation/                 ← Phase 7
│       ├── ground_truth.py
│       └── rag_evaluator.py
│
├── scripts/
│   └── build_vectorstore.py
│
├── notebooks/
│   └── demonstration_pipeline.ipynb
│
├── docs/
│   ├── graphe1_referentiel.png
│   ├── graphe2_rag_resultats.png
│   └── graphe3_scores_themes.png
│
└── tests/
    ├── test_referentiel.py     ←  9 tests  Phase 1
    ├── test_ingestion.py       ← 13 tests  Phase 2
    ├── test_phase3.py          ← 12 tests  Phase 3
    ├── test_phase4.py          ← 17 tests  Phase 4
    ├── test_phase5.py          ← 15 tests  Phase 5
    ├── test_phase6.py          ← 11 tests  Phase 6
    └── test_phase7.py          ← 16 tests  Phase 7  [TOTAL : 93 tests, 0 KO]
```

## Stack technologique

| Couche | Outil | Justification |
|---|---|---|
| Parsing PDF | PyMuPDF | 5-10x plus rapide que pdfplumber ; conserve la typographie |
| Chunking | LangChain RecursiveCharacterTextSplitter | Frontières naturelles ; overlap 80 tokens |
| Embeddings prod | multilingual-e5-large | Top-3 MTEB ; natif FR/EN ; préfixes query:/passage: |
| Embeddings dev | LocalTfidfEmbedder (TF-IDF + LSA) | Offline ; même interface ; swap en 1 ligne |
| Base vectorielle | ChromaDB | Persistant + filtrage métadonnées ; FAISS écarté |
| LLM prod | Claude Sonnet / Mistral-7B (Ollama) | Interface abstraite swappable |
| LLM dev | RuleBasedLLM | Déterministe, offline |
| Scoring | Agrégation pondérée 5 niveaux | SHALL=1.0 / SHOULD=0.6 / MAY=0.3 |
| Interface | Streamlit | 4 onglets + export JSON/CSV natif |
| Évaluation | NDCG@K, MRR, P@K, R@K | Standards IR (TREC/BEIR) |

## Installation

```bash
git clone <url-du-repo>
cd regwatch
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

```bash
# Interface Streamlit
streamlit run app.py



