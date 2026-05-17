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

# Validation complète (93 tests, ~2 min)
for f in tests/test_*.py; do python $f; done
```

## Utilisation rapide

```bash
# Interface Streamlit
streamlit run app.py

# CLI — initialiser ChromaDB + indexer un SFCR
python scripts/build_vectorstore.py \
    --sfcr data/raw/mon_sfcr.pdf \
    --entreprise "AXA France" --annee 2023
```

## Utilisation programmatique

```python
from src.ingestion.referentiel_loader import ReferentielLoader
from src.ingestion.ingestion_pipeline import IngestionPipeline
from src.rag.embedding_engine import creer_embedding_engine
from src.rag.vector_store import VectorStore
from src.rag.llm_engine import creer_llm_engine
from src.rag.rag_pipeline import RAGPipeline
from src.scoring.scoring_engine import ScoringEngine
from pathlib import Path

# 1. Référentiel
loader = ReferentielLoader()
loader.charger()
exigences = loader.get_toutes_exigences()  # 71 exigences

# 2. Engine (dev=offline, prod=multilingual-e5-large)
engine = creer_embedding_engine(
    "local", dimension=256,
    corpus_fit=[e.texte_normalise for e in exigences]
)

# 3. ChromaDB
store = VectorStore(persist_dir=Path("data/vectorstore"))
store.initialiser(engine)

# 4. Ingérer un SFCR
res_ing = IngestionPipeline().ingerer_sfcr(
    Path("data/raw/sfcr.pdf"), "Mon Entreprise", 2023
)
embs = engine.embed_documents(res_ing.textes_embedding)
store.indexer_sfcr(res_ing.chunks, embs.tolist(), "Mon Entreprise", 2023)

# 5. RAG + Scoring
llm = creer_llm_engine("auto")   # auto : Claude > Ollama > RuleBased
rag = RAGPipeline(engine, store, llm, top_k=5)
res_rag = rag.analyser_sfcr(exigences, res_ing.doc_sfcr.id, "Mon Entreprise", 2023)
rapport = ScoringEngine(loader).calculer(res_rag)
print(f"Score : {rapport.score_global:.3f} → Niveau {rapport.niveau_conformite.value}")
```

## Formule de scoring

```
score_final   = 0.40 × cosinus + 0.60 × LLM
score_thème   = Σ(score × poids_obligation) / Σ(poids_obligation)
score_pilier  = Σ(score_thème × poids_dans_pilier)
score_global  = P1×0.30 + P2×0.35 + P3×0.35

Niveau A ≥ 0.80  |  B ≥ 0.65  |  C ≥ 0.45  |  D < 0.45
```

## Passer en production

```python
# Embedding : LocalTfidf → multilingual-e5-large (2.2 Go, HuggingFace)
engine = creer_embedding_engine("production")

# LLM : RuleBased → Claude Sonnet
llm = creer_llm_engine("anthropic", api_key="sk-ant-...")
```
Tout le reste du pipeline est inchangé grâce aux interfaces abstraites.

## Évaluation du RAG

```python
from src.evaluation.ground_truth import GROUND_TRUTH
from src.evaluation.rag_evaluator import RAGEvaluator, indexer_ground_truth_comme_sfcr

doc_id = indexer_ground_truth_comme_sfcr(store, engine, GROUND_TRUTH)
evaluateur = RAGEvaluator(engine, store)
rapport_eval = evaluateur.evaluer(GROUND_TRUTH, k_values=[1, 3, 5, 10], doc_id=doc_id)
print(rapport_eval.formater())
```

## Questions fréquentes du prof

| Question | Réponse clé |
|---|---|
| Pourquoi ChromaDB et non FAISS ? | FAISS = in-memory, pas de persistance. ChromaDB = persistant + filtrage métadonnées |
| Pourquoi multilingual-e5-large ? | Top-3 MTEB multilingue, gratuit, natif FR/EN, entraîné pour le retrieval |
| Pourquoi préfixes query:/passage: ? | Exigence du modèle e5 — -15-20% sans eux (Thakur et al., BEIR 2021) |
| Pourquoi 40% cosinus + 60% LLM ? | Lewis et al. 2020, Gao et al. 2024 — LLM pilote, cosinus ancre |
| Pourquoi SHALL/SHOULD/MAY ? | ISO RFC 2119 — pondération du risque de non-conformité |
| Comment passer en production ? | 2 lignes : creer_embedding_engine("production") + creer_llm_engine("anthropic") |
