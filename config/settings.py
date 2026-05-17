"""
config/settings.py — Configuration centralisée RegWatch
========================================================
Point unique de configuration pour tout le pipeline.
Modifier ici pour ajuster les seuils, chemins et paramètres
sans toucher au code métier.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Chemins du projet
# ---------------------------------------------------------------------------

ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXIGENCES_DIR = PROCESSED_DIR / "exigences"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
RAPPORTS_DIR  = DATA_DIR / "rapports"
DOCS_DIR      = ROOT_DIR / "docs"

# Créer les dossiers s'ils n'existent pas
for d in [RAW_DIR, EXIGENCES_DIR, VECTORSTORE_DIR, RAPPORTS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

EMBEDDING_MODE      = "auto"        # "auto" | "local" | "production"
EMBEDDING_DIMENSION = 256           # Dimension LocalTfidfEmbedder (dev)
EMBEDDING_MODEL_PROD = "intfloat/multilingual-e5-large"  # dim=1024

# ---------------------------------------------------------------------------
# Chunking SFCR
# ---------------------------------------------------------------------------

CHUNK_SIZE_TOKENS   = 450           # Taille cible d'un chunk SFCR
CHUNK_OVERLAP_TOKENS = 80           # Overlap entre chunks consécutifs
CHUNK_MIN_TOKENS    = 50            # Taille minimale pour être indexé
CHUNK_MAX_TOKENS    = 550           # Guard-rail maximum

# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

RAG_TOP_K           = 5            # Passages candidats par exigence
RAG_SEUIL_DISTANCE  = 0.85         # Distance cosinus max acceptée
RAG_POIDS_COSINUS   = 0.40         # Poids signal vectoriel
RAG_POIDS_LLM       = 0.60         # Poids signal sémantique LLM

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

LLM_MODE            = "auto"        # "auto" | "local" | "anthropic" | "ollama" |"Llama"
LLM_MODEL_ANTHROPIC = "claude-sonnet-4-20250514"
LLM_MODEL_OLLAMA    = "mistral"
LLM_OLLAMA_URL      = "http://ollama:11434"
LLM_MAX_TOKENS      = 1000
LLM_TEMPERATURE     = 0.1           # Basse pour réponses déterministes
LLM_MODEL_LLAMA3    = "groq-llama3.3"

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

SEUIL_COUVERT       = 0.75          # score_final >= seuil → COUVERT
SEUIL_PARTIEL       = 0.45          # score_final >= seuil → PARTIEL
                                    # score_final <  seuil → ABSENT

SEUIL_NIVEAU_A      = 0.80          # score_global >= seuil → A (Conforme)
SEUIL_NIVEAU_B      = 0.65          # score_global >= seuil → B (Satisfaisant)
SEUIL_NIVEAU_C      = 0.45          # score_global >= seuil → C (Insuffisant)
                                    # score_global <  seuil → D (Non conforme)

POIDS_PILIER_1      = 0.30          # Exigences quantitatives
POIDS_PILIER_2      = 0.35          # Gouvernance & contrôle
POIDS_PILIER_3      = 0.35          # Reporting & transparence

POIDS_SHALL         = 1.0           # Obligation absolue
POIDS_SHOULD        = 0.6           # Recommandation forte
POIDS_MAY           = 0.3           # Disposition optionnelle

# ---------------------------------------------------------------------------
# Évaluation RAG
# ---------------------------------------------------------------------------

EVAL_K_VALUES       = [1, 3, 5, 10]
EVAL_SEUIL_SIM_NIVEAU2 = 0.35       # Jaccard min pour pertinence niveau 2
EVAL_SEUIL_SIM_NIVEAU1 = 0.25       # Jaccard min pour pertinence niveau 1
