"""
embedding_engine.py — Moteur d'embedding découplé
==================================================
Architecture à double couche :

  EmbeddingEngine (interface abstraite)
      ├── MultilingualE5Embedder     ← Production (HuggingFace / fastembed)
      └── LocalTfidfEmbedder         ← Développement offline (ce fichier)

La séparation est intentionnelle : toute la couche ChromaDB, le pipeline RAG
et le moteur de scoring sont écrits contre l'interface EmbeddingEngine.
Passer en production = changer UNE ligne dans la config.

Pourquoi multilingual-e5-large en production ?
  - Modèle multilingue FR/EN → indispensable (SFCR FR + EIOPA EN)
  - Dimension 1024 → meilleure séparation sémantique
  - Entraîné spécifiquement pour le retrieval (paires query/passage)
  - MTEB Leaderboard : top-3 open-source multilingue (mai 2024)
  - Préfixes query:/passage: → distinction indexation vs recherche

Pourquoi TF-IDF LSA local en développement ?
  - Zéro dépendance réseau
  - Rapide à initialiser et à utiliser
  - Comportement vectoriel identique à e5 (normalisation cosine, même API)
  - Suffisant pour valider tout le pipeline RAG end-to-end
  - Facilement remplaçable : même interface, même format de sortie
"""

from __future__ import annotations
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface abstraite — contrat que tout embedder doit respecter
# ---------------------------------------------------------------------------

class EmbeddingEngine(ABC):
    """
    Interface abstraite pour le moteur d'embedding.

    Tout embedder doit implémenter :
      - embed_documents(textes) → np.ndarray shape (N, dim)
      - embed_query(texte)      → np.ndarray shape (dim,)
      - dimension               → int

    Les vecteurs retournés sont TOUJOURS normalisés L2 (norme = 1.0).
    Cela garantit que la similarité cosinus = produit scalaire,
    ce qui est plus rapide à calculer dans ChromaDB.
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension des vecteurs produits."""
        ...

    @property
    @abstractmethod
    def nom_modele(self) -> str:
        """Nom du modèle pour la documentation et les métadonnées."""
        ...

    @abstractmethod
    def embed_documents(self, textes: list[str]) -> np.ndarray:
        """
        Embed une liste de documents (passages à indexer).
        Retourne un array numpy de shape (N, dimension), normalisé L2.
        """
        ...

    @abstractmethod
    def embed_query(self, texte: str) -> np.ndarray:
        """
        Embed un texte de requête (exigence à comparer).
        Retourne un array numpy de shape (dimension,), normalisé L2.
        """
        ...

    def similarite_cosinus(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calcule la similarité cosinus entre deux vecteurs normalisés.
        Comme les vecteurs sont normalisés L2, c'est un simple produit scalaire.
        Retourne une valeur dans [0, 1] pour des vecteurs positifs, [-1, 1] sinon.
        """
        return float(np.dot(vec_a, vec_b))

    @staticmethod
    def _normaliser_l2(vecteurs: np.ndarray) -> np.ndarray:
        """Normalise les vecteurs à une norme L2 de 1.0."""
        normes = np.linalg.norm(vecteurs, axis=-1, keepdims=True)
        normes = np.where(normes == 0, 1.0, normes)  # Éviter division par 0
        return vecteurs / normes


# ---------------------------------------------------------------------------
# Embedder 1 : MultilingualE5Embedder (Production)
# ---------------------------------------------------------------------------

class MultilingualE5Embedder(EmbeddingEngine):
    """
    Embedder de production basé sur intfloat/multilingual-e5-large.

    Caractéristiques :
      - Dimension : 1024
      - Langues : 100+ dont FR et EN
      - Préfixes obligatoires : "query: " pour les requêtes,
                                "passage: " pour les documents
      - Normalisation : L2 intégrée (normalize_embeddings=True)

    Installation :
        pip install sentence-transformers
        # Le modèle (~2.2 Go) sera téléchargé depuis HuggingFace au premier appel

    Usage :
        embedder = MultilingualE5Embedder()
        vecs = embedder.embed_documents(["passage: texte SFCR..."])
        query = embedder.embed_query("query: Le rapport décrit-il la gouvernance ?")

    Note sur les préfixes :
        multilingual-e5-large est entraîné avec des préfixes spécifiques.
        Ils sont DÉJÀ ajoutés par text_utils.preparer_requete_e5().
        Ne pas les ajouter en double ici.
    """

    _MODEL_NAME = "intfloat/multilingual-e5-large"
    _DIMENSION = 1024

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        logger.info(f"MultilingualE5Embedder initialisé (device={device})")

    def _charger_modele(self):
        """Chargement paresseux du modèle (lazy loading)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Chargement de {self._MODEL_NAME}...")
                self._model = SentenceTransformer(
                    self._MODEL_NAME,
                    device=self._device,
                )
                logger.info(f"Modèle chargé ✓ — dim={self._DIMENSION}")
            except Exception as e:
                raise RuntimeError(
                    f"Impossible de charger {self._MODEL_NAME}.\n"
                    f"Vérifier la connexion HuggingFace ou utiliser "
                    f"LocalTfidfEmbedder pour le développement offline.\n"
                    f"Erreur : {e}"
                )

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    @property
    def nom_modele(self) -> str:
        return self._MODEL_NAME

    def embed_documents(self, textes: list[str]) -> np.ndarray:
        """
        Embed des passages SFCR ou réglementaires.
        Les textes doivent déjà avoir le préfixe "passage: " (via preparer_requete_e5).
        """
        self._charger_modele()
        vecs = self._model.encode(
            textes,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=len(textes) > 100,
        )
        return np.array(vecs, dtype=np.float32)

    def embed_query(self, texte: str) -> np.ndarray:
        """
        Embed une requête (texte_verification d'une exigence).
        Le texte doit déjà avoir le préfixe "query: " (via preparer_requete_e5).
        """
        self._charger_modele()
        vec = self._model.encode(
            [texte],
            normalize_embeddings=True,
        )
        return np.array(vec[0], dtype=np.float32)


# ---------------------------------------------------------------------------
# Embedder 2 : LocalTfidfEmbedder (Développement offline)
# ---------------------------------------------------------------------------

class LocalTfidfEmbedder(EmbeddingEngine):
    """
    Embedder local basé sur TF-IDF + LSA (Latent Semantic Analysis).

    Conçu pour le développement offline quand HuggingFace n'est pas accessible.
    Valide toute l'architecture ChromaDB et le pipeline RAG sans réseau.

    Architecture :
      TF-IDF (50 000 features, bigrammes FR/EN)
        → TruncatedSVD (réduction à 512 dimensions)
          → Normalisation L2

    Dimension : 512 (vs 1024 pour e5-large)

    Limitations vs multilingual-e5-large :
      - Pas de compréhension sémantique profonde (synonymes, paraphrases)
      - Moins performant sur les requêtes longues
      - Pas de transfert cross-lingue FR↔EN natif
      - Nécessite un corpus d'entraînement (fit sur le référentiel)

    Avantages pour le développement :
      - Zéro réseau, zéro GPU
      - Déterministe (reproductible)
      - Rapide à initialiser (<1 seconde)
      - Même interface API que MultilingualE5Embedder
    """

    _DIMENSION = 512
    _MODEL_NAME = "local-tfidf-lsa-512d (développement offline)"

    def __init__(self, dimension: int = 512):
        self._dim = min(dimension, 512)
        self._vectorizer = None
        self._svd = None
        self._fitted = False
        logger.info(
            f"LocalTfidfEmbedder initialisé — dim={self._dim} "
            f"(DÉVELOPPEMENT OFFLINE — remplacer par MultilingualE5Embedder en prod)"
        )

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def nom_modele(self) -> str:
        return self._MODEL_NAME

    def fit(self, corpus: list[str]) -> "LocalTfidfEmbedder":
        """
        Entraîne le vectoriseur sur un corpus de référence.
        Doit être appelé avant embed_documents() et embed_query().

        En pratique : on fit sur la concaténation du référentiel réglementaire
        et d'un SFCR de référence pour couvrir le vocabulaire des deux domaines.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        logger.info(f"Fit TF-IDF sur {len(corpus)} documents...")

        self._vectorizer = TfidfVectorizer(
            max_features=50_000,          # Vocabulaire large
            ngram_range=(1, 2),           # Unigrammes + bigrammes
            analyzer="word",
            min_df=1,                     # Tous les termes (corpus petit)
            sublinear_tf=True,            # log(1+tf) → meilleure pondération
            strip_accents=None,           # Conserver les accents français
            token_pattern=r"(?u)\b\w+\b", # Tokens Unicode (FR + EN)
        )

        tfidf_matrix = self._vectorizer.fit_transform(corpus)
        logger.info(
            f"TF-IDF : {tfidf_matrix.shape[0]} docs × "
            f"{tfidf_matrix.shape[1]} features"
        )

        # LSA : réduction dimensionnelle via SVD tronquée
        n_components = min(self._dim, tfidf_matrix.shape[1] - 1,
                          tfidf_matrix.shape[0] - 1)
        self._svd = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=7,
            random_state=42,
        )
        self._svd.fit(tfidf_matrix)
        variance_expliquee = self._svd.explained_variance_ratio_.sum()
        logger.info(
            f"LSA : {n_components} composantes, "
            f"variance expliquée = {variance_expliquee:.1%}"
        )

        self._fitted = True
        return self

    def _verifier_fitted(self):
        if not self._fitted:
            raise RuntimeError(
                "LocalTfidfEmbedder non entraîné. "
                "Appelez .fit(corpus) avant d'embedder."
            )

    def embed_documents(self, textes: list[str]) -> np.ndarray:
        """Embed une liste de documents. Retourne array (N, dim) normalisé L2."""
        self._verifier_fitted()
        # Nettoyer les préfixes e5 (pas pertinents pour TF-IDF)
        textes_clean = [self._retirer_prefixe(t) for t in textes]
        tfidf = self._vectorizer.transform(textes_clean)
        vecs = self._svd.transform(tfidf).astype(np.float32)
        return self._normaliser_l2(vecs)

    def embed_query(self, texte: str) -> np.ndarray:
        """Embed une requête. Retourne array (dim,) normalisé L2."""
        self._verifier_fitted()
        texte_clean = self._retirer_prefixe(texte)
        tfidf = self._vectorizer.transform([texte_clean])
        vec = self._svd.transform(tfidf).astype(np.float32)
        return self._normaliser_l2(vec)[0]

    @staticmethod
    def _retirer_prefixe(texte: str) -> str:
        """Retire les préfixes query: / passage: non pertinents pour TF-IDF."""
        for prefix in ("query: ", "passage: "):
            if texte.startswith(prefix):
                return texte[len(prefix):]
        return texte


# ---------------------------------------------------------------------------
# Factory — point d'entrée unique pour instancier le bon embedder
# ---------------------------------------------------------------------------

def creer_embedding_engine(
    mode: str = "auto",
    dimension: int = 512,
    corpus_fit: Optional[list[str]] = None,
) -> EmbeddingEngine:
    """
    Factory pour créer le bon moteur d'embedding selon l'environnement.

    Args:
        mode      : "production" → MultilingualE5Embedder
                    "local"      → LocalTfidfEmbedder
                    "auto"       → tente production, fallback local
        dimension : Dimension pour le mode local (ignoré en production)
        corpus_fit: Corpus pour le fit TF-IDF (mode local seulement)

    Returns:
        EmbeddingEngine prêt à l'emploi

    Usage recommandé :
        # Développement (offline) :
        engine = creer_embedding_engine("local", corpus_fit=textes_referentiel)

        # Production :
        engine = creer_embedding_engine("production")
    """
    if mode == "production":
        logger.info("Mode production : MultilingualE5Embedder")
        return MultilingualE5Embedder()

    if mode == "local":
        logger.info("Mode local : LocalTfidfEmbedder")
        embedder = LocalTfidfEmbedder(dimension=dimension)
        if corpus_fit:
            embedder.fit(corpus_fit)
        return embedder

    # Mode "auto" : essayer production, fallback local
    logger.info("Mode auto : tentative MultilingualE5Embedder...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(
            MultilingualE5Embedder._MODEL_NAME,
            device="cpu",
        )
        logger.info("MultilingualE5Embedder disponible ✓")
        return MultilingualE5Embedder()
    except Exception as e:
        logger.warning(
            f"MultilingualE5Embedder indisponible ({e}). "
            f"Fallback : LocalTfidfEmbedder"
        )
        embedder = LocalTfidfEmbedder(dimension=dimension)
        if corpus_fit:
            embedder.fit(corpus_fit)
        return embedder
