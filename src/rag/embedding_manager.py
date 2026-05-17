"""
embedding_manager.py — Gestionnaire d'embeddings avec fallback automatique
==========================================================================
Responsabilités :
  1. Charger multilingual-e5-large si HuggingFace est accessible
  2. Basculer automatiquement sur TF-IDF+LSA en environnement offline
  3. Exposer une interface unique quelle que soit la stratégie active
  4. Gérer les préfixes "query:" / "passage:" requis par le modèle e5
  5. Sérialiser/désérialiser le modèle TF-IDF pour la persistance

Pourquoi multilingual-e5-large comme modèle principal ?
  - Performances SOTA sur les benchmarks BEIR (Retrieval)
  - Multilingue natif : français (SFCR) + anglais (guidelines EIOPA)
  - Fenêtre de 512 tokens — compatible avec nos chunks de 450 tokens
  - Les préfixes "query:"/"passage:" améliorent le retrieval de ~15-20%
    (Thakur et al., 2021 — BEIR benchmark)
  - Licence Apache 2.0 — libre d'utilisation commerciale

Pourquoi TF-IDF+LSA comme fallback ?
  - 100% local, sans dépendance réseau
  - Sémantiquement pertinent sur du texte réglementaire structuré
    (vocabulaire répétitif → TF-IDF très efficace)
  - Entraînable sur le corpus du référentiel en quelques secondes
  - Scikit-learn est disponible dans tout environnement Python

Note sur la dimension des embeddings :
  - multilingual-e5-large : 1024 dimensions
  - TF-IDF+LSA : adaptée au corpus (min(n_docs-1, 256) dimensions)
  ChromaDB gère les deux — la dimension est fixée à la création de la collection.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface abstraite
# ---------------------------------------------------------------------------

class BaseEmbedder:
    """Interface commune pour tous les modèles d'embedding."""

    @property
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    def nom_modele(self) -> str:
        raise NotImplementedError

    def encoder_passages(self, textes: list[str]) -> list[list[float]]:
        """Encode des passages (chunks SFCR) — préfixe 'passage:' si nécessaire."""
        raise NotImplementedError

    def encoder_requetes(self, textes: list[str]) -> list[list[float]]:
        """Encode des requêtes (textes de vérification) — préfixe 'query:' si nécessaire."""
        raise NotImplementedError

    def encoder(self, textes: list[str]) -> list[list[float]]:
        """Encode sans distinction query/passage."""
        return self.encoder_passages(textes)


# ---------------------------------------------------------------------------
# Embedder 1 : multilingual-e5-large (HuggingFace)
# ---------------------------------------------------------------------------

class E5Embedder(BaseEmbedder):
    """
    Embedder basé sur intfloat/multilingual-e5-large.

    CRITIQUE : Ce modèle requiert des préfixes spécifiques :
      - "query: {texte}"   → pour les textes de vérification (exigences)
      - "passage: {texte}" → pour les chunks SFCR à indexer

    Sans ces préfixes, les performances chutent de ~15-20%.
    Source : Wang et al. (2022) — "Text Embeddings by Weakly-Supervised
    Contrastive Pre-training"
    """

    MODEL_NAME = "intfloat/multilingual-e5-large"
    DIMENSION = 1024
    MAX_TOKENS = 512  # Limite du modèle (SentencePiece tokens)
    BATCH_SIZE = 32   # Optimal sur CPU pour ce modèle

    def __init__(self, cache_dir: Optional[Path] = None):
        self._model = None
        self._cache_dir = cache_dir
        self._charge = False

    def charger(self) -> bool:
        """
        Tente de charger le modèle depuis HuggingFace ou le cache local.
        Retourne True si succès, False si le modèle est indisponible.
        """
        try:
            from sentence_transformers import SentenceTransformer
            kwargs = {}
            if self._cache_dir:
                kwargs["cache_folder"] = str(self._cache_dir)

            logger.info(f"Chargement de {self.MODEL_NAME}...")
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                **kwargs
            )
            self._charge = True
            logger.info(f"✓ {self.MODEL_NAME} chargé (dim={self.DIMENSION})")
            return True

        except Exception as e:
            logger.warning(
                f"✗ {self.MODEL_NAME} non disponible : {type(e).__name__}\n"
                f"  → Basculement sur TF-IDF+LSA (mode offline)"
            )
            return False

    @property
    def dimension(self) -> int:
        return self.DIMENSION

    @property
    def nom_modele(self) -> str:
        return self.MODEL_NAME

    def encoder_passages(self, textes: list[str]) -> list[list[float]]:
        """Encode des passages avec le préfixe requis."""
        if not self._charge:
            raise RuntimeError("Modèle non chargé. Appelez charger() d'abord.")
        textes_prefixes = [f"passage: {t}" for t in textes]
        return self._encoder_interne(textes_prefixes)

    def encoder_requetes(self, textes: list[str]) -> list[list[float]]:
        """Encode des requêtes avec le préfixe requis."""
        if not self._charge:
            raise RuntimeError("Modèle non chargé. Appelez charger() d'abord.")
        textes_prefixes = [f"query: {t}" for t in textes]
        return self._encoder_interne(textes_prefixes)

    def _encoder_interne(self, textes: list[str]) -> list[list[float]]:
        """Encode par batches pour éviter les OOM sur CPU."""
        import torch
        tous_embeddings = []

        for i in range(0, len(textes), self.BATCH_SIZE):
            batch = textes[i:i + self.BATCH_SIZE]
            with torch.no_grad():
                embeddings = self._model.encode(
                    batch,
                    normalize_embeddings=True,  # Normalisation L2 pour cosine similarity
                    show_progress_bar=False,
                )
            tous_embeddings.extend(embeddings.tolist())

        return tous_embeddings


# ---------------------------------------------------------------------------
# Embedder 2 : TF-IDF + LSA (fallback local)
# ---------------------------------------------------------------------------

class TFIDFEmbedder(BaseEmbedder):
    """
    Embedder local basé sur TF-IDF + LSA (Latent Semantic Analysis).

    Fallback robuste pour environnements sans accès réseau.

    Architecture :
      1. TF-IDF (n-grammes 1-2) → représentation lexicale sparse
      2. TruncatedSVD (LSA)     → projection dense en espace sémantique
      3. Normalisation L2        → compatibilité cosine similarity

    Avantages sur du texte réglementaire :
      - Vocabulaire hautement spécialisé et répétitif → TF-IDF très efficace
      - Les n-grammes bigrams capturent les expressions clés :
        "provisions techniques", "Best Estimate", "fonctions clés"
      - LSA capture la co-occurrence sémantique entre termes réglementaires

    Limites vs multilingual-e5-large :
      - Pas de compréhension contextuelle (syntaxe ignorée)
      - Moins robuste sur les reformulations sémantiques
      - Dimension plus petite → espace sémantique moins riche

    Usage prévu : développement offline, tests, démonstrations sans GPU.
    """

    def __init__(
        self,
        n_components: int = 256,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self._vectorizer = None
        self._svd = None
        self._dim_effective = None
        self._entrainement_corpus: list[str] = []
        self._entraine = False

    def entrainer(self, corpus: list[str]) -> "TFIDFEmbedder":
        """
        Entraîne le modèle TF-IDF + LSA sur un corpus de textes.

        Le corpus doit contenir TOUS les textes qui seront encodés
        (référentiel + chunks SFCR) pour que le vocabulaire soit complet.

        Args:
            corpus : Liste de textes d'entraînement

        Returns:
            self (pour chaînage)
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        from sklearn.pipeline import make_pipeline

        if not corpus:
            raise ValueError("Le corpus d'entraînement ne peut pas être vide")

        logger.info(
            f"Entraînement TF-IDF+LSA sur {len(corpus)} textes..."
        )

        # TF-IDF
        self._vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=1,           # Garder tous les termes (corpus petit)
            sublinear_tf=True,  # log(TF) → réduit l'influence des termes fréquents
            strip_accents=None, # Conserver les accents français
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-ZÀ-ÿ][a-zA-ZÀ-ÿ\-\']{1,}\b",
        )

        X = self._vectorizer.fit_transform(corpus)
        logger.info(f"  Matrice TF-IDF : {X.shape[0]} docs × {X.shape[1]} features")

        # LSA — dimension adaptée au corpus
        self._dim_effective = min(self.n_components, X.shape[1] - 1, X.shape[0] - 1)
        self._svd = TruncatedSVD(
            n_components=self._dim_effective,
            algorithm="randomized",
            n_iter=10,
            random_state=42,
        )

        X_lsa = self._svd.fit_transform(X)
        variance = self._svd.explained_variance_ratio_.sum()
        logger.info(
            f"  LSA : {self._dim_effective} composantes, "
            f"variance expliquée = {variance:.1%}"
        )

        self._entrainement_corpus = corpus
        self._entraine = True
        return self

    def _encoder_interne(self, textes: list[str]) -> list[list[float]]:
        """Encode et normalise L2 pour compatibilité cosine."""
        if not self._entraine:
            raise RuntimeError(
                "Modèle non entraîné. Appelez entrainer(corpus) d'abord."
            )
        from sklearn.preprocessing import normalize

        # Nettoyage des préfixes e5 si présents (TF-IDF n'en a pas besoin)
        textes_nettoyes = [
            t.removeprefix("query: ").removeprefix("passage: ")
            for t in textes
        ]

        X = self._vectorizer.transform(textes_nettoyes)
        X_lsa = self._svd.transform(X)
        X_norm = normalize(X_lsa, norm="l2")
        return X_norm.tolist()

    def encoder_passages(self, textes: list[str]) -> list[list[float]]:
        return self._encoder_interne(textes)

    def encoder_requetes(self, textes: list[str]) -> list[list[float]]:
        return self._encoder_interne(textes)

    @property
    def dimension(self) -> int:
        return self._dim_effective or self.n_components

    @property
    def nom_modele(self) -> str:
        return f"TF-IDF+LSA(dim={self.dimension},ngram={self.ngram_range})"

    def sauvegarder(self, chemin: Path) -> None:
        """Sérialise le modèle entraîné sur disque."""
        chemin.parent.mkdir(parents=True, exist_ok=True)
        with open(chemin, "wb") as f:
            pickle.dump({
                "vectorizer": self._vectorizer,
                "svd": self._svd,
                "dim_effective": self._dim_effective,
                "n_components": self.n_components,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
            }, f)
        logger.info(f"Modèle TF-IDF sauvegardé : {chemin}")

    @classmethod
    def charger(cls, chemin: Path) -> "TFIDFEmbedder":
        """Recharge un modèle sérialisé depuis disque."""
        with open(chemin, "rb") as f:
            data = pickle.load(f)
        embedder = cls(
            n_components=data["n_components"],
            max_features=data["max_features"],
            ngram_range=data["ngram_range"],
        )
        embedder._vectorizer = data["vectorizer"]
        embedder._svd = data["svd"]
        embedder._dim_effective = data["dim_effective"]
        embedder._entraine = True
        logger.info(f"Modèle TF-IDF rechargé : {chemin}")
        return embedder


# ---------------------------------------------------------------------------
# Manager principal — orchestre le choix du modèle
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """
    Gestionnaire d'embeddings avec sélection automatique de stratégie.

    Ordre de priorité :
      1. multilingual-e5-large (HuggingFace) → qualité maximale
      2. TF-IDF + LSA (local)                → fallback offline

    L'interface est identique dans les deux cas : le reste du pipeline
    n'a pas à connaître la stratégie active.

    Usage :
        manager = EmbeddingManager()
        manager.initialiser(corpus_entrainement=textes_referentiel)

        # Encoder des passages (chunks SFCR)
        embeddings = manager.encoder_passages(["Le rapport décrit..."])

        # Encoder des requêtes (exigences)
        embeddings = manager.encoder_requetes(["Le rapport mentionne-t-il..."])
    """

    def __init__(
        self,
        model_cache_dir: Optional[Path] = None,
        tfidf_model_path: Optional[Path] = None,
        forcer_tfidf: bool = False,
    ):
        """
        Args:
            model_cache_dir  : Répertoire cache pour HuggingFace
            tfidf_model_path : Chemin de sauvegarde du modèle TF-IDF
            forcer_tfidf     : Si True, utilise toujours TF-IDF (debug)
        """
        self._embedder: Optional[BaseEmbedder] = None
        self._model_cache_dir = model_cache_dir
        self._tfidf_model_path = tfidf_model_path or Path(
            "data/vectorstore/tfidf_model.pkl"
        )
        self._forcer_tfidf = forcer_tfidf
        self._strategie: Optional[str] = None

    def initialiser(
        self,
        corpus_entrainement: Optional[list[str]] = None,
    ) -> "EmbeddingManager":
        """
        Initialise le gestionnaire en sélectionnant la meilleure stratégie.

        Args:
            corpus_entrainement : Textes pour entraîner le TF-IDF si nécessaire.
                                  Doit contenir le référentiel + échantillon SFCR.

        Returns:
            self (pour chaînage)
        """
        # ── Stratégie 1 : multilingual-e5-large ───────────────────────
        if not self._forcer_tfidf:
            e5 = E5Embedder(cache_dir=self._model_cache_dir)
            if e5.charger():
                self._embedder = e5
                self._strategie = "e5-large"
                logger.info("Stratégie : multilingual-e5-large (qualité maximale)")
                return self

        # ── Stratégie 2 : TF-IDF + LSA ────────────────────────────────
        logger.info("Stratégie : TF-IDF+LSA (mode offline)")

        # Tenter de recharger un modèle existant
        if self._tfidf_model_path.exists():
            try:
                tfidf = TFIDFEmbedder.charger(self._tfidf_model_path)
                self._embedder = tfidf
                self._strategie = "tfidf-lsa"
                logger.info(f"Modèle TF-IDF rechargé depuis {self._tfidf_model_path}")
                return self
            except Exception as e:
                logger.warning(f"Rechargement TF-IDF échoué : {e} — réentraînement")

        # Entraîner un nouveau modèle
        if not corpus_entrainement:
            raise ValueError(
                "corpus_entrainement requis pour entraîner le modèle TF-IDF.\n"
                "Passez les textes du référentiel + échantillon SFCR."
            )

        tfidf = TFIDFEmbedder(n_components=256, max_features=10000)
        tfidf.entrainer(corpus_entrainement)
        tfidf.sauvegarder(self._tfidf_model_path)

        self._embedder = tfidf
        self._strategie = "tfidf-lsa"
        return self

    # ──────────────────────────────────────────────────────────────────────
    # API publique — identique quelle que soit la stratégie active
    # ──────────────────────────────────────────────────────────────────────

    def encoder_passages(self, textes: list[str]) -> list[list[float]]:
        """
        Encode des passages (chunks SFCR ou réglementaires) pour l'indexation.
        Applique automatiquement le préfixe "passage:" si e5 est actif.
        """
        self._verifier_initialise()
        if not textes:
            return []
        return self._embedder.encoder_passages(textes)

    def encoder_requetes(self, textes: list[str]) -> list[list[float]]:
        """
        Encode des requêtes (textes de vérification d'exigences) pour la recherche.
        Applique automatiquement le préfixe "query:" si e5 est actif.
        """
        self._verifier_initialise()
        if not textes:
            return []
        return self._embedder.encoder_requetes(textes)

    @property
    def dimension(self) -> int:
        self._verifier_initialise()
        return self._embedder.dimension

    @property
    def strategie(self) -> str:
        return self._strategie or "non-initialisé"

    @property
    def nom_modele(self) -> str:
        self._verifier_initialise()
        return self._embedder.nom_modele

    def _verifier_initialise(self):
        if self._embedder is None:
            raise RuntimeError(
                "EmbeddingManager non initialisé. "
                "Appelez initialiser() d'abord."
            )

    def rapport_configuration(self) -> str:
        """Retourne un résumé lisible de la configuration active."""
        return (
            f"EmbeddingManager\n"
            f"  Stratégie : {self.strategie}\n"
            f"  Modèle    : {self.nom_modele if self._embedder else 'N/A'}\n"
            f"  Dimension : {self.dimension if self._embedder else 'N/A'}\n"
        )
