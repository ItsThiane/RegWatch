"""
vector_store.py — Gestionnaire des bases vectorielles ChromaDB
=============================================================
Responsabilités :
  1. Créer et gérer deux collections ChromaDB persistantes :
       - "referentiel_sii"  : exigences réglementaires Solvabilité II
       - "sfcr_documents"   : chunks de rapports SFCR analysés
  2. Indexer les embeddings avec leurs métadonnées
  3. Réaliser la recherche de similarité cosinus
  4. Filtrer les résultats par métadonnée (section, pilier, thème...)
  5. Gérer la persistance sur disque

Pourquoi ChromaDB plutôt que FAISS ?
  - FAISS : ultra-rapide, mais IN-MEMORY uniquement → données perdues
    entre sessions → obligation de ré-indexer à chaque démarrage
  - ChromaDB : persistant sur disque (SQLite + index HNSW) → les
    embeddings sont calculés UNE SEULE FOIS puis réutilisés
  - ChromaDB supporte le filtrage par métadonnées (where clause) →
    on peut chercher UNIQUEMENT dans la Section B pour les exigences
    de gouvernance → retrieval plus précis et plus rapide
  - API Python native, intégration LangChain native

Architecture de stockage ChromaDB :
  data/vectorstore/
  ├── chroma.sqlite3          (index principal)
  └── [UUID]/                 (données des collections)

Schéma des métadonnées :
  Collection referentiel_sii :
    id_exigence, theme_id, pilier_id, source_id,
    niveau_obligation, poids_obligation, article

  Collection sfcr_documents :
    doc_id, entreprise, annee, section,
    page_debut, page_fin, nb_tokens
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

from src.ingestion.models import ChunkDocument, ExigenceAtomique
from src.rag.embedding_manager import EmbeddingManager

logger = logging.getLogger(__name__)

# Noms des collections — fixes pour garantir la cohérence entre sessions
COLLECTION_REFERENTIEL = "referentiel_sii"
COLLECTION_SFCR = "sfcr_documents"

# Nombre de résultats retournés par défaut lors d'une recherche
N_RESULTS_DEFAULT = 5


# ---------------------------------------------------------------------------
# Résultat d'une recherche vectorielle
# ---------------------------------------------------------------------------

class ResultatRecherche:
    """
    Encapsule les résultats d'une recherche de similarité ChromaDB.
    Fournit un accès structuré aux passages, scores et métadonnées.
    """

    def __init__(
        self,
        ids: list[str],
        documents: list[str],
        metadonnees: list[dict],
        distances: list[float],
    ):
        self.ids = ids
        self.documents = documents
        self.metadonnees = metadonnees
        # ChromaDB retourne des distances (L2 ou cosine) → convertir en scores
        # Pour la similarité cosinus : score = 1 - distance
        self.scores_similarite = [
            max(0.0, min(1.0, 1.0 - d)) for d in distances
        ]

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        for i in range(len(self)):
            yield {
                "id": self.ids[i],
                "texte": self.documents[i],
                "metadata": self.metadonnees[i],
                "score": self.scores_similarite[i],
            }

    def top1(self) -> Optional[dict]:
        if not self.ids:
            return None
        return {
            "id": self.ids[0],
            "texte": self.documents[0],
            "metadata": self.metadonnees[0],
            "score": self.scores_similarite[0],
        }

    def to_list(self) -> list[dict]:
        return list(self)

    def __repr__(self) -> str:
        scores = [f"{s:.3f}" for s in self.scores_similarite]
        return f"ResultatRecherche(n={len(self)}, scores={scores})"


# ---------------------------------------------------------------------------
# Gestionnaire ChromaDB
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Interface unifiée pour les deux collections ChromaDB du projet.

    Usage :
        store = VectorStore(persist_dir=Path("data/vectorstore"))
        store.initialiser(embedding_manager)

        # Indexer le référentiel
        store.indexer_referentiel(chunks, embeddings, metadonnees)

        # Indexer un SFCR
        store.indexer_sfcr(chunks, embeddings)

        # Rechercher les passages pertinents pour une exigence
        resultats = store.rechercher_passages_sfcr(
            embedding_requete, n_results=5
        )
    """

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or Path("data/vectorstore")
        self._client: Optional[chromadb.ClientAPI] = None
        self._col_referentiel: Optional[chromadb.Collection] = None
        self._col_sfcr: Optional[chromadb.Collection] = None
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._dimension: Optional[int] = None

    # ──────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────

    def initialiser(self, embedding_manager: EmbeddingManager) -> "VectorStore":
        """
        Initialise le client ChromaDB persistant et crée/charge les collections.

        ChromaDB PersistentClient utilise :
          - SQLite pour les métadonnées et l'index invertis
          - HNSW (Hierarchical Navigable Small World) pour l'index vectoriel
          - Fichiers binaires pour les embeddings bruts

        La distance "cosine" est choisie car :
          - Nos embeddings sont normalisés L2 (les deux stratégies)
          - La similarité cosinus est invariante à la magnitude
          - Standard pour la comparaison de représentations textuelles

        Args:
            embedding_manager : Gestionnaire d'embeddings initialisé

        Returns:
            self (pour chaînage)
        """
        self._embedding_manager = embedding_manager
        self._dimension = embedding_manager.dimension

        # Créer le répertoire de persistance
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Client persistant ChromaDB
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,  # Pas de télémétrie
                allow_reset=True,
            ),
        )

        # Métadonnées de la collection (pour compatibilité dimension)
        metadata_collection = {
            "hnsw:space": "cosine",       # Distance cosinus
            "hnsw:construction_ef": 200,  # Qualité de construction HNSW
            "hnsw:search_ef": 100,        # Qualité de recherche HNSW
            "hnsw:M": 16,                 # Connexions par nœud HNSW
            "dimension": str(self._dimension),
            "modele": embedding_manager.nom_modele,
        }

        # Créer ou récupérer les collections
        self._col_referentiel = self._client.get_or_create_collection(
            name=COLLECTION_REFERENTIEL,
            metadata=metadata_collection,
        )
        self._col_sfcr = self._client.get_or_create_collection(
            name=COLLECTION_SFCR,
            metadata=metadata_collection,
        )

        logger.info(
            f"VectorStore initialisé : {self.persist_dir}\n"
            f"  [{COLLECTION_REFERENTIEL}] : "
            f"{self._col_referentiel.count()} documents\n"
            f"  [{COLLECTION_SFCR}] : "
            f"{self._col_sfcr.count()} documents"
        )

        return self

    # ──────────────────────────────────────────────────────────────────────
    # Indexation du référentiel réglementaire
    # ──────────────────────────────────────────────────────────────────────

    def indexer_referentiel(
        self,
        exigences: list[ExigenceAtomique],
        embeddings: list[list[float]],
    ) -> int:
        """
        Indexe les exigences réglementaires dans la collection référentiel.

        Chaque exigence = 1 document ChromaDB avec :
          - id      : id_exigence (ex: "SII-P3-REG2015-Art290-§1")
          - document: texte_verification (texte utilisé pour l'embedding)
          - embedding: vecteur dense calculé par EmbeddingManager
          - metadata: tous les champs utiles pour le filtrage et le scoring

        Args:
            exigences  : Liste d'ExigenceAtomique
            embeddings : Embeddings correspondants (même ordre)

        Returns:
            Nombre d'exigences indexées
        """
        self._verifier_initialise()

        if len(exigences) != len(embeddings):
            raise ValueError(
                f"Nombre d'exigences ({len(exigences)}) ≠ "
                f"nombre d'embeddings ({len(embeddings)})"
            )

        # Vérifier quelles exigences ne sont pas encore indexées
        ids_existants = set(self._col_referentiel.get()["ids"])
        a_indexer = [
            (e, emb) for e, emb in zip(exigences, embeddings)
            if e.id_exigence not in ids_existants
        ]

        if not a_indexer:
            logger.info("Référentiel déjà indexé — aucune mise à jour nécessaire")
            return 0

        ids = [e.id_exigence for e, _ in a_indexer]
        docs = [e.texte_verification for e, _ in a_indexer]
        embs = [emb for _, emb in a_indexer]
        metas = [self._meta_exigence(e) for e, _ in a_indexer]

        # Indexation par batches (ChromaDB limite à ~41666 par appel)
        BATCH = 500
        nb_indexe = 0
        for i in range(0, len(ids), BATCH):
            self._col_referentiel.add(
                ids=ids[i:i+BATCH],
                documents=docs[i:i+BATCH],
                embeddings=embs[i:i+BATCH],
                metadatas=metas[i:i+BATCH],
            )
            nb_indexe += len(ids[i:i+BATCH])

        logger.info(
            f"Référentiel indexé : {nb_indexe} exigences "
            f"(total: {self._col_referentiel.count()})"
        )
        return nb_indexe

    def _meta_exigence(self, e: ExigenceAtomique) -> dict:
        """Construit les métadonnées ChromaDB d'une exigence."""
        # Récupérer le pilier depuis le theme_id
        pilier_id = f"P{e.theme_id[1]}" if len(e.theme_id) >= 2 else "P3"
        return {
            "id_exigence": e.id_exigence,
            "theme_id": e.theme_id,
            "pilier_id": pilier_id,
            "source_id": e.source_id,
            "article": e.article,
            "niveau_obligation": e.niveau_obligation.value,
            "poids_obligation": e.poids_obligation,
            "mots_cles": json.dumps(e.mots_cles, ensure_ascii=False),
            "texte_original": e.texte_original[:500],  # Tronqué pour ChromaDB
        }

    # ──────────────────────────────────────────────────────────────────────
    # Indexation des chunks SFCR
    # ──────────────────────────────────────────────────────────────────────

    def indexer_sfcr(
        self,
        chunks: list[ChunkDocument],
        embeddings: list[list[float]],
        entreprise: str = "",
        annee: int = 0,
    ) -> int:
        """
        Indexe les chunks d'un rapport SFCR dans la collection sfcr_documents.

        Args:
            chunks     : Chunks produits par SFCRChunker
            embeddings : Embeddings correspondants
            entreprise : Nom de l'entreprise (pour filtrage multi-doc)
            annee      : Année du rapport

        Returns:
            Nombre de chunks indexés
        """
        self._verifier_initialise()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Nombre de chunks ({len(chunks)}) ≠ "
                f"nombre d'embeddings ({len(embeddings)})"
            )

        if not chunks:
            logger.warning("Aucun chunk à indexer")
            return 0

        # Éviter les doublons (même chunk_id)
        ids_existants = set(self._col_sfcr.get()["ids"])
        a_indexer = [
            (c, emb) for c, emb in zip(chunks, embeddings)
            if c.chunk_id not in ids_existants
        ]

        if not a_indexer:
            logger.info("Chunks déjà indexés — aucune mise à jour nécessaire")
            return 0

        ids = [c.chunk_id for c, _ in a_indexer]
        docs = [c.texte for c, _ in a_indexer]
        embs = [emb for _, emb in a_indexer]
        metas = [
            self._meta_chunk(c, entreprise, annee)
            for c, _ in a_indexer
        ]

        BATCH = 500
        nb_indexe = 0
        for i in range(0, len(ids), BATCH):
            self._col_sfcr.add(
                ids=ids[i:i+BATCH],
                documents=docs[i:i+BATCH],
                embeddings=embs[i:i+BATCH],
                metadatas=metas[i:i+BATCH],
            )
            nb_indexe += len(ids[i:i+BATCH])

        logger.info(
            f"SFCR indexé : {nb_indexe} chunks "
            f"(total: {self._col_sfcr.count()})"
        )
        return nb_indexe

    def _meta_chunk(
        self, chunk: ChunkDocument, entreprise: str, annee: int
    ) -> dict:
        """Construit les métadonnées ChromaDB d'un chunk SFCR."""
        return {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "type_doc": chunk.type_doc,
            "section": chunk.section or "",
            "page_debut": chunk.page_debut or 0,
            "page_fin": chunk.page_fin or 0,
            "nb_tokens": chunk.nb_tokens,
            "position": chunk.position_dans_doc,
            "entreprise": entreprise,
            "annee": annee,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Recherche de similarité
    # ──────────────────────────────────────────────────────────────────────

    def rechercher_passages_sfcr(
        self,
        embedding_requete: list[float],
        n_results: int = N_RESULTS_DEFAULT,
        filtre_section: Optional[str] = None,
        filtre_doc_id: Optional[str] = None,
        filtre_entreprise: Optional[str] = None,
        filtre_annee: Optional[int] = None,
    ) -> ResultatRecherche:
        """
        Recherche les passages SFCR les plus similaires à une requête.

        Le filtrage par métadonnée (section, doc_id) est réalisé AVANT
        le calcul de similarité → plus efficace qu'un post-filtrage.
        ChromaDB implémente ce filtrage nativement dans l'index HNSW.

        Args:
            embedding_requete : Embedding de l'exigence à vérifier
            n_results         : Nombre de résultats à retourner
            filtre_section    : Filtrer sur une section SFCR spécifique
                                Ex: "B_GOUVERNANCE" pour les exigences Pilier 2
            filtre_doc_id     : Filtrer sur un document spécifique
            filtre_entreprise : Filtrer sur une entreprise
            filtre_annee      : Filtrer sur une année

        Returns:
            ResultatRecherche avec passages, scores et métadonnées
        """
        self._verifier_initialise()

        # Construction du filtre ChromaDB (syntaxe MongoDB-like)
        where = self._construire_filtre(
            section=filtre_section,
            doc_id=filtre_doc_id,
            entreprise=filtre_entreprise,
            annee=filtre_annee,
        )

        try:
            kwargs = {
                "query_embeddings": [embedding_requete],
                "n_results": min(n_results, self._col_sfcr.count() or 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            resultats = self._col_sfcr.query(**kwargs)

            return ResultatRecherche(
                ids=resultats["ids"][0],
                documents=resultats["documents"][0],
                metadonnees=resultats["metadatas"][0],
                distances=resultats["distances"][0],
            )
        except Exception as e:
            logger.error(f"Erreur recherche SFCR : {e}")
            return ResultatRecherche([], [], [], [])

    def rechercher_exigences(
        self,
        embedding_passage: list[float],
        n_results: int = 3,
        filtre_theme: Optional[str] = None,
        filtre_niveau: Optional[str] = None,
    ) -> ResultatRecherche:
        """
        Recherche les exigences réglementaires similaires à un passage SFCR.

        Usage inverse : à partir d'un passage SFCR, retrouver les exigences
        qu'il couvre. Utile pour l'analyse exploratoire.
        """
        self._verifier_initialise()

        where = self._construire_filtre(
            theme=filtre_theme,
            niveau_obligation=filtre_niveau,
        )

        try:
            kwargs = {
                "query_embeddings": [embedding_passage],
                "n_results": min(n_results, self._col_referentiel.count() or 1),
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            resultats = self._col_referentiel.query(**kwargs)

            return ResultatRecherche(
                ids=resultats["ids"][0],
                documents=resultats["documents"][0],
                metadonnees=resultats["metadatas"][0],
                distances=resultats["distances"][0],
            )
        except Exception as e:
            logger.error(f"Erreur recherche référentiel : {e}")
            return ResultatRecherche([], [], [], [])

    def _construire_filtre(self, **kwargs) -> Optional[dict]:
        """
        Construit un filtre ChromaDB (syntaxe where) depuis des arguments nommés.
        Retourne None si aucun filtre actif.
        """
        conditions = []

        mapping = {
            "section": "section",
            "doc_id": "doc_id",
            "entreprise": "entreprise",
            "theme": "theme_id",
            "niveau_obligation": "niveau_obligation",
        }

        for arg_name, meta_key in mapping.items():
            valeur = kwargs.get(arg_name)
            if valeur is not None:
                conditions.append({meta_key: {"$eq": valeur}})

        # Filtre numérique pour l'année
        annee = kwargs.get("annee")
        if annee is not None:
            conditions.append({"annee": {"$eq": annee}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ──────────────────────────────────────────────────────────────────────
    # Statistiques et diagnostic
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Statistiques des deux collections."""
        self._verifier_initialise()
        return {
            "referentiel": {
                "nb_documents": self._col_referentiel.count(),
                "nom_collection": COLLECTION_REFERENTIEL,
            },
            "sfcr": {
                "nb_documents": self._col_sfcr.count(),
                "nom_collection": COLLECTION_SFCR,
            },
            "persist_dir": str(self.persist_dir),
            "dimension": self._dimension,
            "modele": self._embedding_manager.nom_modele
            if self._embedding_manager else "N/A",
        }

    def afficher_stats(self) -> str:
        s = self.stats()
        return (
            f"{'='*50}\n"
            f"VECTOR STORE — STATISTIQUES\n"
            f"{'='*50}\n"
            f"Répertoire    : {s['persist_dir']}\n"
            f"Modèle        : {s['modele']}\n"
            f"Dimension     : {s['dimension']}\n"
            f"Référentiel   : {s['referentiel']['nb_documents']} exigences indexées\n"
            f"SFCR          : {s['sfcr']['nb_documents']} chunks indexés\n"
        )

    def reset_sfcr(self) -> None:
        """Supprime tous les chunks SFCR (utile pour re-indexer un document)."""
        self._verifier_initialise()
        self._client.delete_collection(COLLECTION_SFCR)
        from chromadb.config import Settings
        self._col_sfcr = self._client.get_or_create_collection(
            name=COLLECTION_SFCR,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection SFCR réinitialisée")

    def _verifier_initialise(self):
        if self._client is None:
            raise RuntimeError(
                "VectorStore non initialisé. Appelez initialiser() d'abord."
            )
