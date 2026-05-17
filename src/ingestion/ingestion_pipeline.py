"""
ingestion_pipeline.py — Pipeline d'ingestion complet
=====================================================
Orchestre le flux : PDF → parse → chunk → validation → prêt pour ChromaDB.

Ce module est le point d'entrée unique pour ingérer un document SFCR.
Il produit deux artefacts :
  1. Un DocumentSFCR (métadonnées du document)
  2. Une liste de ChunkDocument (prêts pour l'embedding et l'indexation)

Il gère aussi l'ingestion du référentiel réglementaire
(exigences atomiques → chunks → prêts pour ChromaDB).

Usage typique :
    pipeline = IngestionPipeline()

    # Ingérer un SFCR
    resultat = pipeline.ingerer_sfcr(
        Path("sfcr_2023_groupama.pdf"),
        entreprise="Groupama",
        annee=2023,
    )

    # Ingérer le référentiel
    chunks_reg = pipeline.ingerer_referentiel()
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.ingestion.models import DocumentSFCR, ChunkDocument, ExigenceAtomique
from src.ingestion.pdf_parser import SFCRParser, DocumentParse
from src.ingestion.sfcr_chunker import SFCRChunker, ResultatChunking
from src.ingestion.referentiel_loader import ReferentielLoader
from src.ingestion.text_utils import preparer_requete_e5, compter_tokens

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de résultat
# ---------------------------------------------------------------------------

@dataclass
class ResultatIngestionSFCR:
    """Résultat complet de l'ingestion d'un rapport SFCR."""
    doc_sfcr: DocumentSFCR
    doc_parse: DocumentParse
    resultat_chunking: ResultatChunking
    chunks: list[ChunkDocument]
    textes_embedding: list[str]      # Textes formatés "passage: ..." pour e5
    succes: bool = True
    erreurs: list[str] = field(default_factory=list)

    @property
    def pret_pour_indexation(self) -> bool:
        return self.succes and len(self.chunks) > 0

    def rapport(self) -> str:
        lines = [
            "=" * 60,
            "RAPPORT D'INGESTION SFCR",
            "=" * 60,
            f"Entreprise   : {self.doc_sfcr.entreprise}",
            f"Année        : {self.doc_sfcr.annee_rapport}",
            f"Fichier      : {self.doc_sfcr.nom_fichier}",
            f"Hash SHA-256 : {self.doc_sfcr.hash_sha256[:16]}...",
            f"Pages        : {self.doc_sfcr.nb_pages}",
            f"Langue       : {self.doc_sfcr.langue}",
            "",
            "--- Parsing ---",
            f"Sections détectées : {', '.join(self.doc_parse.sections_detectees) or 'aucune'}",
            f"Pages QRT         : {self.doc_parse.nb_pages_tableau}",
            f"Qualité parsing   : {self.doc_parse.qualite_extraction if hasattr(self.doc_parse, 'qualite_extraction') else 'N/A'}",
            "",
            "--- Chunking ---",
            self.resultat_chunking.rapport(),
            "",
            f"Prêt pour indexation : {'✓ OUI' if self.pret_pour_indexation else '✗ NON'}",
        ]
        if self.erreurs:
            lines.append("Erreurs :")
            for e in self.erreurs:
                lines.append(f"  ✗ {e}")
        return "\n".join(lines)


@dataclass
class ResultatIngestionReferentiel:
    """Résultat de l'ingestion du référentiel réglementaire."""
    chunks: list[ChunkDocument]
    textes_embedding: list[str]
    ids_exigences: list[str]         # Traçabilité chunk → exigence
    nb_exigences: int = 0
    nb_chunks: int = 0

    def rapport(self) -> str:
        return (
            f"Référentiel ingéré : {self.nb_exigences} exigences → "
            f"{self.nb_chunks} chunks prêts pour indexation"
        )


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Pipeline d'ingestion complet : PDF → chunks prêts pour ChromaDB.

    Paramètres :
        chunk_size    : Taille cible des chunks en tokens (défaut: 450)
        chunk_overlap : Overlap entre chunks en tokens (défaut: 80)
        data_dir      : Répertoire des données (défaut: data/)
    """

    def __init__(
        self,
        chunk_size: int = 450,
        chunk_overlap: int = 80,
        data_dir: Optional[Path] = None,
    ):
        self.data_dir = data_dir or Path("data")
        self.parser = SFCRParser()
        self.chunker = SFCRChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.loader = ReferentielLoader(
            data_dir=self.data_dir / "processed" / "exigences"
        )
        self._referentiel_charge = False

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion d'un SFCR
    # ──────────────────────────────────────────────────────────────────────

    def ingerer_sfcr(
        self,
        pdf_path: Path,
        entreprise: str,
        annee: int,
        langue: str = "fr",
    ) -> ResultatIngestionSFCR:
        """
        Ingère un rapport SFCR complet.

        Flux :
          1. Validation du fichier
          2. Parsing PDF (SFCRParser)
          3. Création du DocumentSFCR (métadonnées)
          4. Chunking (SFCRChunker)
          5. Préparation des textes pour l'embedding
          6. Retour du ResultatIngestionSFCR

        Args:
            pdf_path  : Chemin vers le PDF
            entreprise: Nom de l'entreprise (ex: "Groupama SA")
            annee     : Année de l'exercice de référence (ex: 2023)
            langue    : Langue du rapport ("fr" par défaut)

        Returns:
            ResultatIngestionSFCR prêt pour l'indexation ChromaDB
        """
        logger.info(f"Ingestion SFCR : {pdf_path.name} ({entreprise} {annee})")
        erreurs = []

        # ── Étape 1 : Parsing PDF ──────────────────────────────────────
        try:
            doc_parse = self.parser.parser(pdf_path)
        except FileNotFoundError as e:
            return self._resultat_echec(str(e), pdf_path, entreprise, annee)
        except ValueError as e:
            return self._resultat_echec(f"PDF invalide : {e}", pdf_path, entreprise, annee)
        except Exception as e:
            return self._resultat_echec(f"Erreur parsing : {e}", pdf_path, entreprise, annee)

        # ── Étape 2 : Création du DocumentSFCR ────────────────────────
        doc_sfcr = DocumentSFCR(
            id=str(uuid.uuid4()),
            nom_fichier=pdf_path.name,
            entreprise=entreprise,
            annee_rapport=annee,
            nb_pages=doc_parse.nb_pages,
            langue=langue,
            hash_sha256=doc_parse.hash_sha256,
        )

        # Avertissement si sections manquantes
        sections_detectees = doc_parse.sections_detectees
        if len(sections_detectees) < 3:
            erreurs.append(
                f"Seulement {len(sections_detectees)}/5 sections SFCR "
                f"détectées ({', '.join(sections_detectees) or 'aucune'}). "
                f"La qualité du RAG sera réduite."
            )

        # ── Étape 3 : Chunking ─────────────────────────────────────────
        resultat_chunking = self.chunker.chunker(doc_parse, doc_sfcr)

        if resultat_chunking.nb_chunks == 0:
            erreurs.append("Aucun chunk valide produit — vérifier le PDF")

        # ── Étape 4 : Préparation pour l'embedding ────────────────────
        textes_embedding = self.chunker.preparer_textes_embedding(
            resultat_chunking.chunks
        )

        succes = len(erreurs) == 0 or resultat_chunking.nb_chunks > 0

        logger.info(
            f"Ingestion terminée : {resultat_chunking.nb_chunks} chunks, "
            f"succes={succes}"
        )

        return ResultatIngestionSFCR(
            doc_sfcr=doc_sfcr,
            doc_parse=doc_parse,
            resultat_chunking=resultat_chunking,
            chunks=resultat_chunking.chunks,
            textes_embedding=textes_embedding,
            succes=succes,
            erreurs=erreurs,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Ingestion du référentiel réglementaire
    # ──────────────────────────────────────────────────────────────────────

    def ingerer_referentiel(self) -> ResultatIngestionReferentiel:
        """
        Ingère le référentiel Solvabilité II complet.

        Chaque exigence atomique devient UN chunk réglementaire.
        Le texte utilisé pour l'embedding est le texte_verification
        (la question posée au LLM) — c'est ce texte qui sera comparé
        aux passages du SFCR lors du retrieval.

        IMPORTANT : on utilise texte_verification et non texte_original
        car texte_verification est formulé comme une question/assertion
        orientée vers le contenu attendu dans le SFCR, ce qui améliore
        la similarité cosinus avec les passages réels du rapport.

        Returns:
            ResultatIngestionReferentiel avec chunks et textes prêts
        """
        if not self._referentiel_charge:
            self.loader.charger(strict=True)
            self._referentiel_charge = True

        exigences = self.loader.get_toutes_exigences()
        chunks: list[ChunkDocument] = []
        textes_embedding: list[str] = []
        ids_exigences: list[str] = []
        position = 0

        for exigence in exigences:
            # Texte pour le chunk = texte_normalise (complet, structuré)
            texte_chunk = exigence.texte_normalise
            if not texte_chunk.strip():
                logger.warning(f"texte_normalise vide pour {exigence.id_exigence}")
                continue

            # Chunking de l'exigence (généralement 1 seul chunk)
            chunks_exigence = self.chunker.chunker_referentiel(
                texte=texte_chunk,
                source_id=exigence.source_id,
                article=exigence.article,
                id_exigence=exigence.id_exigence,
                position=position,
            )

            for chunk in chunks_exigence:
                chunks.append(chunk)
                ids_exigences.append(exigence.id_exigence)

                # CRITIQUE : préfixe "query: " pour le texte_verification
                # Le texte_verification est la "requête" qui cherche dans le SFCR
                texte_emb = preparer_requete_e5(
                    exigence.texte_verification, type_doc="query"
                )
                textes_embedding.append(texte_emb)

            position += len(chunks_exigence)

        logger.info(
            f"Référentiel ingéré : {len(exigences)} exigences → "
            f"{len(chunks)} chunks"
        )

        return ResultatIngestionReferentiel(
            chunks=chunks,
            textes_embedding=textes_embedding,
            ids_exigences=ids_exigences,
            nb_exigences=len(exigences),
            nb_chunks=len(chunks),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────

    def _resultat_echec(
        self,
        message: str,
        pdf_path: Path,
        entreprise: str,
        annee: int,
    ) -> ResultatIngestionSFCR:
        """Construit un ResultatIngestionSFCR d'échec."""
        logger.error(f"Échec ingestion : {message}")
        doc_sfcr = DocumentSFCR(
            id=str(uuid.uuid4()),
            nom_fichier=pdf_path.name,
            entreprise=entreprise,
            annee_rapport=annee,
        )
        # DocumentParse minimal pour éviter les erreurs en aval
        from src.ingestion.pdf_parser import DocumentParse
        doc_parse = DocumentParse(
            chemin_fichier=str(pdf_path),
            hash_sha256="",
            nb_pages=0,
        )
        resultat_chunking = ResultatChunking(
            doc_id=doc_sfcr.id,
            nom_fichier=pdf_path.name,
            chunks=[],
        )
        return ResultatIngestionSFCR(
            doc_sfcr=doc_sfcr,
            doc_parse=doc_parse,
            resultat_chunking=resultat_chunking,
            chunks=[],
            textes_embedding=[],
            succes=False,
            erreurs=[message],
        )
