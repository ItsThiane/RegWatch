"""
sfcr_chunker.py — Chunker haute qualité pour les rapports SFCR
===============================================================
Responsabilités :
  1. Segmenter le DocumentParse en chunks exploitables par ChromaDB
  2. Appliquer une stratégie différenciée selon le type de document
  3. Attacher des métadonnées riches à chaque chunk (section, page, tokens)
  4. Valider la qualité de chaque chunk avant indexation
  5. Produire des chunks prêts pour l'embedding multilingual-e5-large

Stratégie de chunking — deux régimes distincts :

  SFCR (texte narratif) :
    → RecursiveCharacterTextSplitter
    → Taille cible : 450 tokens, overlap : 80 tokens
    → Séparateurs : ["\n\n", "\n", ". ", " "]
    → Grain : paragraphe avec contexte de transition
    → Métadonnées : section (A-E), page_debut, page_fin, sous_section

  Référentiel réglementaire :
    → Chaque exigence atomique = 1 chunk (déjà fait en Phase 1)
    → Taille : variable (100-400 tokens)
    → Pas d'overlap (exigences autonomes)

Pourquoi RecursiveCharacterTextSplitter ?
  - Respecte les frontières naturelles : paragraphes > phrases > mots
  - Évite de couper en milieu de phrase (perte de sens pour l'embedding)
  - L'overlap de 80 tokens capture les phrases de transition entre paragraphes
    qui contiennent souvent les liens logiques entre exigences
  - Configurable par liste de séparateurs → adapté au français
"""

from __future__ import annotations
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.models import ChunkDocument, DocumentSFCR
from src.ingestion.pdf_parser import DocumentParse

# Mapping lettre section 2192 identifiant normalis00e9
SECTIONS_SFCR = {
    "A": "A_ACTIVITE_RESULTATS",
    "B": "B_GOUVERNANCE",
    "C": "C_PROFIL_RISQUE",
    "D": "D_VALORISATION",
    "E": "E_GESTION_CAPITAL",
}
from src.ingestion.text_utils import (
    compter_tokens,
    normaliser_pour_embedding,
    preparer_requete_e5,
    statistiques_texte,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes de chunking — justifiées et documentées
# ---------------------------------------------------------------------------

# Taille cible en tokens pour les chunks SFCR
# Justification : multilingual-e5-large a une fenêtre de 512 tokens.
# On vise 450 pour laisser de la marge au préfixe "passage: " et à la
# sécurité de troncature. En dessous de 200, le contexte est insuffisant.
CHUNK_SIZE_SFCR_TOKENS = 450

# Overlap entre chunks consécutifs (en tokens)
# Justification : ~18% de la taille du chunk.
# Ratio recommandé en littérature RAG : 15-25%.
# 80 tokens ≈ 2-3 phrases, suffisant pour les transitions de paragraphe.
CHUNK_OVERLAP_TOKENS = 80

# Taille minimale d'un chunk valide (en tokens)
# En dessous, le chunk n'a pas assez de contexte pour l'embedding.
CHUNK_MIN_TOKENS = 50

# Taille maximale d'un chunk (en tokens) — guard-rail de sécurité
CHUNK_MAX_TOKENS = 550

# Séparateurs pour le RecursiveCharacterTextSplitter
# Ordre décroissant de préférence : couper d'abord sur des frontières larges
SEPARATEURS_FR = [
    "\n\n",     # Frontière de paragraphe (préféré)
    "\n",       # Saut de ligne
    ". ",       # Fin de phrase (français)
    ".\n",      # Fin de phrase + saut
    "! ",       # Exclamation
    "? ",       # Question
    "; ",       # Point-virgule
    ", ",       # Virgule (dernier recours)
    " ",        # Espace (dernier recours absolu)
    "",         # Caractère par caractère (garde-fou)
]


# ---------------------------------------------------------------------------
# Résultat du chunking
# ---------------------------------------------------------------------------

@dataclass
class ResultatChunking:
    """Résultat complet du chunking d'un document."""
    doc_id: str
    nom_fichier: str
    chunks: list[ChunkDocument]
    nb_chunks: int = 0
    nb_chunks_rejetes: int = 0
    nb_tokens_total: int = 0
    raisons_rejet: dict[str, int] = field(default_factory=dict)
    avertissements: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.nb_chunks = len(self.chunks)
        self.nb_tokens_total = sum(c.nb_tokens for c in self.chunks)

    def rapport(self) -> str:
        lines = [
            "=" * 55,
            "RÉSULTAT DU CHUNKING",
            "=" * 55,
            f"Document     : {self.nom_fichier}",
            f"Chunks valides : {self.nb_chunks}",
            f"Chunks rejetés : {self.nb_chunks_rejetes}",
            f"Tokens total   : {self.nb_tokens_total:,}",
        ]
        if self.raisons_rejet:
            lines.append("Raisons de rejet :")
            for raison, nb in self.raisons_rejet.items():
                lines.append(f"  {raison:<30} : {nb}")
        if self.avertissements:
            lines.append("Avertissements :")
            for w in self.avertissements:
                lines.append(f"  ⚠ {w}")

        # Distribution par section
        sections_count: dict[str, int] = {}
        for chunk in self.chunks:
            s = chunk.section or "NON_CLASSIFIE"
            sections_count[s] = sections_count.get(s, 0) + 1
        if sections_count:
            lines.append("Distribution par section :")
            for sec, nb in sorted(sections_count.items()):
                lines.append(f"  [{sec:<25}] : {nb:>3} chunks")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chunker principal
# ---------------------------------------------------------------------------

class SFCRChunker:
    """
    Transforme un DocumentParse en liste de ChunkDocument prêts pour ChromaDB.

    Usage :
        chunker = SFCRChunker()
        resultat = chunker.chunker(doc_parse, doc_sfcr)
        print(resultat.rapport())
        chunks = resultat.chunks  # → ChromaDB
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE_SFCR_TOKENS,
        chunk_overlap: int = CHUNK_OVERLAP_TOKENS,
        min_tokens: int = CHUNK_MIN_TOKENS,
        max_tokens: int = CHUNK_MAX_TOKENS,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens

        # Initialisation du splitter LangChain
        # length_function = compter_tokens garantit que les tailles sont en
        # tokens réels (cl100k_base) et non en caractères bruts.
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=compter_tokens,
            separators=SEPARATEURS_FR,
            keep_separator=True,  # Conserver les séparateurs pour la lisibilité
            is_separator_regex=False,
        )

        logger.info(
            f"SFCRChunker initialisé : "
            f"size={chunk_size} tokens, overlap={chunk_overlap} tokens"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────────────────────────────

    def chunker(
        self,
        doc_parse: DocumentParse,
        doc_sfcr: DocumentSFCR,
    ) -> ResultatChunking:
        """
        Chunke un DocumentParse complet en chunks valides pour ChromaDB.

        Args:
            doc_parse  : Résultat du SFCRParser
            doc_sfcr   : Métadonnées du document SFCR (id, entreprise...)

        Returns:
            ResultatChunking avec tous les chunks et le rapport de diagnostic
        """
        chunks_valides: list[ChunkDocument] = []
        nb_rejetes = 0
        raisons_rejet: dict[str, int] = {}
        avertissements: list[str] = []

        # Stratégie : chunker section par section pour préserver le contexte
        # Si des sections ont été détectées, on les traite individuellement
        # Sinon, fallback sur le texte page par page

        if doc_parse.index_sections:
            # Mode nominal : sections A-E détectées
            chunks_par_section = self._chunker_par_sections(
                doc_parse, doc_sfcr.id
            )
        else:
            # Mode dégradé : pas de sections → chunking page par page
            avertissements.append(
                "Aucune section A-E détectée — chunking page par page "
                "(qualité réduite)"
            )
            chunks_par_section = self._chunker_par_pages(
                doc_parse, doc_sfcr.id
            )

        # Validation de chaque chunk
        for chunk in chunks_par_section:
            valide, raison = self._valider_chunk(chunk)
            if valide:
                chunks_valides.append(chunk)
            else:
                nb_rejetes += 1
                raisons_rejet[raison] = raisons_rejet.get(raison, 0) + 1

        # Mise à jour du compteur de chunks dans le doc_sfcr
        doc_sfcr.nb_chunks = len(chunks_valides)

        logger.info(
            f"Chunking terminé : {len(chunks_valides)} chunks valides, "
            f"{nb_rejetes} rejetés"
        )

        return ResultatChunking(
            doc_id=doc_sfcr.id,
            nom_fichier=doc_sfcr.nom_fichier,
            chunks=chunks_valides,
            nb_chunks_rejetes=nb_rejetes,
            raisons_rejet=raisons_rejet,
            avertissements=avertissements,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Chunking par section (mode nominal)
    # ──────────────────────────────────────────────────────────────────────

    def _chunker_par_sections(
        self, doc_parse: DocumentParse, doc_id: str
    ) -> list[ChunkDocument]:
        """
        Chunke le document section par section.

        L'avantage fondamental : chaque chunk hérite de la section SFCR
        dont il est issu. Le RAG peut ainsi filtrer par section lors du
        retrieval (ex : ne chercher les exigences de gouvernance que dans
        la Section B).
        """
        tous_chunks: list[ChunkDocument] = []
        position_globale = 0

        for section_lettre, pages_nums in sorted(doc_parse.index_sections.items()):
            # Assembler le texte de la section depuis les pages concernées
            texte_section = self._assembler_texte_section(
                doc_parse, pages_nums
            )

            if not texte_section.strip():
                logger.debug(f"Section {section_lettre} vide — ignorée")
                continue

            # Identifier l'id de section normalisé
            section_id = SECTIONS_SFCR.get(section_lettre, f"SECTION_{section_lettre}")

            # Découper en chunks
            morceaux = self._splitter.split_text(texte_section)

            for i, morceau in enumerate(morceaux):
                chunk_id = str(uuid.uuid4())
                nb_tokens = compter_tokens(morceau)

                chunk = ChunkDocument(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    type_doc="SFCR",
                    texte=morceau,
                    page_debut=pages_nums[0] if pages_nums else None,
                    page_fin=pages_nums[-1] if pages_nums else None,
                    section=section_id,
                    position_dans_doc=position_globale + i,
                    nb_tokens=nb_tokens,
                )
                tous_chunks.append(chunk)

            position_globale += len(morceaux)
            logger.debug(
                f"Section {section_lettre} : {len(morceaux)} chunks "
                f"({compter_tokens(texte_section)} tokens)"
            )

        return tous_chunks

    def _assembler_texte_section(
        self, doc_parse: DocumentParse, pages_nums: list[int]
    ) -> str:
        """
        Assemble le texte propre des pages d'une section.
        Exclut les pages de tableaux QRT pour le chunking narratif.
        """
        morceaux = []
        pages_dict = {p.numero: p for p in doc_parse.pages}

        for num in pages_nums:
            page = pages_dict.get(num)
            if page and not page.contient_tableau_qrt:
                texte = page.texte_propre.strip()
                if texte:
                    morceaux.append(texte)

        return "\n\n".join(morceaux)

    # ──────────────────────────────────────────────────────────────────────
    # Chunking page par page (mode dégradé)
    # ──────────────────────────────────────────────────────────────────────

    def _chunker_par_pages(
        self, doc_parse: DocumentParse, doc_id: str
    ) -> list[ChunkDocument]:
        """
        Fallback : chunke le texte page par page sans distinction de section.
        Utilisé quand la détection des sections A-E a échoué.
        """
        tous_chunks: list[ChunkDocument] = []
        position = 0

        # Assembler tout le texte narratif (hors tableaux)
        texte_complet = "\n\n".join(
            p.texte_propre
            for p in doc_parse.pages
            if not p.contient_tableau_qrt and p.texte_propre.strip()
        )

        morceaux = self._splitter.split_text(texte_complet)

        # Estimation de la page pour chaque chunk (approximative)
        tokens_par_page = doc_parse.nb_tokens_total / max(doc_parse.nb_pages, 1)

        for i, morceau in enumerate(morceaux):
            nb_tokens = compter_tokens(morceau)
            page_estimee = min(
                int(i * self.chunk_size / max(tokens_par_page, 1)) + 1,
                doc_parse.nb_pages
            )

            chunk = ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id=doc_id,
                type_doc="SFCR",
                texte=morceau,
                page_debut=page_estimee,
                page_fin=page_estimee,
                section="DOCUMENT_COMPLET",
                position_dans_doc=position + i,
                nb_tokens=nb_tokens,
            )
            tous_chunks.append(chunk)

        return tous_chunks

    # ──────────────────────────────────────────────────────────────────────
    # Chunking du référentiel réglementaire
    # ──────────────────────────────────────────────────────────────────────

    def chunker_referentiel(
        self,
        texte: str,
        source_id: str,
        article: str,
        id_exigence: str,
        position: int = 0,
    ) -> list[ChunkDocument]:
        """
        Chunke un texte réglementaire (article de loi, guideline EIOPA).

        Différence clé avec le SFCR :
        - Pas d'overlap (chaque article est une unité sémantique autonome)
        - Taille cible plus petite (les articles sont concis)
        - Section = id_exigence (traçabilité directe)

        Dans la pratique, la plupart des articles réglementaires tiennent
        en un seul chunk. Le splitter n'intervient que pour les articles
        longs (rares dans Solvabilité II).
        """
        splitter_reglementaire = RecursiveCharacterTextSplitter(
            chunk_size=350,       # Plus petit : les articles sont plus denses
            chunk_overlap=0,      # Pas d'overlap : articles autonomes
            length_function=compter_tokens,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

        morceaux = splitter_reglementaire.split_text(texte)
        chunks = []

        for i, morceau in enumerate(morceaux):
            chunk = ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id=source_id,
                type_doc="REGLEMENTAIRE",
                texte=morceau,
                page_debut=None,
                page_fin=None,
                section=id_exigence,  # Traçabilité directe exigence → chunk
                position_dans_doc=position + i,
                nb_tokens=compter_tokens(morceau),
            )
            chunks.append(chunk)

        return chunks

    # ──────────────────────────────────────────────────────────────────────
    # Validation des chunks
    # ──────────────────────────────────────────────────────────────────────

    def _valider_chunk(self, chunk: ChunkDocument) -> tuple[bool, str]:
        """
        Valide qu'un chunk est exploitable pour l'embedding et le retrieval.

        Critères de rejet (dans l'ordre de priorité) :
        1. Texte vide ou trop court (< CHUNK_MIN_TOKENS)
        2. Texte trop long (> CHUNK_MAX_TOKENS) — guard-rail splitter
        3. Ratio alphanumérique trop faible (bruit PDF résiduel)
        4. Texte composé uniquement de chiffres (tableau non filtré)
        5. Ligne unique (pas assez de contexte pour l'embedding)

        Returns:
            (True, "") si valide, (False, raison) sinon
        """
        texte = chunk.texte.strip()

        # 1. Texte vide
        if not texte:
            return False, "texte_vide"

        # 2. Trop court
        if chunk.nb_tokens < self.min_tokens:
            return False, f"trop_court_{chunk.nb_tokens}tokens"

        # 3. Trop long — le splitter ne devrait pas produire ça
        if chunk.nb_tokens > self.max_tokens:
            logger.warning(
                f"Chunk trop long : {chunk.nb_tokens} tokens "
                f"(max={self.max_tokens}) — chunk_id={chunk.chunk_id[:8]}"
            )
            return False, f"trop_long_{chunk.nb_tokens}tokens"

        # 4. Ratio alphanumérique — détecte le bruit PDF résiduel
        nb_alpha = sum(1 for c in texte if c.isalnum() or c.isspace())
        ratio_alpha = nb_alpha / len(texte)
        if ratio_alpha < 0.55:
            return False, "bruit_pdf_residuel"

        # 5. Texte composé uniquement de chiffres / symboles
        mots = texte.split()
        nb_mots_alpha = sum(1 for m in mots if any(c.isalpha() for c in m))
        if nb_mots_alpha < 3:
            return False, "pas_de_texte_alphabetique"

        # 6. Ligne unique sans contexte (souvent un titre seul)
        nb_lignes = len([l for l in texte.split("\n") if l.strip()])
        if nb_lignes == 1 and chunk.nb_tokens < 20:
            return False, "titre_sans_contexte"

        return True, ""

    # ──────────────────────────────────────────────────────────────────────
    # Préparation pour l'embedding
    # ──────────────────────────────────────────────────────────────────────

    def preparer_textes_embedding(
        self, chunks: list[ChunkDocument]
    ) -> list[str]:
        """
        Prépare les textes des chunks pour le modèle multilingual-e5-large.

        CRITIQUE : le modèle e5 requiert le préfixe "passage: " pour les
        documents à indexer. Sans ce préfixe, les performances baissent de
        15-20% sur les benchmarks BEIR (Thakur et al., 2021).

        Returns:
            Liste de textes formatés prêts pour model.encode()
        """
        return [
            preparer_requete_e5(chunk.texte, type_doc="passage")
            for chunk in chunks
        ]

    def preparer_texte_requete(self, texte_verification: str) -> str:
        """
        Prépare un texte de vérification (exigence) pour la recherche.
        Utilise le préfixe "query: " requis par multilingual-e5-large.
        """
        return preparer_requete_e5(texte_verification, type_doc="query")
