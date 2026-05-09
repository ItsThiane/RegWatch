"""
pdf_parser.py — Parseur PDF SFCR avec PyMuPDF
==============================================
Responsabilités :
  1. Extraire le texte brut page par page depuis un PDF SFCR
  2. Identifier la structure hiérarchique (sections A-E, sous-sections)
  3. Séparer le texte narratif des tableaux QRT
  4. Extraire les métadonnées du document (titre, date, entreprise)
  5. Produire un objet DocumentParse structuré prêt pour le chunker

Pourquoi PyMuPDF (fitz) plutôt que pdfplumber ou pdfminer ?
  - Vitesse : 10-50x plus rapide que pdfminer sur les gros PDFs
  - Précision : meilleure extraction de la structure (blocs, polices)
  - Robustesse : gère les PDFs multi-colonnes, les OCR, les images
  - Accès aux métadonnées PDF natives (auteur, date de création)
  - Détection des polices : permet de distinguer titres et corps de texte
    sur la base de la taille et du style de police

Architecture d'extraction :
  PDF → pages → blocs de texte → classification → sections → texte propre
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.ingestion.text_utils import (
    nettoyer_texte_pdf,
    est_bruit,
    est_tableau_qrt,
    extraire_section_sfcr,
    statistiques_texte,
    compter_tokens,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures de données intermédiaires du parseur
# ---------------------------------------------------------------------------

@dataclass
class BlocTexte:
    """
    Unité élémentaire extraite d'une page PDF.
    Un bloc correspond à un rectangle de texte cohérent dans PyMuPDF.
    """
    page: int
    texte: str
    x0: float          # Coordonnées du rectangle (points PDF)
    y0: float
    x1: float
    y1: float
    taille_police: float      # Taille dominante dans le bloc
    est_gras: bool            # Police en gras → probable titre
    est_titre: bool = False   # Classification finale
    est_tableau: bool = False
    est_bruit: bool = False
    section_sfcr: Optional[str] = None  # "A", "B", "C", "D", "E" ou None


@dataclass
class PageParsee:
    """Résultat de l'analyse d'une page PDF."""
    numero: int               # 1-indexé
    texte_brut: str           # Texte complet de la page (avant nettoyage)
    texte_propre: str         # Texte après nettoyage
    blocs: list[BlocTexte] = field(default_factory=list)
    section_principale: Optional[str] = None
    contient_tableau_qrt: bool = False
    nb_tokens: int = 0


@dataclass
class DocumentParse:
    """
    Résultat complet de l'analyse d'un PDF SFCR.
    C'est l'input du chunker (sfcr_chunker.py).
    """
    chemin_fichier: str
    hash_sha256: str
    nb_pages: int
    pages: list[PageParsee] = field(default_factory=list)

    # Métadonnées extraites du PDF ou inférées du contenu
    entreprise: Optional[str] = None
    annee_rapport: Optional[int] = None
    langue: str = "fr"
    titre_document: Optional[str] = None

    # Statistiques d'extraction
    nb_pages_bruit: int = 0
    nb_pages_tableau: int = 0
    nb_tokens_total: int = 0
    sections_detectees: list[str] = field(default_factory=list)

    # Index sections → pages (pour le retrieval contextuel)
    index_sections: dict[str, list[int]] = field(default_factory=dict)

    @property
    def texte_complet(self) -> str:
        """Concaténation du texte propre de toutes les pages."""
        return "\n\n".join(
            p.texte_propre for p in self.pages if p.texte_propre.strip()
        )


# ---------------------------------------------------------------------------
# Parseur principal
# ---------------------------------------------------------------------------

class SFCRParser:
    """
    Parseur PDF spécialisé pour les rapports SFCR Solvabilité II.

    Paramètres de configuration :
        seuil_titre_pts: Taille minimale de police (en points) pour
                         considérer un bloc comme un titre (défaut: 11.5)
        min_tokens_bloc: Nombre minimum de tokens pour qu'un bloc
                         soit conservé (défaut: 5)
        extraire_tableaux: Si True, extrait aussi le contenu des tableaux
                           (défaut: True, mais marqué separément)
    """

    # Patterns de détection des sections SFCR dans les titres
    _SECTIONS_PATTERNS = {
        "A": re.compile(
            r"^[A-Z]?\.?\s*(?:A\b|section\s+a\b|activit[eé]s?\s+et\s+r[eé]sultats)",
            re.IGNORECASE
        ),
        "B": re.compile(
            r"^[A-Z]?\.?\s*(?:B\b|section\s+b\b|syst[eè]me\s+de\s+gouvernance)",
            re.IGNORECASE
        ),
        "C": re.compile(
            r"^[A-Z]?\.?\s*(?:C\b|section\s+c\b|profil\s+de\s+risque)",
            re.IGNORECASE
        ),
        "D": re.compile(
            r"^[A-Z]?\.?\s*(?:D\b|section\s+d\b|valorisation\s+aux\s+fins)",
            re.IGNORECASE
        ),
        "E": re.compile(
            r"^[A-Z]?\.?\s*(?:E\b|section\s+e\b|gestion\s+du\s+capital)",
            re.IGNORECASE
        ),
    }

    # Pattern pour extraire l'année du rapport
    _PATTERN_ANNEE = re.compile(
        r"(?:exercice|rapport|année|au)\s+(?:clos|clôtur[eé]|de\s+référence)?"
        r"\s*(?:le\s+)?(?:31\s+d[eé]cembre\s+)?(\d{4})",
        re.IGNORECASE
    )

    def __init__(
        self,
        seuil_titre_pts: float = 11.5,
        min_tokens_bloc: int = 5,
        extraire_tableaux: bool = True,
    ):
        self.seuil_titre_pts = seuil_titre_pts
        self.min_tokens_bloc = min_tokens_bloc
        self.extraire_tableaux = extraire_tableaux

    # ──────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────────────────────────────

    def parser(self, chemin_pdf: str | Path) -> DocumentParse:
        """
        Parse un PDF SFCR complet et retourne un DocumentParse structuré.

        Args:
            chemin_pdf: Chemin vers le fichier PDF

        Returns:
            DocumentParse avec toutes les pages, blocs et métadonnées

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ValueError: Si le fichier n'est pas un PDF valide
            RuntimeError: Si l'extraction échoue sur trop de pages
        """
        chemin = Path(chemin_pdf)
        if not chemin.exists():
            raise FileNotFoundError(f"Fichier PDF introuvable : {chemin}")
        if chemin.suffix.lower() != ".pdf":
            raise ValueError(f"Format attendu : .pdf — reçu : {chemin.suffix}")

        logger.info(f"Début du parsing : {chemin.name}")

        # Calcul du hash pour détection de doublons
        hash_sha256 = self._calculer_hash(chemin)

        try:
            doc = fitz.open(str(chemin))
        except Exception as e:
            raise ValueError(f"Impossible d'ouvrir le PDF : {e}")

        nb_pages = len(doc)
        logger.info(f"  PDF ouvert : {nb_pages} pages")

        # Extraction des métadonnées PDF natives
        meta_pdf = doc.metadata or {}

        # Initialisation du résultat
        resultat = DocumentParse(
            chemin_fichier=str(chemin),
            hash_sha256=hash_sha256,
            nb_pages=nb_pages,
            titre_document=meta_pdf.get("title", ""),
        )

        # Parse page par page
        section_courante = None
        nb_echecs = 0

        for num_page in range(nb_pages):
            try:
                page_fitz = doc[num_page]
                page_parsee = self._parser_page(page_fitz, num_page + 1)

                # Propager la section depuis la page précédente si non détectée
                if page_parsee.section_principale:
                    section_courante = page_parsee.section_principale
                elif section_courante:
                    page_parsee.section_principale = section_courante

                # Mettre à jour l'index des sections
                if page_parsee.section_principale:
                    s = page_parsee.section_principale
                    if s not in resultat.index_sections:
                        resultat.index_sections[s] = []
                    resultat.index_sections[s].append(num_page + 1)

                # Comptabilisation
                if page_parsee.contient_tableau_qrt:
                    resultat.nb_pages_tableau += 1

                resultat.pages.append(page_parsee)
                resultat.nb_tokens_total += page_parsee.nb_tokens

                if (num_page + 1) % 20 == 0:
                    logger.debug(f"  Parsing : {num_page + 1}/{nb_pages} pages")

            except Exception as e:
                logger.warning(f"  Erreur page {num_page + 1}: {e}")
                nb_echecs += 1
                if nb_echecs > nb_pages * 0.1:  # Plus de 10% d'échecs
                    raise RuntimeError(
                        f"Trop d'erreurs d'extraction ({nb_echecs}/{nb_pages})"
                    )

        doc.close()

        # Post-traitement : inférer les métadonnées manquantes
        self._inferer_metadonnees(resultat)

        # Finaliser les sections détectées
        resultat.sections_detectees = sorted(resultat.index_sections.keys())
        resultat.nb_pages_bruit = sum(
            1 for p in resultat.pages
            if p.texte_propre.strip() == ""
        )

        logger.info(
            f"Parsing terminé : {nb_pages} pages, "
            f"{resultat.nb_tokens_total} tokens, "
            f"sections {resultat.sections_detectees}, "
            f"{resultat.nb_pages_bruit} pages vides/bruit"
        )

        return resultat

    # ──────────────────────────────────────────────────────────────────────
    # Parsing d'une page
    # ──────────────────────────────────────────────────────────────────────

    def _parser_page(self, page: fitz.Page, numero: int) -> PageParsee:
        """
        Analyse une page PDF et extrait ses blocs de texte classifiés.

        PyMuPDF expose les blocs de texte via get_text("dict") qui retourne
        la structure complète : blocs → lignes → spans avec attributs de police.
        On utilise cette structure pour classifier chaque bloc (titre, tableau,
        corps de texte, bruit) avant de concaténer le texte propre.
        """
        # Extraction du texte avec informations de mise en page
        try:
            dict_page = page.get_text("dict", flags=fitz.TEXT_PRESERVE_LIGATURES)
        except Exception:
            # Fallback : extraction texte simple
            texte_brut = page.get_text("text")
            texte_propre = nettoyer_texte_pdf(texte_brut)
            return PageParsee(
                numero=numero,
                texte_brut=texte_brut,
                texte_propre=texte_propre,
                nb_tokens=compter_tokens(texte_propre),
            )

        blocs = []
        blocs_texte_propre = []
        section_page = None
        contient_qrt = False

        for bloc in dict_page.get("blocks", []):
            # On ne traite que les blocs texte (type 0), pas les images (type 1)
            if bloc.get("type") != 0:
                continue

            texte_bloc, taille_police, est_gras = self._extraire_infos_bloc(bloc)

            if not texte_bloc.strip():
                continue

            # Classification du bloc
            bloc_classe = BlocTexte(
                page=numero,
                texte=texte_bloc,
                x0=bloc["bbox"][0],
                y0=bloc["bbox"][1],
                x1=bloc["bbox"][2],
                y1=bloc["bbox"][3],
                taille_police=taille_police,
                est_gras=est_gras,
            )

            # Est-ce du bruit ?
            if est_bruit(texte_bloc):
                bloc_classe.est_bruit = True
                blocs.append(bloc_classe)
                continue

            # Est-ce un tableau QRT ?
            if est_tableau_qrt(texte_bloc):
                bloc_classe.est_tableau = True
                contient_qrt = True
                if self.extraire_tableaux:
                    blocs_texte_propre.append(f"[TABLEAU QRT]\n{texte_bloc}")
                blocs.append(bloc_classe)
                continue

            # Est-ce un titre de section ?
            est_titre = (
                taille_police >= self.seuil_titre_pts
                or est_gras
                or self._est_titre_numerote(texte_bloc)
            )
            bloc_classe.est_titre = est_titre

            # Détecter la section SFCR depuis ce titre
            if est_titre:
                section_detectee = self._detecter_section(texte_bloc)
                if section_detectee:
                    section_page = section_detectee
                    bloc_classe.section_sfcr = section_detectee

            # Nettoyer et conserver
            texte_propre_bloc = nettoyer_texte_pdf(texte_bloc)
            tokens_bloc = compter_tokens(texte_propre_bloc)

            if tokens_bloc >= self.min_tokens_bloc:
                bloc_classe.section_sfcr = bloc_classe.section_sfcr or section_page
                blocs_texte_propre.append(texte_propre_bloc)
                blocs.append(bloc_classe)

        # Assemblage du texte propre de la page
        texte_propre_page = "\n\n".join(blocs_texte_propre)
        texte_propre_page = nettoyer_texte_pdf(texte_propre_page, agressif=True)

        # Section de la page (inférée depuis les blocs)
        if not section_page:
            section_page = extraire_section_sfcr(texte_propre_page)

        return PageParsee(
            numero=numero,
            texte_brut=page.get_text("text"),
            texte_propre=texte_propre_page,
            blocs=blocs,
            section_principale=section_page,
            contient_tableau_qrt=contient_qrt,
            nb_tokens=compter_tokens(texte_propre_page),
        )

    def _extraire_infos_bloc(
        self, bloc: dict
    ) -> tuple[str, float, bool]:
        """
        Extrait le texte, la taille de police dominante et le style gras
        d'un bloc PyMuPDF.

        PyMuPDF structure : bloc → lines → spans
        Chaque span a : text, size (taille police), flags (0b=bold, 1b=italic)
        """
        morceaux_texte = []
        tailles = []
        est_gras = False

        for ligne in bloc.get("lines", []):
            morceaux_ligne = []
            for span in ligne.get("spans", []):
                texte_span = span.get("text", "")
                if texte_span.strip():
                    morceaux_ligne.append(texte_span)
                    tailles.append(span.get("size", 10))
                    # flags : bit 4 = bold (16), bit 1 = italic (2)
                    if span.get("flags", 0) & 16:
                        est_gras = True
            if morceaux_ligne:
                morceaux_texte.append("".join(morceaux_ligne))

        texte = "\n".join(morceaux_texte)
        taille_dominante = (
            sorted(tailles, reverse=True)[len(tailles) // 4]  # Quartile 75%
            if tailles else 10.0
        )

        return texte, taille_dominante, est_gras

    def _est_titre_numerote(self, texte: str) -> bool:
        """
        Détecte les titres numérotés type "1.2.3 Titre" ou "A.1 Titre".
        Ces patterns sont communs dans les SFCR même sans mise en gras.
        """
        pattern = re.compile(
            r"^(?:[A-E]\.?\d*\.?\d*|[IVX]+\.|\d+\.\d*\.?\d*)\s+[A-ZÁÀÂÉÈÊ]"
        )
        return bool(pattern.match(texte.strip()))

    def _detecter_section(self, texte_titre: str) -> Optional[str]:
        """
        Tente de détecter la section SFCR (A à E) depuis un titre de section.
        Retourne la lettre de section ou None.
        """
        texte_norm = texte_titre.strip()
        for section, pattern in self._SECTIONS_PATTERNS.items():
            if pattern.match(texte_norm):
                return section
        # Fallback : détection dans le texte complet
        return extraire_section_sfcr(texte_titre)

    # ──────────────────────────────────────────────────────────────────────
    # Inférence de métadonnées
    # ──────────────────────────────────────────────────────────────────────

    def _inferer_metadonnees(self, doc: DocumentParse) -> None:
        """
        Infère les métadonnées manquantes depuis le contenu du document.
        Analyse les premières pages (couverture, sommaire) pour extraire
        le nom de l'entreprise et l'année du rapport.
        """
        # Chercher dans les 5 premières pages
        texte_debut = "\n".join(
            p.texte_propre for p in doc.pages[:5]
        )

        # Extraction de l'année
        if not doc.annee_rapport:
            match_annee = self._PATTERN_ANNEE.search(texte_debut)
            if match_annee:
                doc.annee_rapport = int(match_annee.group(1))
            else:
                # Chercher une année seule dans les 500 premiers caractères
                annees = re.findall(r"\b(20\d{2})\b", texte_debut[:500])
                if annees:
                    doc.annee_rapport = int(
                        max(set(annees), key=annees.count)
                    )

        # Le nom d'entreprise doit être fourni explicitement (trop variable)
        # On l'infère depuis le nom de fichier si non renseigné
        if not doc.entreprise:
            nom_fichier = Path(doc.chemin_fichier).stem
            # Nettoyer le nom de fichier (supprimer SFCR, année, etc.)
            nom_propre = re.sub(
                r"(?i)sfcr|rapport|solvabilit[eé]|20\d{2}|[-_]",
                " ", nom_fichier
            ).strip()
            doc.entreprise = nom_propre if nom_propre else "Inconnu"

        logger.debug(
            f"Métadonnées inférées : entreprise='{doc.entreprise}', "
            f"année={doc.annee_rapport}, langue={doc.langue}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _calculer_hash(chemin: Path) -> str:
        """Calcule le SHA-256 du fichier PDF pour la détection de doublons."""
        sha256 = hashlib.sha256()
        with open(chemin, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def rapport_parsing(self, doc: DocumentParse) -> str:
        """
        Génère un rapport lisible sur le résultat du parsing.
        Utile pour le diagnostic et la validation manuelle.
        """
        lignes = [
            "=" * 60,
            "RAPPORT DE PARSING PDF",
            "=" * 60,
            f"Fichier       : {Path(doc.chemin_fichier).name}",
            f"Entreprise    : {doc.entreprise or 'Non détectée'}",
            f"Année         : {doc.annee_rapport or 'Non détectée'}",
            f"Pages totales : {doc.nb_pages}",
            f"Pages bruit   : {doc.nb_pages_bruit}",
            f"Pages QRT     : {doc.nb_pages_tableau}",
            f"Tokens totaux : {doc.nb_tokens_total:,}",
            f"Sections      : {', '.join(doc.sections_detectees) or 'Non détectées'}",
            "",
            "Répartition par section :",
        ]
        for section, pages in sorted(doc.index_sections.items()):
            lignes.append(
                f"  Section {section} : pages {pages[0]}–{pages[-1]} "
                f"({len(pages)} pages)"
            )

        lignes.append("")
        lignes.append("Aperçu du texte (50 premiers tokens) :")
        apercu = doc.texte_complet[:300].replace('\n', ' ')
        lignes.append(f"  {apercu}...")

        return "\n".join(lignes)
