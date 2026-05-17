"""
models.py — Modèle de données RegWatch
=======================================
Définit les structures de données qui traversent tout le pipeline :
  Source → Pilier → Thème → Exigence → Chunk → Matching → Score

Toutes les entités sont des dataclasses Python avec validation intégrée.
La sérialisation JSON est native via asdict() pour l'export vers ChromaDB.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
from datetime import datetime
import re


# ---------------------------------------------------------------------------
# Enums métier
# ---------------------------------------------------------------------------

class NiveauObligation(str, Enum):
    """
    Hiérarchie des obligations issue du droit européen et des guidelines EIOPA.
    Détermine le poids de la non-conformité dans le scoring final.

    SHALL  → Obligation absolue (directive / règlement délégué)
             Non-conformité = écart critique → pénalité forte
    SHOULD → Recommandation forte (guidelines EIOPA, best practices)
             Non-conformité = point d'amélioration → pénalité modérée
    MAY    → Disposition optionnelle
             Non-conformité = observation → pénalité faible
    """
    SHALL = "SHALL"
    SHOULD = "SHOULD"
    MAY = "MAY"


class StatutConformite(str, Enum):
    """
    Résultat du matching RAG entre une exigence et le contenu du rapport.
    Calculé à partir du score de similarité cosinus + analyse LLM.
    """
    COUVERT = "COUVERT"          # score_similarite >= 0.75 ET validation LLM positive
    PARTIEL = "PARTIEL"          # 0.45 <= score_similarite < 0.75 OU réserve LLM
    ABSENT  = "ABSENT"           # score_similarite < 0.45 OU aucun passage retrouvé
    NON_APPLICABLE = "N/A"       # exigence non applicable au type d'entité analysée


class NiveauConformiteGlobal(str, Enum):
    """
    Notation finale du rapport selon l'échelle inspirée des évaluations EIOPA.
    A = Conforme    (score >= 0.80)
    B = Satisfaisant (score >= 0.65)
    C = Insuffisant  (score >= 0.45)
    D = Non conforme (score < 0.45)
    """
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class NumeroPilier(int, Enum):
    PILIER_1 = 1   # Exigences quantitatives (capital, valorisation)
    PILIER_2 = 2   # Gouvernance et contrôle interne
    PILIER_3 = 3   # Reporting et transparence


# ---------------------------------------------------------------------------
# Entités hiérarchiques du référentiel
# ---------------------------------------------------------------------------

@dataclass
class SourceReglementaire:
    """
    Représente un texte juridique source.
    Ex : Règlement délégué (UE) 2015/35 du 10 octobre 2014.
    """
    id: str                          # Ex : "REG_2015_35"
    code: str                        # Ex : "2015/35"
    intitule: str                    # Titre complet officiel
    type_source: str                 # "DIRECTIVE" | "REGLEMENT" | "GUIDELINE" | "ITS" | "RTS"
    date_publication: str            # Format ISO "YYYY-MM-DD"
    url_eur_lex: Optional[str] = None
    version: str = "1.0"

    def __post_init__(self):
        valid_types = {"DIRECTIVE", "REGLEMENT", "GUIDELINE", "ITS", "RTS"}
        if self.type_source not in valid_types:
            raise ValueError(f"type_source invalide : {self.type_source}. "
                             f"Valeurs acceptées : {valid_types}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Pilier:
    """
    Les 3 piliers structurants de Solvabilité II.
    La pondération reflète l'importance relative dans le scoring global.
    """
    id: str                          # Ex : "P1", "P2", "P3"
    numero: NumeroPilier
    intitule: str
    description: str
    poids_scoring: float             # Somme des 3 piliers = 1.0
    themes: list[str] = field(default_factory=list)  # ids des thèmes rattachés

    def __post_init__(self):
        if not (0 < self.poids_scoring <= 1):
            raise ValueError(f"poids_scoring doit être dans ]0, 1] : {self.poids_scoring}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Theme:
    """
    Thème réglementaire — unité de scoring intermédiaire.
    Un thème regroupe un ensemble d'exigences cohérentes sur un sujet donné.

    La pondération_scoring est relative au sein du pilier parent.
    Ex : Gouvernance (T04) pèse 0.30 dans le Pilier 2.
    """
    id: str                          # Ex : "T04"
    libelle: str                     # Ex : "Système de gouvernance"
    pilier_id: str                   # Référence vers Pilier
    poids_dans_pilier: float         # Poids relatif dans son pilier (somme = 1.0 par pilier)
    description: str = ""
    articles_reference: list[str] = field(default_factory=list)  # Ex : ["Art.258", "Art.259"]

    def __post_init__(self):
        if not (0 < self.poids_dans_pilier <= 1):
            raise ValueError(f"poids_dans_pilier invalide pour {self.id}: {self.poids_dans_pilier}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExigenceAtomique:
    """
    Unité élémentaire du référentiel — une obligation réglementaire précise et non décomposable.

    C'est la brique fondamentale du système de conformité.
    Chaque exigence est vérifiable indépendamment dans un rapport SFCR.

    Convention d'identifiant : SII-{PILIER}-{SOURCE_CODE}-{ARTICLE}-§{PARAGRAPHE}
    Ex : "SII-P3-REG2015-Art293-§2a"

    Le texte_verification est la question posée au LLM lors du matching :
    "Le rapport mentionne-t-il explicitement [texte_verification] ?"
    """
    id_exigence: str                         # Identifiant unique (convention ci-dessus)
    theme_id: str                            # Référence vers Theme
    source_id: str                           # Référence vers SourceReglementaire
    article: str                             # Ex : "Art. 293 §2 al. a)"
    niveau_obligation: NiveauObligation
    texte_original: str                      # Verbatim réglementaire (FR ou EN)
    texte_normalise: str                     # Version nettoyée pour l'embedding
    texte_verification: str                  # Question pour le LLM (ex: "Le rapport décrit-il...")
    mots_cles: list[str] = field(default_factory=list)  # Pour filtrage lexical complémentaire
    notes_interpretatives: str = ""          # Précisions de l'analyste
    entites_concernees: list[str] = field(  # Ex : ["entreprise_assurance", "groupe"]
        default_factory=lambda: ["entreprise_assurance"]
    )
    actif: bool = True                       # Permet de désactiver sans supprimer

    def __post_init__(self):
        self._valider_id()
        self._valider_textes()

    def _valider_id(self):
        """
        Vérifie que l'identifiant respecte la convention de nommage.
        Préfixe accepté : Art (articles réglementaires) ou GL (guidelines EIOPA).
        """
        pattern = r"^SII-P[123]-[A-Z0-9_]+-(?:Art|GL)\d+.*$"
        if not re.match(pattern, self.id_exigence):
            raise ValueError(
                f"Format d'id invalide : '{self.id_exigence}'\n"
                f"Format attendu : SII-P{{1|2|3}}-{{SOURCE}}-{{Art|GL}}{{N}}-§{{ref}}\n"
                f"Exemples       : SII-P3-REG2015-Art293-§2a\n"
                f"                 SII-P3-EIOPA-GL01-§1"
            )

    def _valider_textes(self):
        if len(self.texte_original) < 10:
            raise ValueError(f"texte_original trop court pour {self.id_exigence}")
        if len(self.texte_verification) < 10:
            raise ValueError(f"texte_verification trop court pour {self.id_exigence}")

    @property
    def poids_obligation(self) -> float:
        """
        Poids numérique de l'obligation pour le scoring.
        SHALL = 1.0 (non-conformité critique)
        SHOULD = 0.6 (non-conformité importante)
        MAY = 0.3 (non-conformité mineure)
        """
        poids = {
            NiveauObligation.SHALL: 1.0,
            NiveauObligation.SHOULD: 0.6,
            NiveauObligation.MAY: 0.3,
        }
        return poids[self.niveau_obligation]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["niveau_obligation"] = self.niveau_obligation.value
        d["poids_obligation"] = self.poids_obligation
        return d

    def to_chroma_metadata(self) -> dict:
        """
        Format de métadonnées pour ChromaDB.
        ChromaDB n'accepte que str, int, float, bool dans les métadonnées.
        Les listes sont sérialisées en string JSON.
        """
        import json
        return {
            "id_exigence": self.id_exigence,
            "theme_id": self.theme_id,
            "source_id": self.source_id,
            "article": self.article,
            "niveau_obligation": self.niveau_obligation.value,
            "poids_obligation": self.poids_obligation,
            "mots_cles": json.dumps(self.mots_cles, ensure_ascii=False),
            "entites_concernees": json.dumps(self.entites_concernees, ensure_ascii=False),
            "actif": self.actif,
        }


# ---------------------------------------------------------------------------
# Entités liées aux documents analysés (SFCR)
# ---------------------------------------------------------------------------

@dataclass
class DocumentSFCR:
    """
    Métadonnées d'un rapport SFCR importé dans le système.
    """
    id: str                                  # UUID généré à l'import
    nom_fichier: str
    entreprise: str
    annee_rapport: int
    date_import: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    nb_pages: Optional[int] = None
    langue: str = "fr"
    nb_chunks: int = 0
    hash_sha256: Optional[str] = None       # Détection de doublons

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ChunkDocument:
    """
    Fragment de texte issu d'un document (SFCR ou réglementaire).
    Unité élémentaire stockée dans ChromaDB avec son embedding.

    chunk_size cible : 400-600 tokens avec overlap de 80 tokens.
    """
    chunk_id: str                            # UUID
    doc_id: str                              # Référence vers DocumentSFCR ou SourceReglementaire
    type_doc: str                            # "SFCR" | "REGLEMENTAIRE"
    texte: str
    page_debut: Optional[int] = None
    page_fin: Optional[int] = None
    section: Optional[str] = None           # Ex : "A. Activité et résultats"
    position_dans_doc: int = 0              # Index du chunk dans le document
    nb_tokens: int = 0
    # L'embedding n'est PAS stocké ici — il vit dans ChromaDB
    # On stocke uniquement les métadonnées pour la traçabilité

    def to_chroma_metadata(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "type_doc": self.type_doc,
            "page_debut": self.page_debut or 0,
            "page_fin": self.page_fin or 0,
            "section": self.section or "",
            "position_dans_doc": self.position_dans_doc,
            "nb_tokens": self.nb_tokens,
        }


# ---------------------------------------------------------------------------
# Entités de résultat (matching + scoring)
# ---------------------------------------------------------------------------

@dataclass
class ResultatMatching:
    """
    Résultat du matching RAG entre une exigence et les passages du SFCR.
    Produit par le moteur RAG (Phase 4).

    score_similarite : similarité cosinus entre embedding exigence et chunk SFCR [0, 1]
    score_llm        : score attribué par le LLM après lecture du passage [0, 1]
    score_final      : combinaison pondérée (0.4 * cosinus + 0.6 * llm)
    """
    id: str                                  # UUID
    id_exigence: str
    chunk_sfcr_id: str
    score_similarite: float                  # Similarité cosinus [0, 1]
    score_llm: float                         # Évaluation LLM [0, 1]
    score_final: float                       # Score combiné [0, 1]
    statut: StatutConformite
    passage_trouve: str                      # Extrait du SFCR utilisé
    justification_llm: str                   # Explication générée par le LLM
    date_analyse: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def __post_init__(self):
        for nom, val in [("score_similarite", self.score_similarite),
                         ("score_llm", self.score_llm),
                         ("score_final", self.score_final)]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{nom} doit être dans [0, 1] : {val}")

    @classmethod
    def calculer_statut(cls, score_final: float) -> StatutConformite:
        """Règle de classification du statut à partir du score final."""
        if score_final >= 0.75:
            return StatutConformite.COUVERT
        elif score_final >= 0.45:
            return StatutConformite.PARTIEL
        else:
            return StatutConformite.ABSENT

    def to_dict(self) -> dict:
        d = asdict(self)
        d["statut"] = self.statut.value
        return d


@dataclass
class ScoreTheme:
    """Score de conformité agrégé pour un thème réglementaire."""
    theme_id: str
    libelle_theme: str
    nb_exigences_total: int
    nb_couvertes: int
    nb_partielles: int
    nb_absentes: int
    score_brut: float                        # Moyenne pondérée des scores d'exigences
    score_pondere: float                     # score_brut * poids_dans_pilier
    exigences_critiques_absentes: list[str] = field(default_factory=list)

    @property
    def taux_couverture(self) -> float:
        if self.nb_exigences_total == 0:
            return 0.0
        return self.nb_couvertes / self.nb_exigences_total

    def to_dict(self) -> dict:
        return {**asdict(self), "taux_couverture": self.taux_couverture}


@dataclass
class RapportConformite:
    """
    Résultat final complet d'une analyse de conformité.
    C'est l'objet retourné à l'interface Streamlit.
    """
    id_analyse: str
    document_id: str
    entreprise: str
    annee_rapport: int
    date_analyse: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    scores_themes: list[ScoreTheme] = field(default_factory=list)
    score_pilier_1: float = 0.0
    score_pilier_2: float = 0.0
    score_pilier_3: float = 0.0
    score_global: float = 0.0
    niveau_conformite: NiveauConformiteGlobal = NiveauConformiteGlobal.D
    nb_exigences_analysees: int = 0
    nb_exigences_couvertes: int = 0
    nb_exigences_partielles: int = 0
    nb_exigences_absentes: int = 0
    recommandations: list[str] = field(default_factory=list)

    @classmethod
    def niveau_depuis_score(cls, score: float) -> NiveauConformiteGlobal:
        if score >= 0.80:
            return NiveauConformiteGlobal.A
        elif score >= 0.65:
            return NiveauConformiteGlobal.B
        elif score >= 0.45:
            return NiveauConformiteGlobal.C
        else:
            return NiveauConformiteGlobal.D

    def to_dict(self) -> dict:
        d = asdict(self)
        d["niveau_conformite"] = self.niveau_conformite.value
        d["scores_themes"] = [s.to_dict() for s in self.scores_themes]
        return d
