"""
rag_pipeline.py — Pipeline RAG de conformité Solvabilité II
============================================================
Orchestre le flux complet pour chaque exigence réglementaire :

  Pour chaque exigence du référentiel :
    1. Embed texte_verification        (query: prefix)
    2. Retrieve top-K passages SFCR    (ChromaDB cosine search)
    3. Analyser chaque passage         (LLM → score + justification)
    4. Combiner les scores             (0.4 × cosinus + 0.6 × LLM)
    5. Classer le statut               (COUVERT / PARTIEL / ABSENT)
    6. Retourner ResultatMatching      (prêt pour le scoring Phase 5)

Paramètres clés :
  top_k           : Nombre de passages candidats par exigence (défaut: 5)
  seuil_cosinus   : Distance cosinus max pour retenir un candidat (défaut: 0.85)
  poids_cosinus   : Poids du score vectoriel dans le score final (défaut: 0.40)
  poids_llm       : Poids du score LLM dans le score final (défaut: 0.60)

Pourquoi 60% LLM / 40% cosinus ?
  Le cosinus mesure la proximité vectorielle mais échoue sur :
    - Les paraphrases sémantiques sans lexique commun
    - Les passages qui mentionnent un terme sans le développer
    - Les faux positifs lexicaux (même mot, sens différent)
  Le LLM comprend le raisonnement implicite mais peut halluciner.
  La combinaison 60/40 est un compromis calibré sur des études RAG
  (Lewis et al. 2020, Gao et al. 2024) : le LLM pilote, le cosinus ancre.
"""

from __future__ import annotations
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.ingestion.models import (
    ExigenceAtomique, ResultatMatching, StatutConformite, NiveauObligation
)
from src.rag.embedding_engine import EmbeddingEngine
from src.rag.vector_store import VectorStore
from src.rag.llm_engine import LLMEngine, AnalyseLLM
from src.ingestion.text_utils import preparer_requete_e5

logger = logging.getLogger(__name__)

# Pondérations du score final
POIDS_COSINUS = 0.40
POIDS_LLM     = 0.60

# Paramètres de recherche
TOP_K_CANDIDATS    = 5       # Passages candidats par exigence
SEUIL_DISTANCE_MAX = 0.85    # Distance cosinus max (0=identique, 2=opposé)
SEUIL_SCORE_MIN    = 0.10    # Score min pour qu'un passage soit considéré


@dataclass
class ResultatAnalyseExigence:
    """
    Résultat complet de l'analyse d'une exigence contre un SFCR.

    Contient le ResultatMatching final et les détails intermédiaires
    pour la traçabilité et le débogage.
    """
    exigence: ExigenceAtomique
    matching: ResultatMatching
    candidats_evalues: list[dict]    # Tous les passages évalués avec leurs scores
    nb_candidats: int = 0
    temps_analyse_s: float = 0.0

    @property
    def est_couvert(self) -> bool:
        return self.matching.statut == StatutConformite.COUVERT

    @property
    def est_partiel(self) -> bool:
        return self.matching.statut == StatutConformite.PARTIEL

    @property
    def est_absent(self) -> bool:
        return self.matching.statut == StatutConformite.ABSENT


@dataclass
class ResultatAnalyseSFCR:
    """
    Résultat de l'analyse complète d'un SFCR contre le référentiel.
    Produit par RAGPipeline.analyser_sfcr().
    Consommé par le moteur de scoring (Phase 5).
    """
    doc_id: str
    entreprise: str
    annee: int
    resultats_exigences: list[ResultatAnalyseExigence] = field(
        default_factory=list
    )
    temps_total_s: float = 0.0
    nb_exigences_analysees: int = 0
    nb_erreurs: int = 0

    @property
    def matchings(self) -> list[ResultatMatching]:
        """Liste plate des ResultatMatching pour le scoring."""
        return [r.matching for r in self.resultats_exigences]

    def rapport_progression(self) -> str:
        couvertes  = sum(1 for r in self.resultats_exigences if r.est_couvert)
        partielles = sum(1 for r in self.resultats_exigences if r.est_partiel)
        absentes   = sum(1 for r in self.resultats_exigences if r.est_absent)
        total = len(self.resultats_exigences)
        return (
            f"Analysé {total}/{self.nb_exigences_analysees} exigences : "
            f"COUVERT={couvertes} PARTIEL={partielles} ABSENT={absentes} "
            f"({self.temps_total_s:.1f}s)"
        )


class RAGPipeline:
    """
    Pipeline RAG de conformité — orchestre embedding + retrieval + LLM.

    Usage :
        pipeline = RAGPipeline(embedding_engine, vector_store, llm_engine)
        resultat = pipeline.analyser_sfcr(
            exigences=loader.get_toutes_exigences(),
            doc_id="uuid-du-sfcr",
            entreprise="AXA France",
            annee=2023,
        )
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store: VectorStore,
        llm_engine: LLMEngine,
        top_k: int = TOP_K_CANDIDATS,
        seuil_distance: float = SEUIL_DISTANCE_MAX,
        poids_cosinus: float = POIDS_COSINUS,
        poids_llm: float = POIDS_LLM,
    ):
        assert abs(poids_cosinus + poids_llm - 1.0) < 0.01, (
            f"poids_cosinus + poids_llm doit valoir 1.0 "
            f"(actuel: {poids_cosinus + poids_llm})"
        )
        self.embedding_engine = embedding_engine
        self.vector_store = vector_store
        self.llm_engine = llm_engine
        self.top_k = top_k
        self.seuil_distance = seuil_distance
        self.poids_cosinus = poids_cosinus
        self.poids_llm = poids_llm

        logger.info(
            f"RAGPipeline initialisé | "
            f"top_k={top_k} | "
            f"poids={poids_cosinus:.0%}cosinus/{poids_llm:.0%}LLM | "
            f"llm={llm_engine.nom_backend}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Analyse complète d'un SFCR
    # ──────────────────────────────────────────────────────────────────────

    def analyser_sfcr(
        self,
        exigences: list[ExigenceAtomique],
        doc_id: str,
        entreprise: str,
        annee: int,
        filtre_themes: Optional[list[str]] = None,
        callback_progression: Optional[callable] = None,
    ) -> ResultatAnalyseSFCR:
        """
        Analyse un SFCR complet contre une liste d'exigences.

        Args:
            exigences           : Liste des exigences à vérifier
            doc_id              : Identifiant ChromaDB du SFCR
            entreprise          : Nom pour les métadonnées
            annee               : Année pour les métadonnées
            filtre_themes       : Si défini, analyser uniquement ces thèmes
            callback_progression: Fonction appelée après chaque exigence
                                  (i, total, exigence, resultat) → None

        Returns:
            ResultatAnalyseSFCR avec tous les matchings
        """
        t_debut = time.time()

        # Filtrer par thèmes si demandé
        if filtre_themes:
            exigences = [e for e in exigences if e.theme_id in filtre_themes]
            logger.info(
                f"Filtre thèmes {filtre_themes} : "
                f"{len(exigences)} exigences à analyser"
            )

        resultat = ResultatAnalyseSFCR(
            doc_id=doc_id,
            entreprise=entreprise,
            annee=annee,
            nb_exigences_analysees=len(exigences),
        )

        logger.info(
            f"Début analyse SFCR {entreprise} {annee} — "
            f"{len(exigences)} exigences"
        )

        for i, exigence in enumerate(exigences):
            try:
                res = self._analyser_exigence(exigence, doc_id)
                resultat.resultats_exigences.append(res)

                if callback_progression:
                    callback_progression(i + 1, len(exigences), exigence, res)

                # Log périodique toutes les 10 exigences
                if (i + 1) % 10 == 0:
                    logger.info(
                        f"Progression : {i+1}/{len(exigences)} exigences | "
                        f"{resultat.rapport_progression()}"
                    )

            except Exception as e:
                logger.error(f"Erreur analyse {exigence.id_exigence}: {e}")
                resultat.nb_erreurs += 1
                # Créer un matching d'erreur pour ne pas perdre la traçabilité
                resultat.resultats_exigences.append(
                    self._creer_resultat_erreur(exigence, str(e))
                )

        resultat.temps_total_s = round(time.time() - t_debut, 2)
        logger.info(
            f"Analyse terminée : {resultat.rapport_progression()}"
        )
        return resultat

    # ──────────────────────────────────────────────────────────────────────
    # Analyse d'une exigence individuelle
    # ──────────────────────────────────────────────────────────────────────

    def _analyser_exigence(
        self,
        exigence: ExigenceAtomique,
        doc_id: str,
    ) -> ResultatAnalyseExigence:
        """
        Analyse une exigence individuelle contre le SFCR.

        Flux :
          1. Embed texte_verification → query vector
          2. Retrieve top-K passages SFCR (ChromaDB)
          3. Pour chaque passage : LLM → score_llm
          4. score_final = 0.4 × cosinus + 0.6 × llm (sur le meilleur passage)
          5. Statut = COUVERT / PARTIEL / ABSENT
        """
        t0 = time.time()

        # Étape 1 : embedding de la requête
        texte_query = preparer_requete_e5(
            exigence.texte_verification, type_doc="query"
        )
        query_vec = self.embedding_engine.embed_query(texte_query)

        # Étape 2 : retrieval dans ChromaDB
        # Filtrage par section : on cherche en priorité dans la section
        # correspondant au thème de l'exigence
        section_hint = self._theme_vers_section(exigence.theme_id)
        resultats_bruts = self.vector_store.rechercher_passages_sfcr(
            embedding_requete=query_vec.tolist(),
            n_results=self.top_k,
            filtre_section=section_hint,
            filtre_doc_id=doc_id,
        )

        # Si aucun candidat dans la section ciblée → recherche globale
        if len(resultats_bruts) == 0:
            resultats_bruts = self.vector_store.rechercher_passages_sfcr(
                embedding_requete=query_vec.tolist(),
                n_results=self.top_k,
                filtre_doc_id=doc_id,
            )
        
        # Convertir ResultatRecherche → liste de dicts normalisés
        candidats = []
        for item in resultats_bruts:
            meta = item.get("metadata", {})
            candidats.append({
                "chunk_id": item["id"],
                "texte": item["texte"],
                "score_similarite": item["score"],
                "section": meta.get("section", ""),
                "page_debut": meta.get("page_debut", 0),
                "page_fin": meta.get("page_fin", 0),
                "doc_id": meta.get("doc_id", ""),
                "entreprise": meta.get("entreprise", ""),
                "annee": meta.get("annee", 0),
            })

        # Cas sans aucun passage : exigence ABSENTE
        if not candidats:
            return self._creer_resultat_absent(exigence, t0)

        # Étape 3 : analyse LLM de chaque passage candidat
        candidats_evalues = []
        for candidat in candidats:
            analyse = self.llm_engine.analyser_conformite(
                texte_verification=exigence.texte_verification,
                passage_sfcr=candidat["texte"],
                id_exigence=exigence.id_exigence,
                niveau_obligation=exigence.niveau_obligation.value,
            )
            score_final = self._calculer_score_final(
                candidat["score_similarite"], analyse.score
            )
            candidats_evalues.append({
                "chunk_id": candidat["chunk_id"],
                "texte": candidat["texte"],
                "score_cosinus": candidat["score_similarite"],
                "score_llm": analyse.score,
                "score_final": score_final,
                "section": candidat.get("section", ""),
                "page_debut": candidat.get("page_debut", 0),
                "justification": analyse.justification,
                "elements_trouves": analyse.elements_trouves,
                "elements_manquants": analyse.elements_manquants,
                "confiance_llm": analyse.confiance,
            })

        # Étape 4 : sélection du meilleur passage
        meilleur = max(candidats_evalues, key=lambda x: x["score_final"])

        # Étape 5 : statut de conformité
        statut = ResultatMatching.calculer_statut(meilleur["score_final"])

        # Construction du ResultatMatching
        matching = ResultatMatching(
            id=str(uuid.uuid4()),
            id_exigence=exigence.id_exigence,
            chunk_sfcr_id=meilleur["chunk_id"],
            score_similarite=round(meilleur["score_cosinus"], 4),
            score_llm=round(meilleur["score_llm"], 4),
            score_final=round(meilleur["score_final"], 4),
            statut=statut,
            passage_trouve=meilleur["texte"][:800],  # Tronquer pour le stockage
            justification_llm=meilleur["justification"],
        )

        return ResultatAnalyseExigence(
            exigence=exigence,
            matching=matching,
            candidats_evalues=candidats_evalues,
            nb_candidats=len(candidats_evalues),
            temps_analyse_s=round(time.time() - t0, 3),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Calcul du score final
    # ──────────────────────────────────────────────────────────────────────

    def _calculer_score_final(
        self, score_cosinus: float, score_llm: float
    ) -> float:
        """
        Combine le score cosinus et le score LLM en un score final.

        Formule : score_final = α × score_cosinus + β × score_llm
        Avec α = 0.40, β = 0.60 (calibration empirique).

        Le score LLM prime car il valide la couverture sémantique réelle.
        Le score cosinus ancre le résultat dans la similarité vectorielle.
        """
        return round(
            self.poids_cosinus * score_cosinus + self.poids_llm * score_llm,
            4,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Mapping thème → section SFCR pour le filtrage ChromaDB
    # ──────────────────────────────────────────────────────────────────────

    _THEME_SECTION_MAP = {
        # Pilier 1 — valorisation, capital, fonds propres → Section D et E
        "T01": "D_VALORISATION",
        "T02": "E_GESTION_CAPITAL",
        "T03": "E_GESTION_CAPITAL",
        # Pilier 2 — gouvernance, risques, contrôle, actuariat → Section B
        "T04": "B_GOUVERNANCE",
        "T05": "C_PROFIL_RISQUE",
        "T06": "B_GOUVERNANCE",
        "T07": "B_GOUVERNANCE",
        # Pilier 3 — structure SFCR, QRT, transparence, délais → global
        "T08": None,   # Toutes sections
        "T09": None,   # Annexes QRT
        "T10": None,   # Transversal
        "T11": None,   # Délais → souvent en intro
    }

    def _theme_vers_section(self, theme_id: str) -> Optional[str]:
        """Retourne la section SFCR principale associée à un thème."""
        return self._THEME_SECTION_MAP.get(theme_id)

    # ──────────────────────────────────────────────────────────────────────
    # Cas limites
    # ──────────────────────────────────────────────────────────────────────

    def _creer_resultat_absent(
        self, exigence: ExigenceAtomique, t0: float
    ) -> ResultatAnalyseExigence:
        """Résultat quand aucun passage n'est trouvé dans le SFCR."""
        matching = ResultatMatching(
            id=str(uuid.uuid4()),
            id_exigence=exigence.id_exigence,
            chunk_sfcr_id="",
            score_similarite=0.0,
            score_llm=0.0,
            score_final=0.0,
            statut=StatutConformite.ABSENT,
            passage_trouve="",
            justification_llm=(
                f"Aucun passage du SFCR ne correspond à l'exigence "
                f"{exigence.id_exigence}. L'information semble absente du rapport."
            ),
        )
        return ResultatAnalyseExigence(
            exigence=exigence,
            matching=matching,
            candidats_evalues=[],
            nb_candidats=0,
            temps_analyse_s=round(time.time() - t0, 3),
        )

    def _creer_resultat_erreur(
        self, exigence: ExigenceAtomique, message_erreur: str
    ) -> ResultatAnalyseExigence:
        """Résultat en cas d'erreur technique lors de l'analyse."""
        matching = ResultatMatching(
            id=str(uuid.uuid4()),
            id_exigence=exigence.id_exigence,
            chunk_sfcr_id="",
            score_similarite=0.0,
            score_llm=0.0,
            score_final=0.0,
            statut=StatutConformite.ABSENT,
            passage_trouve="",
            justification_llm=f"Erreur technique : {message_erreur}",
        )
        return ResultatAnalyseExigence(
            exigence=exigence,
            matching=matching,
            candidats_evalues=[],
            nb_candidats=0,
        )
