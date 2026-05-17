"""
scoring_engine.py — Moteur de scoring de conformité Solvabilité II
==================================================================
Transforme les ResultatMatching bruts du RAG en un rapport de
conformité structuré, hiérarchique et actionnable.

Hiérarchie de scoring :

  Niveau 1 — Score par exigence
    score_exigence = score_final (issu du RAG)
    pondéré par poids_obligation (SHALL=1.0, SHOULD=0.6, MAY=0.3)

  Niveau 2 — Score par thème
    score_theme = Σ(score_exigence × poids_obligation) / Σ(poids_obligation)
    → Moyenne pondérée par le niveau d'obligation des exigences du thème

  Niveau 3 — Score par pilier
    score_pilier = Σ(score_theme × poids_dans_pilier)
    → Somme pondérée par l'importance relative de chaque thème dans son pilier

  Niveau 4 — Score global
    score_global = Σ(score_pilier × poids_scoring_pilier)
    → Somme pondérée par les poids des 3 piliers (P1=0.30, P2=0.35, P3=0.35)

  Niveau 5 — Notation
    A (≥0.80) : Conforme
    B (≥0.65) : Satisfaisant
    C (≥0.45) : Insuffisant
    D (<0.45)  : Non conforme

Justification des pondérations :
  - SHALL pèse 1.0 : non-conformité = écart critique (directive européenne)
  - SHOULD pèse 0.6 : non-conformité = point d'amélioration (guideline EIOPA)
  - MAY pèse 0.3 : non-conformité = observation mineure
  - Pilier 2 et 3 ont le même poids (0.35) car le SFCR porte directement
    sur la gouvernance (P2) et la transparence (P3)
  - Pilier 1 pèse 0.30 car les données quantitatives précises sont dans
    les QRT, pas dans le texte narratif du SFCR
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from src.ingestion.models import (
    ExigenceAtomique, ResultatMatching,
    ScoreTheme, RapportConformite,
    StatutConformite, NiveauConformiteGlobal, NiveauObligation,
)
from src.ingestion.referentiel_solvabilite2 import PILIERS, THEMES
from src.ingestion.referentiel_loader import ReferentielLoader
from src.rag.rag_pipeline import ResultatAnalyseSFCR

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structures intermédiaires de scoring
# ---------------------------------------------------------------------------

@dataclass
class ScoreExigence:
    """Score détaillé pour une exigence individuelle."""
    id_exigence: str
    theme_id: str
    niveau_obligation: NiveauObligation
    poids_obligation: float
    score_brut: float          # Score RAG direct (cosinus + LLM)
    score_pondere: float       # score_brut × poids_obligation
    statut: StatutConformite
    passage_trouve: str
    justification: str

    @property
    def est_critique(self) -> bool:
        """Exigence SHALL absente = critique."""
        return (
            self.niveau_obligation == NiveauObligation.SHALL
            and self.statut == StatutConformite.ABSENT
        )


@dataclass
class ScorePilier:
    """Score agrégé pour un pilier."""
    pilier_id: str
    intitule: str
    poids_scoring: float
    score_brut: float          # Moyenne pondérée des thèmes
    score_pondere: float       # score_brut × poids_scoring
    scores_themes: list[ScoreTheme] = field(default_factory=list)

    @property
    def niveau(self) -> str:
        if self.score_brut >= 0.80:   return "A"
        elif self.score_brut >= 0.65: return "B"
        elif self.score_brut >= 0.45: return "C"
        else:                          return "D"


# ---------------------------------------------------------------------------
# Moteur de scoring principal
# ---------------------------------------------------------------------------

class ScoringEngine:
    """
    Calcule le score de conformité hiérarchique à partir des matchings RAG.

    Usage :
        engine = ScoringEngine(loader)
        rapport = engine.calculer(resultat_rag)
        print(rapport.score_global)
        print(rapport.niveau_conformite.value)
    """

    def __init__(self, loader: ReferentielLoader):
        self.loader = loader
        self._exigences_map: dict[str, ExigenceAtomique] = {
            e.id_exigence: e
            for e in loader.get_toutes_exigences()
        }

    # ──────────────────────────────────────────────────────────────────────
    # Point d'entrée principal
    # ──────────────────────────────────────────────────────────────────────

    def calculer(
        self,
        resultat_rag: ResultatAnalyseSFCR,
        doc_sfcr_id: Optional[str] = None,
    ) -> RapportConformite:
        """
        Calcule le rapport de conformité complet à partir du résultat RAG.

        Args:
            resultat_rag : Résultat de RAGPipeline.analyser_sfcr()
            doc_sfcr_id  : ID du document SFCR (optionnel, pour métadonnées)

        Returns:
            RapportConformite avec scores, niveaux et recommandations
        """
        import uuid

        matchings = resultat_rag.matchings
        logger.info(
            f"Calcul du score de conformité : {len(matchings)} matchings"
        )

        # ── Niveau 1 : scores par exigence ────────────────────────────
        scores_exigences = self._calculer_scores_exigences(matchings)

        # ── Niveau 2 : scores par thème ───────────────────────────────
        scores_themes_map = self._calculer_scores_themes(scores_exigences)

        # ── Niveau 3 : scores par pilier ──────────────────────────────
        scores_piliers = self._calculer_scores_piliers(scores_themes_map)

        # ── Niveau 4 : score global ───────────────────────────────────
        score_global = sum(
            sp.score_pondere for sp in scores_piliers.values()
        )
        score_global = round(min(1.0, max(0.0, score_global)), 4)

        # ── Niveau 5 : notation ───────────────────────────────────────
        niveau = RapportConformite.niveau_depuis_score(score_global)

        # ── Statistiques globales ─────────────────────────────────────
        nb_couvertes  = sum(1 for s in scores_exigences if s.statut == StatutConformite.COUVERT)
        nb_partielles = sum(1 for s in scores_exigences if s.statut == StatutConformite.PARTIEL)
        nb_absentes   = sum(1 for s in scores_exigences if s.statut == StatutConformite.ABSENT)

        # ── Recommandations ───────────────────────────────────────────
        recommandations = self._generer_recommandations(
            scores_exigences, scores_themes_map, scores_piliers
        )

        # ── Construction du rapport ───────────────────────────────────
        rapport = RapportConformite(
            id_analyse=str(uuid.uuid4()),
            document_id=doc_sfcr_id or resultat_rag.doc_id,
            entreprise=resultat_rag.entreprise,
            annee_rapport=resultat_rag.annee,
            scores_themes=list(scores_themes_map.values()),
            score_pilier_1=round(scores_piliers["P1"].score_brut, 4),
            score_pilier_2=round(scores_piliers["P2"].score_brut, 4),
            score_pilier_3=round(scores_piliers["P3"].score_brut, 4),
            score_global=score_global,
            niveau_conformite=niveau,
            nb_exigences_analysees=len(matchings),
            nb_exigences_couvertes=nb_couvertes,
            nb_exigences_partielles=nb_partielles,
            nb_exigences_absentes=nb_absentes,
            recommandations=recommandations,
        )

        logger.info(
            f"Score global : {score_global:.3f} → Niveau {niveau.value} | "
            f"C={nb_couvertes} P={nb_partielles} A={nb_absentes}"
        )

        return rapport

    # ──────────────────────────────────────────────────────────────────────
    # Niveau 1 : scores par exigence
    # ──────────────────────────────────────────────────────────────────────

    def _calculer_scores_exigences(
        self, matchings: list[ResultatMatching]
    ) -> list[ScoreExigence]:
        """
        Transforme chaque ResultatMatching en ScoreExigence pondéré.

        La pondération par poids_obligation est l'élément clé :
        un SHALL manquant pèse 3.3× plus qu'un MAY manquant dans
        le score final du thème.
        """
        scores = []
        for matching in matchings:
            exigence = self._exigences_map.get(matching.id_exigence)
            if not exigence:
                logger.warning(
                    f"Exigence {matching.id_exigence} non trouvée "
                    f"dans le référentiel — ignorée"
                )
                continue

            score_brut = matching.score_final
            poids = exigence.poids_obligation
            score_pondere = score_brut * poids

            scores.append(ScoreExigence(
                id_exigence=matching.id_exigence,
                theme_id=exigence.theme_id,
                niveau_obligation=exigence.niveau_obligation,
                poids_obligation=poids,
                score_brut=round(score_brut, 4),
                score_pondere=round(score_pondere, 4),
                statut=matching.statut,
                passage_trouve=matching.passage_trouve,
                justification=matching.justification_llm,
            ))

        return scores

    # ──────────────────────────────────────────────────────────────────────
    # Niveau 2 : scores par thème
    # ──────────────────────────────────────────────────────────────────────

    def _calculer_scores_themes(
        self, scores_exigences: list[ScoreExigence]
    ) -> dict[str, ScoreTheme]:
        """
        Agrège les scores d'exigences par thème via moyenne pondérée.

        Formule :
            score_theme = Σ(score_brut × poids_obligation)
                        / Σ(poids_obligation)

        Cette formule garantit que les SHALLS dominent le score du thème
        même si les SHOULD et MAY sont bien couverts.
        """
        # Grouper les scores par thème
        par_theme: dict[str, list[ScoreExigence]] = {}
        for se in scores_exigences:
            par_theme.setdefault(se.theme_id, []).append(se)

        scores_themes: dict[str, ScoreTheme] = {}

        for theme_id, theme_scores in par_theme.items():
            theme = THEMES.get(theme_id)
            if not theme:
                logger.warning(f"Thème {theme_id} inconnu dans THEMES")
                continue

            # Comptages par statut
            nb_couvertes  = sum(1 for s in theme_scores
                               if s.statut == StatutConformite.COUVERT)
            nb_partielles = sum(1 for s in theme_scores
                               if s.statut == StatutConformite.PARTIEL)
            nb_absentes   = sum(1 for s in theme_scores
                               if s.statut == StatutConformite.ABSENT)

            # Moyenne pondérée par poids_obligation
            somme_ponderee = sum(s.score_brut * s.poids_obligation
                                for s in theme_scores)
            somme_poids    = sum(s.poids_obligation for s in theme_scores)

            score_brut = (
                round(somme_ponderee / somme_poids, 4)
                if somme_poids > 0 else 0.0
            )
            score_pondere = round(score_brut * theme.poids_dans_pilier, 4)

            # Exigences SHALL absentes — alertes critiques
            shall_absentes = [
                s.id_exigence for s in theme_scores
                if s.est_critique
            ]

            scores_themes[theme_id] = ScoreTheme(
                theme_id=theme_id,
                libelle_theme=theme.libelle,
                nb_exigences_total=len(theme_scores),
                nb_couvertes=nb_couvertes,
                nb_partielles=nb_partielles,
                nb_absentes=nb_absentes,
                score_brut=score_brut,
                score_pondere=score_pondere,
                exigences_critiques_absentes=shall_absentes,
            )

        # Ajouter les thèmes sans aucun matching (score 0 par défaut)
        for theme_id, theme in THEMES.items():
            if theme_id not in scores_themes:
                logger.warning(
                    f"Thème {theme_id} sans aucun matching — score=0.0"
                )
                scores_themes[theme_id] = ScoreTheme(
                    theme_id=theme_id,
                    libelle_theme=theme.libelle,
                    nb_exigences_total=0,
                    nb_couvertes=0,
                    nb_partielles=0,
                    nb_absentes=0,
                    score_brut=0.0,
                    score_pondere=0.0,
                )

        return scores_themes

    # ──────────────────────────────────────────────────────────────────────
    # Niveau 3 : scores par pilier
    # ──────────────────────────────────────────────────────────────────────

    def _calculer_scores_piliers(
        self, scores_themes: dict[str, ScoreTheme]
    ) -> dict[str, ScorePilier]:
        """
        Agrège les scores de thèmes par pilier via somme pondérée.

        Formule :
            score_pilier = Σ(score_theme_brut × poids_dans_pilier)

        Les poids_dans_pilier somment à 1.0 par pilier (validé à l'import
        du module referentiel_solvabilite2.py), donc c'est une moyenne
        pondérée implicite.
        """
        scores_piliers: dict[str, ScorePilier] = {}

        for pilier_id, pilier in PILIERS.items():
            themes_du_pilier = [
                t for t in THEMES.values()
                if t.pilier_id == pilier_id
            ]

            score_brut = sum(
                scores_themes[t.id].score_brut * t.poids_dans_pilier
                for t in themes_du_pilier
                if t.id in scores_themes
            )
            score_brut = round(min(1.0, max(0.0, score_brut)), 4)
            score_pondere = round(score_brut * pilier.poids_scoring, 4)

            scores_piliers[pilier_id] = ScorePilier(
                pilier_id=pilier_id,
                intitule=pilier.intitule,
                poids_scoring=pilier.poids_scoring,
                score_brut=score_brut,
                score_pondere=score_pondere,
                scores_themes=[
                    scores_themes[t.id]
                    for t in themes_du_pilier
                    if t.id in scores_themes
                ],
            )

        return scores_piliers

    # ──────────────────────────────────────────────────────────────────────
    # Recommandations automatiques
    # ──────────────────────────────────────────────────────────────────────

    def _generer_recommandations(
        self,
        scores_exigences: list[ScoreExigence],
        scores_themes: dict[str, ScoreTheme],
        scores_piliers: dict[str, ScorePilier],
    ) -> list[str]:
        """
        Génère des recommandations priorisées basées sur les gaps identifiés.

        Priorité :
          1. Exigences SHALL absentes (non-conformité critique)
          2. Thèmes sous le seuil PARTIEL (score < 0.45)
          3. Piliers faibles (score < 0.50)
          4. Recommandations générales si score global élevé
        """
        recommandations = []

        # ── Priorité 1 : SHALL absents ────────────────────────────────
        shall_absents = [
            se for se in scores_exigences if se.est_critique
        ]
        if shall_absents:
            recommandations.append(
                f"🔴 CRITIQUE — {len(shall_absents)} obligation(s) SHALL "
                f"non couverte(s) : "
                + ", ".join(se.id_exigence for se in shall_absents[:5])
                + (" ..." if len(shall_absents) > 5 else "")
                + ". Ces points doivent être traités en priorité absolue."
            )

        # ── Priorité 2 : Thèmes critiques ────────────────────────────
        themes_critiques = [
            st for st in scores_themes.values()
            if st.score_brut < 0.45 and st.nb_exigences_total > 0
        ]
        themes_critiques.sort(key=lambda t: t.score_brut)
        for st in themes_critiques[:3]:
            nb_manquants = st.nb_absentes + st.nb_partielles
            recommandations.append(
                f"🟠 INSUFFISANT — [{st.theme_id}] {st.libelle_theme} "
                f"(score={st.score_brut:.2f}) : "
                f"{nb_manquants} exigence(s) à compléter. "
                + (
                    f"Exigences critiques manquantes : "
                    f"{', '.join(st.exigences_critiques_absentes[:3])}."
                    if st.exigences_critiques_absentes else
                    "Renforcer le niveau de détail dans cette section."
                )
            )

        # ── Priorité 3 : Thèmes partiels ─────────────────────────────
        themes_partiels = [
            st for st in scores_themes.values()
            if 0.45 <= st.score_brut < 0.65 and st.nb_exigences_total > 0
        ]
        themes_partiels.sort(key=lambda t: t.score_brut)
        for st in themes_partiels[:3]:
            recommandations.append(
                f"🟡 PARTIEL — [{st.theme_id}] {st.libelle_theme} "
                f"(score={st.score_brut:.2f}) : "
                f"couverture partielle sur {st.nb_partielles} exigence(s). "
                f"Approfondir les développements narratifs de cette section."
            )

        # ── Priorité 4 : Piliers faibles ─────────────────────────────
        for pid, sp in scores_piliers.items():
            if sp.score_brut < 0.50:
                recommandations.append(
                    f"📊 PILIER {pid} faible (score={sp.score_brut:.2f}) — "
                    f"{sp.intitule} : revoir l'ensemble des sections "
                    f"correspondantes dans le rapport."
                )

        # ── Score global satisfaisant : recommandations de qualité ────
        if not recommandations:
            recommandations.append(
                "✅ Score de conformité satisfaisant. "
                "Points d'amélioration éventuels : enrichir les comparatifs "
                "N/N-1, renforcer les analyses de sensibilité, et s'assurer "
                "de la cohérence entre le texte narratif et les QRT."
            )

        return recommandations

    # ──────────────────────────────────────────────────────────────────────
    # Rapport textuel
    # ──────────────────────────────────────────────────────────────────────

    def formater_rapport(
        self,
        rapport: RapportConformite,
        scores_themes: Optional[dict[str, ScoreTheme]] = None,
        verbose: bool = False,
    ) -> str:
        """
        Formate le rapport de conformité en texte lisible.

        Args:
            rapport       : RapportConformite calculé
            scores_themes : Scores détaillés par thème (optionnel)
            verbose       : Si True, inclut les détails par thème
        """
        niveau_emoji = {
            "A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"
        }
        nv = rapport.niveau_conformite.value
        emoji = niveau_emoji.get(nv, "")

        lignes = [
            "=" * 65,
            "RAPPORT DE CONFORMITÉ SOLVABILITÉ II — RegWatch",
            "=" * 65,
            f"Entreprise    : {rapport.entreprise}",
            f"Exercice      : {rapport.annee_rapport}",
            f"Date analyse  : {rapport.date_analyse[:10]}",
            "",
            f"SCORE GLOBAL  : {rapport.score_global:.3f} / 1.000",
            f"NIVEAU        : {emoji} {nv} — "
            + {"A": "Conforme",
               "B": "Satisfaisant",
               "C": "Insuffisant",
               "D": "Non conforme"}[nv],
            "",
            "Scores par pilier :",
            f"  Pilier 1 (Quantitatif, 30%)  : {rapport.score_pilier_1:.3f}",
            f"  Pilier 2 (Gouvernance, 35%)  : {rapport.score_pilier_2:.3f}",
            f"  Pilier 3 (Reporting, 35%)    : {rapport.score_pilier_3:.3f}",
            "",
            "Statistiques :",
            f"  Exigences analysées : {rapport.nb_exigences_analysees}",
            f"  ✅ Couvertes         : {rapport.nb_exigences_couvertes}"
            f" ({rapport.nb_exigences_couvertes/max(rapport.nb_exigences_analysees,1):.0%})",
            f"  ⚠️  Partielles        : {rapport.nb_exigences_partielles}"
            f" ({rapport.nb_exigences_partielles/max(rapport.nb_exigences_analysees,1):.0%})",
            f"  ❌ Absentes          : {rapport.nb_exigences_absentes}"
            f" ({rapport.nb_exigences_absentes/max(rapport.nb_exigences_analysees,1):.0%})",
        ]

        if verbose and scores_themes:
            lignes += ["", "─" * 65, "DÉTAIL PAR THÈME", "─" * 65]
            for tid in sorted(scores_themes.keys()):
                st = scores_themes[tid]
                if st.nb_exigences_total == 0:
                    continue
                barre = self._barre_progression(st.score_brut)
                niv_st = (
                    "🟢 A" if st.score_brut >= 0.80 else
                    "🟡 B" if st.score_brut >= 0.65 else
                    "🟠 C" if st.score_brut >= 0.45 else
                    "🔴 D"
                )
                lignes.append(
                    f"[{tid}] {st.libelle_theme[:35]:<35} "
                    f"{barre} {st.score_brut:.3f} {niv_st}"
                )
                lignes.append(
                    f"      C={st.nb_couvertes} P={st.nb_partielles} "
                    f"A={st.nb_absentes} / {st.nb_exigences_total}"
                )
                if st.exigences_critiques_absentes:
                    lignes.append(
                        f"      ⚠️  SHALL absents : "
                        f"{', '.join(st.exigences_critiques_absentes[:3])}"
                    )

        lignes += ["", "─" * 65, "RECOMMANDATIONS", "─" * 65]
        for i, rec in enumerate(rapport.recommandations, 1):
            lignes.append(f"{i}. {rec}")

        lignes += ["", "=" * 65]
        return "\n".join(lignes)

    @staticmethod
    def _barre_progression(score: float, largeur: int = 10) -> str:
        """Génère une barre de progression ASCII."""
        nb_pleins = round(score * largeur)
        return "[" + "█" * nb_pleins + "░" * (largeur - nb_pleins) + "]"
