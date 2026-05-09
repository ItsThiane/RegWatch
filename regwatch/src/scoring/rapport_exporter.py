"""
rapport_exporter.py — Export du rapport de conformité
======================================================
Produit deux formats d'export :
  - JSON  : rapport complet structuré (archivage, API)
  - CSV   : tableau plat par exigence (analyse Excel / actuaire)

Le JSON est le format de référence — il contient toutes les
informations nécessaires pour régénérer n'importe quel autre format.
"""

from __future__ import annotations
import csv
import json
import logging
from pathlib import Path
from typing import Optional

from src.ingestion.models import RapportConformite, ScoreTheme
from src.scoring.scoring_engine import ScoreExigence, ScorePilier

logger = logging.getLogger(__name__)


class RapportExporter:
    """Exporte le rapport de conformité en JSON et CSV."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("data/rapports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def exporter_json(
        self,
        rapport: RapportConformite,
        scores_exigences: list[ScoreExigence],
        scores_themes: dict[str, ScoreTheme],
        scores_piliers: dict[str, ScorePilier],
        nom_fichier: Optional[str] = None,
    ) -> Path:
        """
        Exporte le rapport complet en JSON.

        Structure :
          {
            "metadata": {...},
            "score_global": 0.xxx,
            "niveau": "A|B|C|D",
            "piliers": [{...}, ...],
            "themes": [{...}, ...],
            "exigences": [{...}, ...]   ← détail complet
          }
        """
        nom = nom_fichier or (
            f"rapport_{rapport.entreprise.replace(' ', '_')}_"
            f"{rapport.annee_rapport}_{rapport.date_analyse[:10]}.json"
        )
        chemin = self.output_dir / nom

        payload = {
            "metadata": {
                "outil": "RegWatch",
                "version": "1.0",
                "id_analyse": rapport.id_analyse,
                "entreprise": rapport.entreprise,
                "annee_rapport": rapport.annee_rapport,
                "date_analyse": rapport.date_analyse,
                "document_id": rapport.document_id,
            },
            "score_global": rapport.score_global,
            "niveau_conformite": rapport.niveau_conformite.value,
            "statistiques": {
                "nb_analysees": rapport.nb_exigences_analysees,
                "nb_couvertes": rapport.nb_exigences_couvertes,
                "nb_partielles": rapport.nb_exigences_partielles,
                "nb_absentes": rapport.nb_exigences_absentes,
                "taux_couverture": round(
                    rapport.nb_exigences_couvertes /
                    max(rapport.nb_exigences_analysees, 1), 4
                ),
            },
            "piliers": [
                {
                    "id": sp.pilier_id,
                    "intitule": sp.intitule,
                    "poids": sp.poids_scoring,
                    "score": sp.score_brut,
                    "score_pondere": sp.score_pondere,
                    "niveau": sp.niveau,
                }
                for sp in scores_piliers.values()
            ],
            "themes": [
                {
                    "id": st.theme_id,
                    "libelle": st.libelle_theme,
                    "score": st.score_brut,
                    "score_pondere": st.score_pondere,
                    "taux_couverture": st.taux_couverture,
                    "nb_total": st.nb_exigences_total,
                    "nb_couvertes": st.nb_couvertes,
                    "nb_partielles": st.nb_partielles,
                    "nb_absentes": st.nb_absentes,
                    "shall_critiques_absents": st.exigences_critiques_absentes,
                }
                for st in scores_themes.values()
                if st.nb_exigences_total > 0
            ],
            "exigences": [
                {
                    "id": se.id_exigence,
                    "theme_id": se.theme_id,
                    "obligation": se.niveau_obligation.value,
                    "poids_obligation": se.poids_obligation,
                    "statut": se.statut.value,
                    "score_brut": se.score_brut,
                    "score_pondere": se.score_pondere,
                    "justification": se.justification,
                    "passage_trouve": se.passage_trouve[:400]
                    if se.passage_trouve else "",
                }
                for se in scores_exigences
            ],
            "recommandations": rapport.recommandations,
        }

        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        logger.info(f"Rapport JSON exporté : {chemin}")
        return chemin

    def exporter_csv(
        self,
        scores_exigences: list[ScoreExigence],
        scores_themes: dict[str, ScoreTheme],
        rapport: RapportConformite,
        nom_fichier: Optional[str] = None,
    ) -> Path:
        """
        Exporte un tableau CSV par exigence.
        Format optimal pour analyse Excel par les actuaires.

        Colonnes :
          id_exigence, theme_id, libelle_theme, pilier_id,
          obligation, poids_obligation, statut,
          score_brut, score_pondere, justification
        """
        from src.ingestion.referentiel_solvabilite2 import THEMES, PILIERS

        nom = nom_fichier or (
            f"conformite_{rapport.entreprise.replace(' ', '_')}_"
            f"{rapport.annee_rapport}.csv"
        )
        chemin = self.output_dir / nom

        colonnes = [
            "id_exigence", "theme_id", "libelle_theme", "pilier_id",
            "niveau_obligation", "poids_obligation",
            "statut", "score_brut", "score_pondere",
            "score_theme", "score_pilier",
            "justification",
        ]

        with open(chemin, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=colonnes)
            writer.writeheader()

            for se in sorted(scores_exigences, key=lambda x: x.theme_id):
                theme = THEMES.get(se.theme_id)
                st = scores_themes.get(se.theme_id)
                pilier_id = theme.pilier_id if theme else ""

                writer.writerow({
                    "id_exigence": se.id_exigence,
                    "theme_id": se.theme_id,
                    "libelle_theme": theme.libelle if theme else "",
                    "pilier_id": pilier_id,
                    "niveau_obligation": se.niveau_obligation.value,
                    "poids_obligation": se.poids_obligation,
                    "statut": se.statut.value,
                    "score_brut": f"{se.score_brut:.4f}",
                    "score_pondere": f"{se.score_pondere:.4f}",
                    "score_theme": f"{st.score_brut:.4f}" if st else "",
                    "score_pilier": "",  # rempli ci-dessous
                    "justification": se.justification[:200],
                })

        logger.info(f"Rapport CSV exporté : {chemin}")
        return chemin
