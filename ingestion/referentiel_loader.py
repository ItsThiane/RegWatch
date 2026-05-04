"""
referentiel_loader.py — Chargeur et validateur du référentiel
=============================================================
Responsabilités :
  1. Charger les exigences atomiques depuis les fichiers JSON
  2. Valider la cohérence du référentiel complet
  3. Exposer une API simple pour le reste du pipeline

Usage :
    from src.ingestion.referentiel_loader import ReferentielLoader

    loader = ReferentielLoader()
    loader.charger()

    exigences = loader.get_exigences_par_theme("T08")
    stats = loader.stats()
"""

import json
import logging
from pathlib import Path
from typing import Optional

from src.ingestion.models import (
    ExigenceAtomique, NiveauObligation, Theme, Pilier
)
from src.ingestion.referentiel_solvabilite2 import (
    SOURCES, PILIERS, THEMES, rapport_structure
)

logger = logging.getLogger(__name__)


class ReferentielLoader:
    """
    Charge et valide le référentiel Solvabilité II complet.

    Architecture de stockage :
        data/processed/exigences/
            T01_valorisation.json
            T02_capital.json
            T03_fonds_propres.json
            T04_gouvernance.json
            T05_risques_orsa.json
            T06_controle_audit.json
            T07_actuariat.json
            T08_structure_sfcr.json
            T09_qrt.json
            T10_transparence.json
            T11_delais.json

    Chaque fichier JSON contient une liste d'ExigenceAtomique sérialisées.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed/exigences")
        self.sources = SOURCES
        self.piliers = PILIERS
        self.themes = THEMES
        self._exigences: dict[str, ExigenceAtomique] = {}
        self._charge = False

    # ──────────────────────────────────────────────────────────────────────
    # Chargement
    # ──────────────────────────────────────────────────────────────────────

    def charger(self, strict: bool = True) -> "ReferentielLoader":
        """
        Charge toutes les exigences depuis data/processed/exigences/*.json.

        Args:
            strict: Si True, lève une exception si des thèmes sont vides.
                    Si False, log un warning et continue.
        Returns:
            self (pour chaînage)
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        fichiers_json = sorted(self.data_dir.glob("*.json"))

        if not fichiers_json:
            msg = (f"Aucun fichier JSON trouvé dans {self.data_dir}.\n"
                   f"Exécutez d'abord : python scripts/init_referentiel.py")
            if strict:
                raise FileNotFoundError(msg)
            else:
                logger.warning(msg)
                self._charge = True
                return self

        nb_charges = 0
        erreurs = []

        for fichier in fichiers_json:
            try:
                with open(fichier, "r", encoding="utf-8") as f:
                    donnees = json.load(f)

                for item in donnees:
                    try:
                        exigence = self._deserialiser_exigence(item)
                        if exigence.id_exigence in self._exigences:
                            raise ValueError(
                                f"Doublon d'id_exigence : {exigence.id_exigence}"
                            )
                        self._exigences[exigence.id_exigence] = exigence
                        nb_charges += 1
                    except Exception as e:
                        erreurs.append(f"  [{fichier.name}] {e}")

            except json.JSONDecodeError as e:
                erreurs.append(f"  [{fichier.name}] JSON invalide : {e}")

        if erreurs:
            msg = f"{len(erreurs)} erreur(s) au chargement :\n" + "\n".join(erreurs)
            if strict:
                raise ValueError(msg)
            else:
                logger.warning(msg)

        logger.info(f"Référentiel chargé : {nb_charges} exigences depuis {len(fichiers_json)} fichiers")
        self._charge = True
        self._valider()
        return self

    def _deserialiser_exigence(self, data: dict) -> ExigenceAtomique:
        """Reconstruit une ExigenceAtomique depuis un dict JSON."""
        return ExigenceAtomique(
            id_exigence=data["id_exigence"],
            theme_id=data["theme_id"],
            source_id=data["source_id"],
            article=data["article"],
            niveau_obligation=NiveauObligation(data["niveau_obligation"]),
            texte_original=data["texte_original"],
            texte_normalise=data["texte_normalise"],
            texte_verification=data["texte_verification"],
            mots_cles=data.get("mots_cles", []),
            notes_interpretatives=data.get("notes_interpretatives", ""),
            entites_concernees=data.get("entites_concernees", ["entreprise_assurance"]),
            actif=data.get("actif", True),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────────────────────────────────

    def _valider(self):
        """
        Vérifie la cohérence interne du référentiel chargé.
        Lève une ValueError si des incohérences sont détectées.
        """
        erreurs = []

        for exigence in self._exigences.values():
            # theme_id valide
            if exigence.theme_id not in self.themes:
                erreurs.append(
                    f"{exigence.id_exigence} → theme_id inconnu : {exigence.theme_id}"
                )
            # source_id valide
            if exigence.source_id not in self.sources:
                erreurs.append(
                    f"{exigence.id_exigence} → source_id inconnu : {exigence.source_id}"
                )

        if erreurs:
            raise ValueError(
                f"Incohérences dans le référentiel :\n" + "\n".join(f"  • {e}" for e in erreurs)
            )

        # Thèmes sans exigences (warning seulement)
        for theme_id in self.themes:
            exigences_theme = self.get_exigences_par_theme(theme_id)
            if not exigences_theme:
                logger.warning(f"Thème {theme_id} n'a aucune exigence chargée")

        logger.info("Validation du référentiel : OK")

    # ──────────────────────────────────────────────────────────────────────
    # API de consultation
    # ──────────────────────────────────────────────────────────────────────

    def _verifier_charge(self):
        if not self._charge:
            raise RuntimeError("Référentiel non chargé. Appelez d'abord .charger()")

    def get_exigences_par_theme(self, theme_id: str) -> list[ExigenceAtomique]:
        self._verifier_charge()
        return [e for e in self._exigences.values()
                if e.theme_id == theme_id and e.actif]

    def get_exigences_par_pilier(self, pilier_id: str) -> list[ExigenceAtomique]:
        self._verifier_charge()
        themes_ids = {t.id for t in self.themes.values() if t.pilier_id == pilier_id}
        return [e for e in self._exigences.values()
                if e.theme_id in themes_ids and e.actif]

    def get_exigences_shall(self) -> list[ExigenceAtomique]:
        """Retourne uniquement les obligations absolues (SHALL)."""
        self._verifier_charge()
        return [e for e in self._exigences.values()
                if e.niveau_obligation == NiveauObligation.SHALL and e.actif]

    def get_exigence(self, id_exigence: str) -> Optional[ExigenceAtomique]:
        self._verifier_charge()
        return self._exigences.get(id_exigence)

    def get_toutes_exigences(self) -> list[ExigenceAtomique]:
        self._verifier_charge()
        return [e for e in self._exigences.values() if e.actif]

    def get_textes_verification(self) -> list[tuple[str, str]]:
        """
        Retourne les paires (id_exigence, texte_verification) pour le RAG.
        C'est ce texte qui sera embedé pour la recherche de similarité.
        """
        self._verifier_charge()
        return [(e.id_exigence, e.texte_verification)
                for e in self._exigences.values() if e.actif]

    # ──────────────────────────────────────────────────────────────────────
    # Statistiques et diagnostic
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Statistiques complètes du référentiel chargé."""
        self._verifier_charge()
        exigences = list(self._exigences.values())

        par_obligation = {
            level.value: sum(1 for e in exigences if e.niveau_obligation == level)
            for level in NiveauObligation
        }
        par_theme = {
            theme_id: len(self.get_exigences_par_theme(theme_id))
            for theme_id in self.themes
        }
        par_pilier = {
            pilier_id: len(self.get_exigences_par_pilier(pilier_id))
            for pilier_id in self.piliers
        }

        return {
            "total_exigences": len(exigences),
            "par_obligation": par_obligation,
            "par_theme": par_theme,
            "par_pilier": par_pilier,
            "nb_themes": len(self.themes),
            "nb_piliers": len(self.piliers),
            "nb_sources": len(self.sources),
        }

    def afficher_stats(self) -> str:
        """Affiche les statistiques du référentiel sous forme lisible."""
        s = self.stats()
        lines = [
            "=" * 55,
            "RÉFÉRENTIEL CHARGÉ — STATISTIQUES",
            "=" * 55,
            f"Total exigences actives : {s['total_exigences']}",
            "",
            "Par niveau d'obligation :",
            f"  SHALL  (obligatoire) : {s['par_obligation']['SHALL']:>3}",
            f"  SHOULD (recommandé)  : {s['par_obligation']['SHOULD']:>3}",
            f"  MAY    (optionnel)   : {s['par_obligation']['MAY']:>3}",
            "",
            "Par pilier :",
        ]
        for pilier_id, nb in s["par_pilier"].items():
            pilier = self.piliers[pilier_id]
            lines.append(f"  [{pilier_id}] {pilier.intitule[:40]:<40} : {nb:>3} exigences")
        lines.append("")
        lines.append("Par thème :")
        for theme_id, nb in s["par_theme"].items():
            theme = self.themes[theme_id]
            lines.append(f"  [{theme_id}] {theme.libelle[:40]:<40} : {nb:>3} exigences")

        return "\n".join(lines)

    def __repr__(self) -> str:
        status = f"{len(self._exigences)} exigences" if self._charge else "non chargé"
        return f"ReferentielLoader({status})"
