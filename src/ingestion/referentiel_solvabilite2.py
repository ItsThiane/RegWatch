"""
referentiel_solvabilite2.py — Référentiel réglementaire RegWatch
================================================================
Définit la structure complète du référentiel Solvabilité II :
  - 3 sources réglementaires principales
  - 3 piliers avec pondérations
  - 11 thèmes avec poids dans leur pilier

Ce fichier est la "source de vérité" du référentiel.
Les exigences atomiques sont chargées depuis data/processed/exigences/*.json
via le ReferentielLoader (voir referentiel_loader.py).

Pondérations justifiées :
  Pilier 1 (0.30) — Le SFCR traite quantitativement la solvabilité mais
                     les données précises sont dans les QRT ; poids modéré.
  Pilier 2 (0.35) — La gouvernance est le cœur des exigences qualitatives
                     du SFCR ; poids le plus élevé.
  Pilier 3 (0.35) — La structure et le contenu du SFCR sont directement
                     vérifiables dans le document ; poids élevé.
"""

from src.ingestion.models import (
    SourceReglementaire, Pilier, Theme, NumeroPilier
)


# ---------------------------------------------------------------------------
# Sources réglementaires
# ---------------------------------------------------------------------------

SOURCES = {
    "DIR_2009_138": SourceReglementaire(
        id="DIR_2009_138",
        code="2009/138/CE",
        intitule="Directive 2009/138/CE du Parlement européen et du Conseil "
                 "sur l'accès aux activités de l'assurance et de la réassurance "
                 "et leur exercice (solvabilité II)",
        type_source="DIRECTIVE",
        date_publication="2009-11-25",
        url_eur_lex="https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32009L0138",
    ),
    "REG_2015_35": SourceReglementaire(
        id="REG_2015_35",
        code="2015/35",
        intitule="Règlement délégué (UE) 2015/35 de la Commission du 10 octobre 2014 "
                 "complétant la directive 2009/138/CE du Parlement européen et du Conseil "
                 "sur l'accès aux activités de l'assurance et de la réassurance "
                 "et leur exercice (Solvabilité II)",
        type_source="REGLEMENT",
        date_publication="2015-01-17",
        url_eur_lex="https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32015R0035",
    ),
    "EIOPA_SFCR": SourceReglementaire(
        id="EIOPA_SFCR",
        code="EIOPA-BoS-15/109",
        intitule="Guidelines on reporting and public disclosure — EIOPA "
                 "Guidelines on the Solvency and Financial Condition Report",
        type_source="GUIDELINE",
        date_publication="2015-07-02",
        url_eur_lex="https://www.eiopa.europa.eu/document-library/guidelines/"
                    "guidelines-reporting-and-public-disclosure_en",
    ),
}


# ---------------------------------------------------------------------------
# Piliers Solvabilité II
# ---------------------------------------------------------------------------

PILIERS = {
    "P1": Pilier(
        id="P1",
        numero=NumeroPilier.PILIER_1,
        intitule="Exigences quantitatives",
        description="Valorisation prudentielle des actifs et passifs, calcul des "
                    "provisions techniques, exigences de capital (SCR, MCR) et "
                    "gestion des fonds propres éligibles.",
        poids_scoring=0.30,
        themes=["T01", "T02", "T03"],
    ),
    "P2": Pilier(
        id="P2",
        numero=NumeroPilier.PILIER_2,
        intitule="Exigences qualitatives et supervision",
        description="Système de gouvernance, gestion des risques incluant l'ORSA, "
                    "contrôle interne, audit interne et fonction actuarielle.",
        poids_scoring=0.35,
        themes=["T04", "T05", "T06", "T07"],
    ),
    "P3": Pilier(
        id="P3",
        numero=NumeroPilier.PILIER_3,
        intitule="Reporting et transparence",
        description="Obligations de reporting public (SFCR) et prudentiel (RSR, QRT) "
                    "envers les autorités de contrôle et le marché.",
        poids_scoring=0.35,
        themes=["T08", "T09", "T10", "T11"],
    ),
}


# ---------------------------------------------------------------------------
# Thèmes réglementaires (11 thèmes, 3 piliers)
# ---------------------------------------------------------------------------

THEMES = {

    # ── Pilier 1 ──────────────────────────────────────────────────────────

    "T01": Theme(
        id="T01",
        libelle="Valorisation des actifs et passifs",
        pilier_id="P1",
        poids_dans_pilier=0.30,
        description="Méthodes et hypothèses de valorisation prudentielle des actifs, "
                    "des provisions techniques Best Estimate et Risk Margin.",
        articles_reference=["Art.75-86 Dir.", "Art.7-56 Règl.2015/35",
                            "Art.260 §1 Règl.2015/35"],
    ),
    "T02": Theme(
        id="T02",
        libelle="Exigences de capital (SCR / MCR)",
        pilier_id="P1",
        poids_dans_pilier=0.40,
        description="Calcul et couverture du Capital de Solvabilité Requis (SCR) "
                    "et du Minimum de Capital Requis (MCR), formule standard ou "
                    "modèle interne.",
        articles_reference=["Art.100-135 Dir.", "Art.103-126 Dir.",
                            "Art.261-262 Règl.2015/35"],
    ),
    "T03": Theme(
        id="T03",
        libelle="Fonds propres et éligibilité",
        pilier_id="P1",
        poids_dans_pilier=0.30,
        description="Structure, classification et éligibilité des fonds propres "
                    "pour la couverture du SCR et MCR.",
        articles_reference=["Art.87-99 Dir.", "Art.69-81 Règl.2015/35",
                            "Art.262 §2 Règl.2015/35"],
    ),

    # ── Pilier 2 ──────────────────────────────────────────────────────────

    "T04": Theme(
        id="T04",
        libelle="Système de gouvernance",
        pilier_id="P2",
        poids_dans_pilier=0.30,
        description="Structure organisationnelle, séparation des fonctions clés, "
                    "politique de rémunération, fit & proper des dirigeants.",
        articles_reference=["Art.41-49 Dir.", "Art.258-275 Règl.2015/35",
                            "Art.271 Règl.2015/35"],
    ),
    "T05": Theme(
        id="T05",
        libelle="Gestion des risques et ORSA",
        pilier_id="P2",
        poids_dans_pilier=0.30,
        description="Système de gestion des risques, évaluation interne des risques "
                    "et de la solvabilité (ORSA), appétit au risque.",
        articles_reference=["Art.44-45 Dir.", "Art.262-268 Règl.2015/35"],
    ),
    "T06": Theme(
        id="T06",
        libelle="Contrôle interne et audit",
        pilier_id="P2",
        poids_dans_pilier=0.20,
        description="Système de contrôle interne, fonction d'audit interne "
                    "indépendante, fonction de vérification de la conformité.",
        articles_reference=["Art.46-47 Dir.", "Art.271-272 Règl.2015/35"],
    ),
    "T07": Theme(
        id="T07",
        libelle="Fonction actuarielle",
        pilier_id="P2",
        poids_dans_pilier=0.20,
        description="Rôle, missions et indépendance de la fonction actuarielle "
                    "dans le calcul des provisions et la politique de souscription.",
        articles_reference=["Art.48 Dir.", "Art.272 Règl.2015/35"],
    ),

    # ── Pilier 3 ──────────────────────────────────────────────────────────

    "T08": Theme(
        id="T08",
        libelle="Structure et contenu du SFCR",
        pilier_id="P3",
        poids_dans_pilier=0.40,
        description="Exigences de contenu du rapport SFCR : sections A à E, "
                    "informations obligatoires par section (Art.290-303 Règl.2015/35).",
        articles_reference=["Art.51 Dir.", "Art.290-303 Règl.2015/35",
                            "EIOPA-BoS-15/109 GL01-GL06"],
    ),
    "T09": Theme(
        id="T09",
        libelle="Reporting quantitatif (QRT)",
        pilier_id="P3",
        poids_dans_pilier=0.20,
        description="Obligations de publication des tableaux quantitatifs (QRT) "
                    "annexés au SFCR (bilan S.02, primes S.05, SCR S.25...).",
        articles_reference=["Art.304-312 Règl.2015/35",
                            "EIOPA-BoS-15/109 GL07-GL09"],
    ),
    "T10": Theme(
        id="T10",
        libelle="Transparence et communication",
        pilier_id="P3",
        poids_dans_pilier=0.25,
        description="Exigences de clarté, accessibilité et non-tromperie du rapport, "
                    "information sur les changements significatifs.",
        articles_reference=["Art.51 §1 Dir.", "Art.289 Règl.2015/35",
                            "EIOPA-BoS-15/109 GL10-GL12"],
    ),
    "T11": Theme(
        id="T11",
        libelle="Délais et obligations de publication",
        pilier_id="P3",
        poids_dans_pilier=0.15,
        description="Délais de publication du SFCR (14 semaines après clôture), "
                    "mise à jour annuelle et publication en cours d'exercice.",
        articles_reference=["Art.51 §2 Dir.", "Art.308 Règl.2015/35"],
    ),
}


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def get_themes_par_pilier(pilier_id: str) -> list[Theme]:
    """Retourne les thèmes d'un pilier donné, triés par id."""
    return sorted(
        [t for t in THEMES.values() if t.pilier_id == pilier_id],
        key=lambda t: t.id
    )


def verifier_poids_pilier(pilier_id: str, tolerance: float = 0.01) -> bool:
    """
    Vérifie que la somme des poids des thèmes d'un pilier vaut 1.0.
    Lève une assertion si la contrainte n'est pas respectée.
    """
    themes = get_themes_par_pilier(pilier_id)
    total = sum(t.poids_dans_pilier for t in themes)
    ok = abs(total - 1.0) < tolerance
    if not ok:
        raise AssertionError(
            f"Pilier {pilier_id} : somme des poids = {total:.3f} ≠ 1.0\n"
            f"Thèmes : {[(t.id, t.poids_dans_pilier) for t in themes]}"
        )
    return True


def verifier_poids_piliers(tolerance: float = 0.01) -> bool:
    """Vérifie que la somme des poids des 3 piliers vaut 1.0."""
    total = sum(p.poids_scoring for p in PILIERS.values())
    ok = abs(total - 1.0) < tolerance
    if not ok:
        raise AssertionError(
            f"Somme des poids des piliers = {total:.3f} ≠ 1.0\n"
            f"Piliers : {[(p.id, p.poids_scoring) for p in PILIERS.values()]}"
        )
    return True


def rapport_structure() -> str:
    """Affiche un résumé lisible de la structure du référentiel."""
    lines = ["=" * 60, "RÉFÉRENTIEL SOLVABILITÉ II — STRUCTURE", "=" * 60]
    total_poids = sum(p.poids_scoring for p in PILIERS.values())
    lines.append(f"Poids total des piliers : {total_poids:.2f}\n")

    for pilier in PILIERS.values():
        lines.append(f"[{pilier.id}] {pilier.intitule} (poids={pilier.poids_scoring:.0%})")
        themes_pilier = get_themes_par_pilier(pilier.id)
        for theme in themes_pilier:
            lines.append(
                f"  └── [{theme.id}] {theme.libelle} "
                f"(poids_dans_pilier={theme.poids_dans_pilier:.0%})"
            )
        poids_total_themes = sum(t.poids_dans_pilier for t in themes_pilier)
        lines.append(f"       Σ poids thèmes = {poids_total_themes:.2f}")
        lines.append("")

    lines.append(f"Sources : {len(SOURCES)}")
    for src in SOURCES.values():
        lines.append(f"  • [{src.id}] {src.code} ({src.type_source})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-validation au chargement du module
# ---------------------------------------------------------------------------

def _auto_valider():
    """Validations exécutées à l'import du module — fail-fast."""
    # Poids des piliers
    verifier_poids_piliers()
    # Poids des thèmes par pilier
    for pilier_id in PILIERS:
        verifier_poids_pilier(pilier_id)
    # Cohérence thème → pilier
    for theme in THEMES.values():
        assert theme.pilier_id in PILIERS, (
            f"Thème {theme.id} référence un pilier inexistant : {theme.pilier_id}"
        )

_auto_valider()
