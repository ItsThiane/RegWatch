"""
tests/test_referentiel.py — Validation complète du référentiel
==============================================================
Tests exécutables avec : python -m pytest tests/test_referentiel.py -v
Ou directement : python tests/test_referentiel.py
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.models import (
    ExigenceAtomique, NiveauObligation, StatutConformite,
    NiveauConformiteGlobal, ResultatMatching
)
from src.ingestion.referentiel_solvabilite2 import (
    SOURCES, PILIERS, THEMES,
    verifier_poids_piliers, verifier_poids_pilier,
    get_themes_par_pilier, rapport_structure
)
from src.ingestion.referentiel_loader import ReferentielLoader


def test_poids_piliers():
    """La somme des poids des 3 piliers doit valoir 1.0."""
    total = sum(p.poids_scoring for p in PILIERS.values())
    assert abs(total - 1.0) < 0.01, f"Somme poids piliers = {total:.3f} ≠ 1.0"
    print(f"  ✓ Poids piliers : {[f'{p.id}={p.poids_scoring}' for p in PILIERS.values()]} → Σ={total:.2f}")


def test_poids_themes_par_pilier():
    """La somme des poids des thèmes dans chaque pilier doit valoir 1.0."""
    for pilier_id in PILIERS:
        themes = get_themes_par_pilier(pilier_id)
        total = sum(t.poids_dans_pilier for t in themes)
        assert abs(total - 1.0) < 0.01, (
            f"Pilier {pilier_id} : somme poids thèmes = {total:.3f} ≠ 1.0"
        )
        print(f"  ✓ Pilier {pilier_id} : {len(themes)} thèmes, Σ poids = {total:.2f}")


def test_coherence_themes_piliers():
    """Chaque thème référence un pilier existant."""
    for theme in THEMES.values():
        assert theme.pilier_id in PILIERS, (
            f"Thème {theme.id} → pilier_id inconnu : {theme.pilier_id}"
        )
    print(f"  ✓ {len(THEMES)} thèmes valides, tous référencent un pilier existant")


def test_format_id_exigence_valide():
    """Teste la validation du format d'identifiant ExigenceAtomique."""
    ids_valides = [
        "SII-P1-REG2015-Art75-§1",
        "SII-P2-DIR2009-Art44-§2a",
        "SII-P3-EIOPA-Art293-§1",
        "SII-P3-REG2015-Art298-§2",
    ]
    ids_invalides = [
        "P3-REG2015-Art290",       # Manque préfixe SII
        "SII-P4-REG2015-Art290",   # Pilier 4 inexistant
        "SII-P3-Art290",           # Manque source
        "sii-p3-reg2015-art290",   # Minuscules
    ]

    for id_valide in ids_valides:
        try:
            ExigenceAtomique(
                id_exigence=id_valide,
                theme_id="T08", source_id="REG_2015_35",
                article="Art. X",
                niveau_obligation=NiveauObligation.SHALL,
                texte_original="Texte de test suffisamment long pour passer la validation.",
                texte_normalise="Texte normalisé de test.",
                texte_verification="Le rapport mentionne-t-il ce point de test ?",
            )
            print(f"  ✓ ID valide accepté   : {id_valide}")
        except ValueError as e:
            print(f"  ✗ ID valide rejeté    : {id_valide} → {e}")
            raise

    for id_invalide in ids_invalides:
        try:
            ExigenceAtomique(
                id_exigence=id_invalide,
                theme_id="T08", source_id="REG_2015_35",
                article="Art. X",
                niveau_obligation=NiveauObligation.SHALL,
                texte_original="Texte de test suffisamment long pour passer la validation.",
                texte_normalise="Texte normalisé de test.",
                texte_verification="Le rapport mentionne-t-il ce point de test ?",
            )
            raise AssertionError(f"ID invalide accepté à tort : {id_invalide}")
        except ValueError:
            print(f"  ✓ ID invalide rejeté  : {id_invalide}")


def test_poids_obligation():
    """Vérifie les poids numériques des niveaux d'obligation."""
    exigence = ExigenceAtomique(
        id_exigence="SII-P3-REG2015-Art290-§test",
        theme_id="T08", source_id="REG_2015_35",
        article="Art. 290 test",
        niveau_obligation=NiveauObligation.SHALL,
        texte_original="Texte de test suffisamment long.",
        texte_normalise="Texte normalisé.",
        texte_verification="Question de vérification de test ?",
    )
    assert exigence.poids_obligation == 1.0
    exigence.niveau_obligation = NiveauObligation.SHOULD
    assert exigence.poids_obligation == 0.6
    exigence.niveau_obligation = NiveauObligation.MAY
    assert exigence.poids_obligation == 0.3
    print("  ✓ Poids SHALL=1.0, SHOULD=0.6, MAY=0.3")


def test_statut_conformite_depuis_score():
    """Teste la classification du statut de conformité."""
    cas = [
        (0.90, StatutConformite.COUVERT),
        (0.75, StatutConformite.COUVERT),
        (0.74, StatutConformite.PARTIEL),
        (0.45, StatutConformite.PARTIEL),
        (0.44, StatutConformite.ABSENT),
        (0.00, StatutConformite.ABSENT),
    ]
    for score, statut_attendu in cas:
        statut = ResultatMatching.calculer_statut(score)
        assert statut == statut_attendu, (
            f"Score {score} → statut {statut.value} ≠ attendu {statut_attendu.value}"
        )
    print("  ✓ Classification statuts : COUVERT≥0.75, PARTIEL≥0.45, ABSENT<0.45")


def test_niveau_conformite_global():
    """Teste la correspondance score → niveau A/B/C/D."""
    from src.ingestion.models import RapportConformite
    cas = [
        (0.85, NiveauConformiteGlobal.A),
        (0.80, NiveauConformiteGlobal.A),
        (0.79, NiveauConformiteGlobal.B),
        (0.65, NiveauConformiteGlobal.B),
        (0.64, NiveauConformiteGlobal.C),
        (0.45, NiveauConformiteGlobal.C),
        (0.44, NiveauConformiteGlobal.D),
        (0.00, NiveauConformiteGlobal.D),
    ]
    for score, niveau_attendu in cas:
        niveau = RapportConformite.niveau_depuis_score(score)
        assert niveau == niveau_attendu, (
            f"Score {score} → niveau {niveau.value} ≠ attendu {niveau_attendu.value}"
        )
    print("  ✓ Niveaux : A≥0.80, B≥0.65, C≥0.45, D<0.45")


def test_chargement_referentiel():
    """Charge le référentiel depuis les fichiers JSON et vérifie la structure."""
    loader = ReferentielLoader()
    loader.charger(strict=False)

    stats = loader.stats()
    print(f"  ✓ Référentiel chargé : {stats['total_exigences']} exigences")

    exigences_t08 = loader.get_exigences_par_theme("T08")
    if exigences_t08:
        print(f"  ✓ Thème T08 : {len(exigences_t08)} exigences chargées")
        # Vérifier que toutes les exigences SHALL ont un texte_verification non vide
        shall = [e for e in exigences_t08 if e.niveau_obligation == NiveauObligation.SHALL]
        for e in shall:
            assert len(e.texte_verification) > 20, (
                f"texte_verification trop court pour {e.id_exigence}"
            )
        print(f"  ✓ {len(shall)} exigences SHALL avec texte_verification valide")

    textes = loader.get_textes_verification()
    print(f"  ✓ {len(textes)} textes de vérification prêts pour l'embedding")


def test_serialisation_chroma_metadata():
    """Vérifie que la sérialisation pour ChromaDB ne contient que des types scalaires."""
    import json as json_mod
    exigence = ExigenceAtomique(
        id_exigence="SII-P3-REG2015-Art290-§test2",
        theme_id="T08", source_id="REG_2015_35",
        article="Art. 290 test",
        niveau_obligation=NiveauObligation.SHALL,
        texte_original="Texte original de test suffisamment long pour valider.",
        texte_normalise="Texte normalisé de test.",
        texte_verification="Le rapport contient-il ce point de vérification de test ?",
        mots_cles=["test", "validation"],
        entites_concernees=["entreprise_assurance"],
    )
    meta = exigence.to_chroma_metadata()

    # ChromaDB n'accepte que str, int, float, bool
    types_acceptes = (str, int, float, bool)
    for k, v in meta.items():
        assert isinstance(v, types_acceptes), (
            f"Métadonnée '{k}' de type {type(v).__name__} non accepté par ChromaDB"
        )
    # Les listes doivent être sérialisées en JSON string
    mots_cles = json_mod.loads(meta["mots_cles"])
    assert isinstance(mots_cles, list)
    print(f"  ✓ Métadonnées ChromaDB : {len(meta)} champs, tous types scalaires")


def run_all_tests():
    tests = [
        ("Poids des piliers", test_poids_piliers),
        ("Poids des thèmes par pilier", test_poids_themes_par_pilier),
        ("Cohérence thèmes → piliers", test_coherence_themes_piliers),
        ("Format id_exigence", test_format_id_exigence_valide),
        ("Poids d'obligation (SHALL/SHOULD/MAY)", test_poids_obligation),
        ("Statut conformité depuis score", test_statut_conformite_depuis_score),
        ("Niveau conformité global (A/B/C/D)", test_niveau_conformite_global),
        ("Chargement référentiel JSON", test_chargement_referentiel),
        ("Sérialisation ChromaDB metadata", test_serialisation_chroma_metadata),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION DU RÉFÉRENTIEL SOLVABILITÉ II")
    print("=" * 60)

    nb_ok = 0
    nb_ko = 0
    for nom, test_fn in tests:
        print(f"\n[TEST] {nom}")
        try:
            test_fn()
            nb_ok += 1
        except Exception as e:
            print(f"  ✗ ÉCHEC : {e}")
            nb_ko += 1

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)

    if nb_ko > 0:
        sys.exit(1)
    else:
        print("\nRéférentiel valide — prêt pour la Phase 2 (ingestion & chunking)")


if __name__ == "__main__":
    run_all_tests()
