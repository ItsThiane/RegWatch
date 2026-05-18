"""
tests/test_phase5.py — Validation complète Phase 5
===================================================
Teste :
  - ScoringEngine : calculs par niveau, pondérations, agrégations
  - Propriétés mathématiques : bornes, monotonicité, pondérations
  - Recommandations : pertinence selon le niveau de conformité
  - Export JSON/CSV : structure et contenu
  - Pipeline E2E : RAG → scoring → rapport complet

Exécution : python tests/test_phase5.py
"""

import sys
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.referentiel_loader import ReferentielLoader
from src.ingestion.referentiel_solvabilite2 import PILIERS, THEMES
from src.ingestion.models import (
    ExigenceAtomique, ResultatMatching,
    StatutConformite, NiveauObligation,
    RapportConformite, NiveauConformiteGlobal,
)
from src.scoring.scoring_engine import ScoringEngine, ScoreExigence
from src.scoring.rapport_exporter import RapportExporter
from src.rag.rag_pipeline import ResultatAnalyseSFCR

OUTPUT_TEST_DIR = Path("/tmp/regwatch_test_phase5_exports")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _loader() -> ReferentielLoader:
    loader = ReferentielLoader()
    loader.charger(strict=True)
    return loader


def _resultat_rag_synthetique(
    loader: ReferentielLoader,
    score_uniforme: float = 0.5,
) -> ResultatAnalyseSFCR:
    """
    Crée un ResultatAnalyseSFCR synthétique avec un score uniforme
    pour toutes les exigences — utile pour tester les propriétés
    mathématiques du scoring.
    """
    exigences = loader.get_toutes_exigences()
    matchings = [
        ResultatMatching(
            id=str(uuid.uuid4()),
            id_exigence=e.id_exigence,
            chunk_sfcr_id=str(uuid.uuid4()),
            score_similarite=score_uniforme,
            score_llm=score_uniforme,
            score_final=score_uniforme,
            statut=ResultatMatching.calculer_statut(score_uniforme),
            passage_trouve=f"Passage test pour {e.id_exigence}",
            justification_llm="Justification synthétique pour test.",
        )
        for e in exigences
    ]

    from src.rag.rag_pipeline import ResultatAnalyseExigence
    resultats_exigences = [
        ResultatAnalyseExigence(
            exigence=e,
            matching=m,
            candidats_evalues=[],
        )
        for e, m in zip(exigences, matchings)
    ]

    return ResultatAnalyseSFCR(
        doc_id=str(uuid.uuid4()),
        entreprise="Test SA",
        annee=2023,
        resultats_exigences=resultats_exigences,
        nb_exigences_analysees=len(exigences),
    )


def _resultat_rag_par_statut(
    loader: ReferentielLoader,
    statut_par_defaut: StatutConformite = StatutConformite.ABSENT,
    overrides: dict = None,
) -> ResultatAnalyseSFCR:
    """
    Crée un ResultatAnalyseSFCR avec des statuts configurables par exigence.
    overrides = {id_exigence: score_final}
    """
    overrides = overrides or {}
    score_map = {
        StatutConformite.COUVERT: 0.85,
        StatutConformite.PARTIEL: 0.60,
        StatutConformite.ABSENT:  0.20,
    }
    score_defaut = score_map[statut_par_defaut]
    exigences = loader.get_toutes_exigences()

    from src.rag.rag_pipeline import ResultatAnalyseExigence
    resultats = []
    for e in exigences:
        score = overrides.get(e.id_exigence, score_defaut)
        statut = ResultatMatching.calculer_statut(score)
        m = ResultatMatching(
            id=str(uuid.uuid4()),
            id_exigence=e.id_exigence,
            chunk_sfcr_id="",
            score_similarite=score,
            score_llm=score,
            score_final=score,
            statut=statut,
            passage_trouve="Passage test." if score > 0.2 else "",
            justification_llm="Justification test.",
        )
        resultats.append(ResultatAnalyseExigence(
            exigence=e, matching=m, candidats_evalues=[]
        ))

    return ResultatAnalyseSFCR(
        doc_id=str(uuid.uuid4()),
        entreprise="Test SA",
        annee=2023,
        resultats_exigences=resultats,
        nb_exigences_analysees=len(exigences),
    )


# ---------------------------------------------------------------------------
# Tests ScoringEngine — propriétés mathématiques
# ---------------------------------------------------------------------------

def test_score_borne_entre_0_et_1():
    """Le score global est toujours dans [0, 1]."""
    loader = _loader()
    engine = ScoringEngine(loader)

    for score_test in [0.0, 0.3, 0.5, 0.75, 1.0]:
        res = _resultat_rag_synthetique(loader, score_test)
        rapport = engine.calculer(res)
        assert 0.0 <= rapport.score_global <= 1.0, (
            f"Score global hors [0,1] : {rapport.score_global} "
            f"pour score_uniforme={score_test}"
        )

    print("  ✓ Score global ∈ [0, 1] pour tous les scores d'entrée")


def test_score_uniforme_0_donne_niveau_D():
    """
    Si toutes les exigences ont un score 0, le rapport doit être
    niveau D (Non conforme).
    """
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.0)
    rapport = engine.calculer(res)

    assert rapport.score_global == 0.0
    assert rapport.niveau_conformite == NiveauConformiteGlobal.D
    assert rapport.score_pilier_1 == 0.0
    assert rapport.score_pilier_2 == 0.0
    assert rapport.score_pilier_3 == 0.0
    print(f"  ✓ Score=0.0 → Niveau D | score_global={rapport.score_global}")


def test_score_uniforme_1_donne_niveau_A():
    """
    Si toutes les exigences ont un score 1, le rapport doit être
    niveau A (Conforme).
    """
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=1.0)
    rapport = engine.calculer(res)

    assert rapport.score_global > 0.95, (
        f"Score global trop bas pour score=1.0 : {rapport.score_global}"
    )
    assert rapport.niveau_conformite == NiveauConformiteGlobal.A
    print(f"  ✓ Score=1.0 → Niveau A | score_global={rapport.score_global:.4f}")


def test_monotonie_score_global():
    """
    Un meilleur score RAG doit produire un meilleur score global.
    Propriété fondamentale de cohérence du moteur de scoring.
    """
    loader = _loader()
    engine = ScoringEngine(loader)

    scores_testes = [0.0, 0.25, 0.50, 0.75, 1.0]
    scores_globaux = []
    for s in scores_testes:
        res = _resultat_rag_synthetique(loader, s)
        rapport = engine.calculer(res)
        scores_globaux.append(rapport.score_global)

    for i in range(len(scores_globaux) - 1):
        assert scores_globaux[i] <= scores_globaux[i+1], (
            f"Monotonie violée : score_rag={scores_testes[i]} → "
            f"score_global={scores_globaux[i]:.4f} > "
            f"score_rag={scores_testes[i+1]} → "
            f"score_global={scores_globaux[i+1]:.4f}"
        )

    print(f"  ✓ Monotonie : {[round(s, 3) for s in scores_globaux]}")


def test_poids_piliers_somment_a_1():
    """
    La somme des poids × scores piliers doit reconstruire le score global.
    Vérifie la cohérence de l'agrégation.
    """
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.6)
    rapport = engine.calculer(res)

    score_reconstruit = (
        rapport.score_pilier_1 * PILIERS["P1"].poids_scoring +
        rapport.score_pilier_2 * PILIERS["P2"].poids_scoring +
        rapport.score_pilier_3 * PILIERS["P3"].poids_scoring
    )
    assert abs(score_reconstruit - rapport.score_global) < 0.01, (
        f"Score reconstruit ({score_reconstruit:.4f}) ≠ "
        f"score global ({rapport.score_global:.4f})"
    )
    print(
        f"  ✓ Reconstruction score global : "
        f"Σ(pilier × poids) = {score_reconstruit:.4f} ≈ {rapport.score_global:.4f}"
    )


def test_shall_pese_plus_que_should():
    """
    Un SHALL manquant doit pénaliser plus qu'un SHOULD manquant.
    Valide la pondération SHALL=1.0 > SHOULD=0.6.
    """
    loader = _loader()
    engine = ScoringEngine(loader)

    # Cas 1 : tous ABSENT sauf les SHOULD qui sont COUVERT
    exigences = loader.get_toutes_exigences()
    shall_ids = {e.id_exigence for e in exigences
                 if e.niveau_obligation == NiveauObligation.SHALL}
    should_ids = {e.id_exigence for e in exigences
                  if e.niveau_obligation == NiveauObligation.SHOULD}

    # Scénario A : SHALLs couverts, SHOULDs absents
    overrides_A = {id_e: 0.85 for id_e in shall_ids}
    overrides_A.update({id_e: 0.20 for id_e in should_ids})
    res_A = _resultat_rag_par_statut(loader, StatutConformite.ABSENT, overrides_A)
    rapport_A = engine.calculer(res_A)

    # Scénario B : SHALLs absents, SHOULDs couverts
    overrides_B = {id_e: 0.20 for id_e in shall_ids}
    overrides_B.update({id_e: 0.85 for id_e in should_ids})
    res_B = _resultat_rag_par_statut(loader, StatutConformite.ABSENT, overrides_B)
    rapport_B = engine.calculer(res_B)

    assert rapport_A.score_global > rapport_B.score_global, (
        f"Pondération erronée : SHALL couverts ({rapport_A.score_global:.3f}) "
        f"≤ SHALL absents ({rapport_B.score_global:.3f})"
    )
    print(
        f"  ✓ Pondération : SHALL couverts → {rapport_A.score_global:.3f} > "
        f"SHALL absents → {rapport_B.score_global:.3f}"
    )


def test_niveaux_notation_seuils():
    """Les seuils A/B/C/D correspondent aux scores attendus."""
    cas = [
        (0.82, NiveauConformiteGlobal.A),
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
            f"Score {score} → {niveau.value} ≠ {niveau_attendu.value}"
        )
    print("  ✓ Seuils : A≥0.80 | B≥0.65 | C≥0.45 | D<0.45")


def test_statistiques_rapport_coherentes():
    """Les statistiques du rapport sont cohérentes avec les matchings."""
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.60)
    rapport = engine.calculer(res)

    total = (rapport.nb_exigences_couvertes +
             rapport.nb_exigences_partielles +
             rapport.nb_exigences_absentes)
    assert total == rapport.nb_exigences_analysees, (
        f"Statistiques incohérentes : C+P+A={total} ≠ "
        f"analysées={rapport.nb_exigences_analysees}"
    )
    assert rapport.nb_exigences_analysees == len(loader.get_toutes_exigences())
    print(
        f"  ✓ Statistiques cohérentes : "
        f"C={rapport.nb_exigences_couvertes} + "
        f"P={rapport.nb_exigences_partielles} + "
        f"A={rapport.nb_exigences_absentes} = {total}"
    )


# ---------------------------------------------------------------------------
# Tests recommandations
# ---------------------------------------------------------------------------

def test_recommandations_niveau_D():
    """Un niveau D doit générer des recommandations critiques."""
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.10)
    rapport = engine.calculer(res)

    assert len(rapport.recommandations) > 0
    texte = " ".join(rapport.recommandations)
    assert "CRITIQUE" in texte or "SHALL" in texte or "INSUFFISANT" in texte
    print(
        f"  ✓ Niveau D : {len(rapport.recommandations)} recommandation(s) "
        f"dont critiques"
    )


def test_recommandations_niveau_A():
    """Un niveau A doit générer des recommandations positives."""
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.90)
    rapport = engine.calculer(res)

    assert len(rapport.recommandations) > 0
    # Pas de recommandation CRITIQUE pour un niveau A
    texte = " ".join(rapport.recommandations)
    assert "CRITIQUE" not in texte
    print(
        f"  ✓ Niveau A : {len(rapport.recommandations)} recommandation(s) "
        f"positives (pas de critique)"
    )


def test_recommandations_priorites_ordonnees():
    """
    Les recommandations critiques (SHALL absents) doivent apparaître
    avant les recommandations mineures.
    """
    loader = _loader()
    engine = ScoringEngine(loader)
    # Score bas → beaucoup de SHALL absents
    res = _resultat_rag_synthetique(loader, score_uniforme=0.20)
    rapport = engine.calculer(res)

    if len(rapport.recommandations) >= 2:
        premiere = rapport.recommandations[0]
        assert "CRITIQUE" in premiere or "SHALL" in premiere, (
            f"La première recommandation devrait être critique : {premiere[:80]}"
        )
    print("  ✓ Priorité recommandations : CRITIQUE en premier")


# ---------------------------------------------------------------------------
# Tests formatage et export
# ---------------------------------------------------------------------------

def test_formater_rapport_verbose():
    """Le rapport formaté contient toutes les sections attendues."""
    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.55)
    rapport = engine.calculer(res)

    # Recalculer les scores intermédiaires
    scores_exig = engine._calculer_scores_exigences(res.matchings)
    scores_themes = engine._calculer_scores_themes(scores_exig)
    scores_piliers = engine._calculer_scores_piliers(scores_themes)

    texte = engine.formater_rapport(rapport, scores_themes, verbose=True)

    sections_attendues = [
        "RAPPORT DE CONFORMITÉ",
        "SCORE GLOBAL",
        "Pilier 1", "Pilier 2", "Pilier 3",
        "DÉTAIL PAR THÈME",
        "RECOMMANDATIONS",
    ]
    for section in sections_attendues:
        assert section in texte, f"Section manquante : '{section}'"

    print(f"  ✓ Rapport formaté : {len(texte)} chars, toutes sections présentes")


def test_export_json():
    """L'export JSON produit un fichier valide avec la bonne structure."""
    import json

    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.60)
    rapport = engine.calculer(res)

    scores_exig = engine._calculer_scores_exigences(res.matchings)
    scores_themes = engine._calculer_scores_themes(scores_exig)
    scores_piliers = engine._calculer_scores_piliers(scores_themes)

    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    exporter = RapportExporter(output_dir=OUTPUT_TEST_DIR)
    chemin = exporter.exporter_json(
        rapport, scores_exig, scores_themes, scores_piliers,
        nom_fichier="test_rapport.json"
    )

    assert chemin.exists()
    with open(chemin) as f:
        data = json.load(f)

    cles_attendues = [
        "metadata", "score_global", "niveau_conformite",
        "statistiques", "piliers", "themes", "exigences", "recommandations"
    ]
    for cle in cles_attendues:
        assert cle in data, f"Clé manquante dans JSON : {cle}"

    assert data["score_global"] == rapport.score_global
    assert data["niveau_conformite"] == rapport.niveau_conformite.value
    assert len(data["exigences"]) == rapport.nb_exigences_analysees
    assert len(data["piliers"]) == 3

    print(
        f"  ✓ Export JSON : {chemin.name} "
        f"({chemin.stat().st_size:,} bytes, "
        f"{len(data['exigences'])} exigences)"
    )


def test_export_csv():
    """L'export CSV produit un fichier valide avec les bonnes colonnes."""
    import csv

    loader = _loader()
    engine = ScoringEngine(loader)
    res = _resultat_rag_synthetique(loader, score_uniforme=0.55)
    rapport = engine.calculer(res)

    scores_exig = engine._calculer_scores_exigences(res.matchings)
    scores_themes = engine._calculer_scores_themes(scores_exig)

    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    exporter = RapportExporter(output_dir=OUTPUT_TEST_DIR)
    chemin = exporter.exporter_csv(
        scores_exig, scores_themes, rapport,
        nom_fichier="test_conformite.csv"
    )

    assert chemin.exists()
    with open(chemin, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        lignes = list(reader)

    assert len(lignes) == len(scores_exig)
    colonnes_attendues = [
        "id_exigence", "theme_id", "niveau_obligation",
        "statut", "score_brut", "justification"
    ]
    for col in colonnes_attendues:
        assert col in lignes[0], f"Colonne manquante : {col}"

    print(
        f"  ✓ Export CSV : {chemin.name} "
        f"({len(lignes)} lignes, {len(lignes[0])} colonnes)"
    )


# ---------------------------------------------------------------------------
# Test E2E complet : RAG → Scoring → Export
# ---------------------------------------------------------------------------

def test_pipeline_e2e_complet():
    """
    Test end-to-end Phase 1→5 :
    Référentiel → Embeddings → ChromaDB → RAG → Scoring → Export
    """
    import shutil
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore
    from src.rag.llm_engine import RuleBasedLLM
    from src.rag.rag_pipeline import RAGPipeline
    from src.ingestion.models import ChunkDocument
    from src.ingestion.text_utils import compter_tokens

    loader = _loader()
    exigences = loader.get_toutes_exigences()

    # Embedding engine
    textes_fit = [e.texte_normalise for e in exigences]
    engine_emb = LocalTfidfEmbedder(dimension=256)
    engine_emb.fit(textes_fit)

    # VectorStore
    vs_dir = Path("/tmp/regwatch_e2e_p5")
    if vs_dir.exists():
        shutil.rmtree(vs_dir)
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine_emb)

    # Indexer référentiel
    embs_ref = engine_emb.embed_documents(
        [e.texte_verification for e in exigences]
    )
    store.indexer_referentiel(exigences, embs_ref.tolist())

    # SFCR synthétique riche (couvre explicitement plusieurs thèmes)
    sfcr_riche = [
        ("Le conseil d'administration se réunit 4 fois par an. "
         "Les quatre fonctions clés Solvabilité II sont opérationnelles : "
         "gestion des risques, conformité, audit interne, actuarielle.",
         "B_GOUVERNANCE"),
        ("Système de contrôle interne : procédures administratives et comptables "
         "documentées. Cadre à 3 lignes de défense. Reporting mensuel direction.",
         "B_GOUVERNANCE"),
        ("ORSA 2023 : besoins globaux de solvabilité évalués. Profil de risque "
         "spécifique analysé. SCR couvert dans scénarios adverses.",
         "C_PROFIL_RISQUE"),
        ("SCR formule standard : 312 M€. Fonds propres : 589 M€. "
         "Ratio couverture SCR : 189%. MCR : 78 M€.",
         "E_GESTION_CAPITAL"),
        ("Provisions techniques Best Estimate : 2,1 Md€. Risk Margin : 45 M€. "
         "Valorisation IFRS avec retraitements Solvabilité II.",
         "D_VALORISATION"),
        ("SFCR publié le 25 mars 2024, dans le délai de 14 semaines. "
         "QRT S.02.01, S.05.01, S.25.01 publiés en annexe.",
         "E_GESTION_CAPITAL"),
        ("La politique de rémunération distingue composante fixe et variable "
         "pour dirigeants et fonctions clés. Évaluations fit & proper réalisées.",
         "B_GOUVERNANCE"),
    ]

    doc_id = str(uuid.uuid4())
    chunks = [ChunkDocument(
        chunk_id=str(uuid.uuid4()), doc_id=doc_id, type_doc="SFCR",
        texte=t, page_debut=i+1, page_fin=i+1, section=s,
        position_dans_doc=i, nb_tokens=compter_tokens(t)
    ) for i, (t, s) in enumerate(sfcr_riche)]

    embs_sfcr = engine_emb.embed_documents([c.texte for c in chunks])
    store.indexer_sfcr(chunks, embs_sfcr.tolist(), "Groupe Test SA", 2023)

    # RAG
    llm = RuleBasedLLM()
    rag = RAGPipeline(engine_emb, store, llm, top_k=3)
    resultat_rag = rag.analyser_sfcr(
        exigences=exigences,
        doc_id=doc_id,
        entreprise="Groupe Test SA",
        annee=2023,
    )
    assert resultat_rag.nb_erreurs == 0

    # Scoring
    scoring = ScoringEngine(loader)
    rapport = scoring.calculer(resultat_rag)

    assert 0.0 <= rapport.score_global <= 1.0
    assert rapport.niveau_conformite in list(NiveauConformiteGlobal)
    assert rapport.nb_exigences_analysees == len(exigences)

    # Export
    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)
    scores_exig = scoring._calculer_scores_exigences(resultat_rag.matchings)
    scores_themes = scoring._calculer_scores_themes(scores_exig)
    scores_piliers = scoring._calculer_scores_piliers(scores_themes)

    exporter = RapportExporter(output_dir=OUTPUT_TEST_DIR)
    chemin_json = exporter.exporter_json(
        rapport, scores_exig, scores_themes, scores_piliers,
        nom_fichier="e2e_rapport.json"
    )
    chemin_csv = exporter.exporter_csv(
        scores_exig, scores_themes, rapport,
        nom_fichier="e2e_conformite.csv"
    )

    assert chemin_json.exists()
    assert chemin_csv.exists()

    # Rapport formaté
    texte_rapport = scoring.formater_rapport(rapport, scores_themes, verbose=True)
    assert len(texte_rapport) > 500

    shutil.rmtree(vs_dir)

    print(
        f"  ✓ Pipeline E2E Phases 1→5 :\n"
        f"    {len(exigences)} exigences | score_global={rapport.score_global:.3f} "
        f"| niveau={rapport.niveau_conformite.value}\n"
        f"    C={rapport.nb_exigences_couvertes} "
        f"P={rapport.nb_exigences_partielles} "
        f"A={rapport.nb_exigences_absentes}\n"
        f"    P1={rapport.score_pilier_1:.3f} "
        f"P2={rapport.score_pilier_2:.3f} "
        f"P3={rapport.score_pilier_3:.3f}\n"
        f"    JSON={chemin_json.name} | CSV={chemin_csv.name}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # Propriétés mathématiques
        ("Score global ∈ [0, 1]",                test_score_borne_entre_0_et_1),
        ("Score=0.0 → Niveau D",                 test_score_uniforme_0_donne_niveau_D),
        ("Score=1.0 → Niveau A",                 test_score_uniforme_1_donne_niveau_A),
        ("Monotonie du score global",            test_monotonie_score_global),
        ("Reconstruction score global",          test_poids_piliers_somment_a_1),
        ("SHALL pèse plus que SHOULD",           test_shall_pese_plus_que_should),
        ("Seuils A/B/C/D corrects",              test_niveaux_notation_seuils),
        ("Statistiques C+P+A = total",           test_statistiques_rapport_coherentes),
        # Recommandations
        ("Recommandations niveau D",             test_recommandations_niveau_D),
        ("Recommandations niveau A",             test_recommandations_niveau_A),
        ("Priorité recommandations",             test_recommandations_priorites_ordonnees),
        # Export
        ("Rapport formaté verbose",              test_formater_rapport_verbose),
        ("Export JSON structure",                test_export_json),
        ("Export CSV colonnes",                  test_export_csv),
        # E2E
        ("Pipeline E2E Phases 1→5",              test_pipeline_e2e_complet),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 5 — MOTEUR DE SCORING")
    print("=" * 60)

    nb_ok = nb_ko = 0
    for nom, fn in tests:
        print(f"\n[TEST] {nom}")
        try:
            fn()
            nb_ok += 1
        except Exception as e:
            import traceback
            print(f"  ✗ ÉCHEC : {e}")
            traceback.print_exc()
            nb_ko += 1

    if OUTPUT_TEST_DIR.exists():
        shutil.rmtree(OUTPUT_TEST_DIR)

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)
    if nb_ko == 0:
        print("\n✓ Phase 5 validée — prêt pour la Phase 6 (interface Streamlit)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
