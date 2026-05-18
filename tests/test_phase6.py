"""
tests/test_phase6.py — Validation Phase 6 (Interface Streamlit)
================================================================
On ne peut pas tester Streamlit avec un vrai navigateur dans cet
environnement. On valide donc :
  1. Syntaxe et imports de app.py
  2. Les fonctions utilitaires de l'interface
  3. La cohérence des données préparées pour l'affichage
  4. Les exports JSON/CSV depuis l'interface
  5. L'intégration complète Phase 1→6

Exécution : python tests/test_phase6.py
"""

import sys
import ast
import json
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_TEST_DIR = Path("/tmp/regwatch_test_phase6")


# ---------------------------------------------------------------------------
# Fixtures réutilisées depuis Phase 5
# ---------------------------------------------------------------------------

def _setup_complet():
    """
    Crée un résultat d'analyse complet pour les tests d'interface.
    Réutilise le pipeline Phase 1→5 avec un SFCR synthétique.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from src.ingestion.referentiel_loader import ReferentielLoader
    from src.ingestion.models import ChunkDocument
    from src.ingestion.text_utils import compter_tokens
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore
    from src.rag.llm_engine import RuleBasedLLM
    from src.rag.rag_pipeline import RAGPipeline
    from src.scoring.scoring_engine import ScoringEngine

    loader = ReferentielLoader()
    loader.charger()
    exigences = loader.get_toutes_exigences()

    textes_fit = [e.texte_normalise for e in exigences]
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(textes_fit)

    vs_dir = Path(f"/tmp/regwatch_p6_{uuid.uuid4().hex[:8]}")
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine)

    embs_ref = engine.embed_documents([e.texte_verification for e in exigences])
    store.indexer_referentiel(exigences, embs_ref.tolist())

    sfcr = [
        ("Les quatre fonctions clés Solvabilité II sont opérationnelles. "
         "Gouvernance robuste avec conseil d'administration indépendant.",
         "B_GOUVERNANCE"),
        ("SCR formule standard : 312 M€. Ratio couverture : 189%. "
         "Fonds propres éligibles : 589 M€. MCR couvert à 755%.",
         "E_GESTION_CAPITAL"),
        ("Best Estimate : 2,1 Md€. Risk Margin : 45 M€. "
         "Valorisation IFRS ajustée SII.",
         "D_VALORISATION"),
        ("SFCR publié dans le délai de 14 semaines réglementaire. "
         "QRT S.02.01 et S.25.01 annexés.",
         "A_ACTIVITE_RESULTATS"),
    ]

    doc_id = str(uuid.uuid4())
    chunks = [ChunkDocument(
        chunk_id=str(uuid.uuid4()), doc_id=doc_id, type_doc="SFCR",
        texte=t, page_debut=i+1, page_fin=i+1, section=s,
        position_dans_doc=i, nb_tokens=compter_tokens(t)
    ) for i, (t, s) in enumerate(sfcr)]

    embs = engine.embed_documents([c.texte for c in chunks])
    store.indexer_sfcr(chunks, embs.tolist(), "Test Interface SA", 2023)

    llm = RuleBasedLLM()
    rag = RAGPipeline(engine, store, llm, top_k=3)
    res_rag = rag.analyser_sfcr(exigences, doc_id, "Test Interface SA", 2023)

    scoring = ScoringEngine(loader)
    rapport = scoring.calculer(res_rag)
    scores_exig    = scoring._calculer_scores_exigences(res_rag.matchings)
    scores_themes  = scoring._calculer_scores_themes(scores_exig)
    scores_piliers = scoring._calculer_scores_piliers(scores_themes)

    return {
        "loader": loader,
        "rapport": rapport,
        "scores_exig": scores_exig,
        "scores_themes": scores_themes,
        "scores_piliers": scores_piliers,
        "res_rag": res_rag,
        "vs_dir": vs_dir,
    }


# ---------------------------------------------------------------------------
# Tests syntaxe et structure de app.py
# ---------------------------------------------------------------------------

def test_syntaxe_app_py():
    """app.py est syntaxiquement valide."""
    chemin = Path("app.py")
    assert chemin.exists(), "app.py introuvable — lancer depuis la racine du projet"
    with open(chemin) as f:
        source = f.read()
    try:
        ast.parse(source)
    except SyntaxError as e:
        raise AssertionError(f"Erreur de syntaxe dans app.py : {e}")
    print("  ✓ app.py syntaxiquement valide")


def test_imports_critiques_app():
    """Tous les imports critiques de app.py sont disponibles."""
    modules_requis = [
        "streamlit",
        "plotly.graph_objects",
        "plotly.express",
        "pandas",
        "src.ingestion.referentiel_loader",
        "src.ingestion.ingestion_pipeline",
        "src.rag.embedding_engine",
        "src.rag.vector_store",
        "src.rag.llm_engine",
        "src.rag.rag_pipeline",
        "src.scoring.scoring_engine",
        "src.scoring.rapport_exporter",
    ]
    manquants = []
    for mod in modules_requis:
        try:
            __import__(mod)
        except ImportError as e:
            manquants.append(f"{mod} : {e}")

    assert not manquants, (
        f"Modules manquants :\n" + "\n".join(f"  • {m}" for m in manquants)
    )
    print(f"  ✓ {len(modules_requis)} imports critiques disponibles")


def test_sections_app_presentes():
    """app.py contient toutes les sections UI attendues."""
    with open("app.py") as f:
        source = f.read()

    sections_attendues = [
        "st.set_page_config",      # Configuration page
        "st.file_uploader",        # Upload PDF
        "charger_referentiel",     # Chargement référentiel
        "initialiser_engine",      # Engine embedding
        "RAGPipeline",             # Pipeline RAG
        "ScoringEngine",           # Scoring
        "st.tabs",                 # Onglets
        "st.plotly_chart",         # Graphes
        "st.download_button",      # Export
        "st.dataframe",            # Tableau
        "st.expander",             # Drill-down
        ".metric",               # KPIs (st.metric ou col.metric)
    ]

    for section in sections_attendues:
        assert section in source, f"Section absente de app.py : '{section}'"

    print(f"  ✓ {len(sections_attendues)} sections UI présentes dans app.py")


# ---------------------------------------------------------------------------
# Tests des données préparées pour l'affichage
# ---------------------------------------------------------------------------

def test_donnees_kpis():
    """Les données KPI sont calculées correctement."""
    ctx = _setup_complet()
    rapport = ctx["rapport"]

    total = (
        rapport.nb_exigences_couvertes +
        rapport.nb_exigences_partielles +
        rapport.nb_exigences_absentes
    )
    assert total == rapport.nb_exigences_analysees
    assert 0.0 <= rapport.score_global <= 1.0
    assert rapport.niveau_conformite.value in ["A", "B", "C", "D"]

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ KPIs : score={rapport.score_global:.3f} | "
        f"niveau={rapport.niveau_conformite.value} | "
        f"C={rapport.nb_exigences_couvertes}/"
        f"P={rapport.nb_exigences_partielles}/"
        f"A={rapport.nb_exigences_absentes}"
    )


def test_donnees_barres_piliers():
    """Les données pour les barres piliers sont dans [0, 1]."""
    ctx = _setup_complet()
    rapport = ctx["rapport"]

    scores_piliers_vals = [
        rapport.score_pilier_1,
        rapport.score_pilier_2,
        rapport.score_pilier_3,
    ]
    for score in scores_piliers_vals:
        assert 0.0 <= score <= 1.0, f"Score pilier hors [0,1] : {score}"

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ Barres piliers : "
        f"P1={rapport.score_pilier_1:.3f} | "
        f"P2={rapport.score_pilier_2:.3f} | "
        f"P3={rapport.score_pilier_3:.3f}"
    )


def test_donnees_radar():
    """Les données pour le radar sont cohérentes (thèmes avec exigences)."""
    ctx = _setup_complet()
    scores_themes = ctx["scores_themes"]

    themes_actifs = [
        (tid, st_th)
        for tid, st_th in scores_themes.items()
        if st_th.nb_exigences_total > 0
    ]

    assert len(themes_actifs) == 11, (
        f"Nombre de thèmes actifs inattendu : {len(themes_actifs)}"
    )
    for tid, st_th in themes_actifs:
        assert 0.0 <= st_th.score_brut <= 1.0
        assert st_th.nb_exigences_total > 0

    shutil.rmtree(ctx["vs_dir"])
    print(f"  ✓ Radar : {len(themes_actifs)} thèmes actifs, scores ∈ [0, 1]")


def test_dataframe_themes():
    """Le DataFrame themes est constructible et correct."""
    import pandas as pd
    ctx = _setup_complet()
    scores_themes = ctx["scores_themes"]

    rows = []
    for tid, st_th in sorted(scores_themes.items()):
        if st_th.nb_exigences_total == 0:
            continue
        rows.append({
            "ID": tid,
            "Thème": st_th.libelle_theme,
            "Score": st_th.score_brut,
            "✅ Couvertes": st_th.nb_couvertes,
            "⚠️ Partielles": st_th.nb_partielles,
            "❌ Absentes": st_th.nb_absentes,
        })

    df = pd.DataFrame(rows)
    assert len(df) == 11
    assert "Score" in df.columns
    assert df["Score"].between(0, 1).all()
    assert (df["✅ Couvertes"] + df["⚠️ Partielles"] + df["❌ Absentes"]).equals(
        pd.Series([st_th.nb_exigences_total
                   for _, st_th in sorted(scores_themes.items())
                   if st_th.nb_exigences_total > 0])
    )

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ DataFrame thèmes : {len(df)} lignes | "
        f"scores valides | C+P+A = total"
    )


def test_drill_down_exigences():
    """Les données de drill-down sont accessibles par thème."""
    ctx = _setup_complet()
    scores_exig = ctx["scores_exig"]

    from src.ingestion.referentiel_solvabilite2 import THEMES
    for tid in THEMES:
        exig_theme = [se for se in scores_exig if se.theme_id == tid]
        for se in exig_theme:
            assert hasattr(se, "id_exigence")
            assert hasattr(se, "score_brut")
            assert hasattr(se, "statut")
            assert hasattr(se, "justification")
            assert hasattr(se, "passage_trouve")
            assert 0.0 <= se.score_brut <= 1.0

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ Drill-down : {len(scores_exig)} exigences avec "
        f"tous les champs d'affichage"
    )


# ---------------------------------------------------------------------------
# Tests exports depuis l'interface
# ---------------------------------------------------------------------------

def test_export_json_interface():
    """L'export JSON produit depuis l'interface est valide."""
    ctx = _setup_complet()
    rapport       = ctx["rapport"]
    scores_themes = ctx["scores_themes"]
    scores_exig   = ctx["scores_exig"]
    scores_piliers = ctx["scores_piliers"]

    # Simuler la construction du payload JSON de app.py
    payload = {
        "metadata": {
            "outil": "RegWatch",
            "version": "1.0",
            "entreprise": rapport.entreprise,
            "annee": rapport.annee_rapport,
            "date_analyse": rapport.date_analyse,
        },
        "score_global": rapport.score_global,
        "niveau_conformite": rapport.niveau_conformite.value,
        "statistiques": {
            "nb_analysees":  rapport.nb_exigences_analysees,
            "nb_couvertes":  rapport.nb_exigences_couvertes,
            "nb_partielles": rapport.nb_exigences_partielles,
            "nb_absentes":   rapport.nb_exigences_absentes,
        },
        "piliers": [
            {"id": pid, "score": sp.score_brut, "niveau": sp.niveau}
            for pid, sp in scores_piliers.items()
        ],
        "themes": [
            {
                "id": tid,
                "libelle": st_th.libelle_theme,
                "score": st_th.score_brut,
                "couvertes":  st_th.nb_couvertes,
                "partielles": st_th.nb_partielles,
                "absentes":   st_th.nb_absentes,
                "shall_critiques": st_th.exigences_critiques_absentes,
            }
            for tid, st_th in scores_themes.items()
            if st_th.nb_exigences_total > 0
        ],
        "exigences": [
            {
                "id": se.id_exigence,
                "theme": se.theme_id,
                "obligation": se.niveau_obligation.value,
                "statut": se.statut.value,
                "score": se.score_brut,
                "justification": se.justification,
            }
            for se in scores_exig
        ],
        "recommandations": rapport.recommandations,
    }

    # Sérialiser et désérialiser
    json_str = json.dumps(payload, ensure_ascii=False, indent=2)
    data = json.loads(json_str)

    assert data["score_global"] == rapport.score_global
    assert data["niveau_conformite"] == rapport.niveau_conformite.value
    assert len(data["exigences"]) == len(scores_exig)
    assert len(data["piliers"]) == 3
    assert len(data["themes"]) == 11

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ Export JSON interface : "
        f"{len(json_str):,} bytes | "
        f"{len(data['exigences'])} exigences | "
        f"{len(data['themes'])} thèmes"
    )


def test_export_csv_interface():
    """L'export CSV produit depuis l'interface est valide."""
    import pandas as pd
    ctx = _setup_complet()
    scores_exig   = ctx["scores_exig"]
    scores_themes = ctx["scores_themes"]
    rapport       = ctx["rapport"]

    rows_csv = []
    for se in sorted(scores_exig, key=lambda x: x.theme_id):
        st_th = scores_themes.get(se.theme_id)
        rows_csv.append({
            "id_exigence":      se.id_exigence,
            "theme_id":         se.theme_id,
            "libelle_theme":    st_th.libelle_theme if st_th else "",
            "niveau_obligation": se.niveau_obligation.value,
            "poids_obligation": se.poids_obligation,
            "statut":           se.statut.value,
            "score_brut":       round(se.score_brut, 4),
            "score_pondere":    round(se.score_pondere, 4),
            "score_theme":      round(st_th.score_brut, 4) if st_th else 0,
            "justification":    se.justification[:200],
        })

    df = pd.DataFrame(rows_csv)
    csv_str = df.to_csv(index=False, encoding="utf-8-sig")

    assert len(df) == len(scores_exig)
    assert "statut" in df.columns
    assert "score_brut" in df.columns
    assert df["score_brut"].between(0, 1).all()

    shutil.rmtree(ctx["vs_dir"])
    print(
        f"  ✓ Export CSV interface : "
        f"{len(df)} lignes | "
        f"{len(df.columns)} colonnes | "
        f"{len(csv_str):,} bytes"
    )


# ---------------------------------------------------------------------------
# Test E2E interface simulée
# ---------------------------------------------------------------------------

def test_e2e_interface_simulation():
    """
    Simule le flux complet de l'interface sans Streamlit :
    Upload → Ingestion → RAG → Scoring → KPIs → Export
    """
    import tempfile
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore
    from src.rag.llm_engine import creer_llm_engine
    from src.rag.rag_pipeline import RAGPipeline
    from src.scoring.scoring_engine import ScoringEngine
    from src.ingestion.referentiel_loader import ReferentielLoader

    # Simule un PDF minimal (non analysable mais testable)
    loader = ReferentielLoader()
    loader.charger()
    exigences = loader.get_toutes_exigences()

    textes_fit = [e.texte_normalise for e in exigences]
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(textes_fit)

    vs_dir = Path(f"/tmp/regwatch_e2e6_{uuid.uuid4().hex[:8]}")
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine)

    embs_ref = engine.embed_documents([e.texte_verification for e in exigences])
    store.indexer_referentiel(exigences, embs_ref.tolist())

    # Simuler session_state Streamlit avec un SFCR synthétique
    from src.ingestion.models import ChunkDocument
    from src.ingestion.text_utils import compter_tokens

    chunks_sim = [
        ChunkDocument(
            chunk_id=str(uuid.uuid4()),
            doc_id="doc-sim",
            type_doc="SFCR",
            texte=(
                "Gouvernance : conseil d'administration, quatre fonctions clés, "
                "audit interne indépendant, fonction actuarielle opérationnelle."
            ),
            page_debut=1, page_fin=1,
            section="B_GOUVERNANCE",
            position_dans_doc=0,
            nb_tokens=30,
        )
    ]
    embs_sim = engine.embed_documents([c.texte for c in chunks_sim])
    store.indexer_sfcr(chunks_sim, embs_sim.tolist(), "Interface Test SA", 2023)

    llm = creer_llm_engine("local")
    rag = RAGPipeline(engine, store, llm, top_k=3)
    res_rag = rag.analyser_sfcr(
        exigences=exigences,
        doc_id="doc-sim",
        entreprise="Interface Test SA",
        annee=2023,
    )

    scoring = ScoringEngine(loader)
    rapport = scoring.calculer(res_rag)
    scores_exig    = scoring._calculer_scores_exigences(res_rag.matchings)
    scores_themes  = scoring._calculer_scores_themes(scores_exig)
    scores_piliers = scoring._calculer_scores_piliers(scores_themes)

    # Vérifier que tout ce qu'affiche Streamlit est calculable
    assert rapport.score_global is not None
    assert rapport.niveau_conformite is not None
    assert len(scores_exig) == len(exigences)
    assert len(scores_themes) == 11
    assert len(scores_piliers) == 3
    assert len(rapport.recommandations) > 0

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ Simulation interface E2E : "
        f"score={rapport.score_global:.3f} | "
        f"niveau={rapport.niveau_conformite.value} | "
        f"{len(rapport.recommandations)} recommandation(s)"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # Syntaxe et structure
        ("Syntaxe app.py valide",                 test_syntaxe_app_py),
        ("Imports critiques disponibles",         test_imports_critiques_app),
        ("Sections UI présentes dans app.py",     test_sections_app_presentes),
        # Données d'affichage
        ("Données KPIs cohérentes",               test_donnees_kpis),
        ("Données barres piliers ∈ [0, 1]",       test_donnees_barres_piliers),
        ("Données radar 11 thèmes",               test_donnees_radar),
        ("DataFrame thèmes constructible",        test_dataframe_themes),
        ("Drill-down exigences complet",          test_drill_down_exigences),
        # Exports
        ("Export JSON interface",                 test_export_json_interface),
        ("Export CSV interface",                  test_export_csv_interface),
        # E2E
        ("Simulation E2E interface complète",     test_e2e_interface_simulation),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 6 — INTERFACE STREAMLIT")
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
        print("\n✓ Phase 6 validée — interface prête")
        print("\nPour lancer l'interface :")
        print("  streamlit run app.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
