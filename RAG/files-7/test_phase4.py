"""
tests/test_phase4.py — Validation complète Phase 4
===================================================
Teste :
  - LLMEngine : interface, RuleBasedLLM, parsing JSON, scores
  - RAGPipeline : retrieval, scoring combiné, cas limites
  - Intégration E2E : référentiel → ChromaDB → RAG → matchings

Exécution : python tests/test_phase4.py
"""

import sys
import shutil
import uuid
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.llm_engine import (
    RuleBasedLLM, AnthropicLLM, LLMEngine,
    AnalyseLLM, creer_llm_engine,
)
from src.rag.rag_pipeline import (
    RAGPipeline, ResultatAnalyseExigence, ResultatAnalyseSFCR,
    POIDS_COSINUS, POIDS_LLM,
)
from src.rag.embedding_engine import LocalTfidfEmbedder
from src.rag.vector_store import VectorStore
from src.ingestion.models import (
    ExigenceAtomique, NiveauObligation,
    StatutConformite, ResultatMatching,
)
from src.ingestion.text_utils import compter_tokens

VECTORSTORE_TEST_DIR = Path("/tmp/regwatch_test_phase4")

def _dir_test_unique() -> Path:
    """Crée un répertoire ChromaDB unique pour chaque test."""
    import uuid as _uuid2
    d = Path(f"/tmp/regwatch_p4_{_uuid2.uuid4().hex[:8]}")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _exigence_test(
    id_ex: str = "SII-P2-DIR2009-Art46-§1",
    theme: str = "T06",
    obligation: NiveauObligation = NiveauObligation.SHALL,
) -> ExigenceAtomique:
    return ExigenceAtomique(
        id_exigence=id_ex,
        theme_id=theme,
        source_id="DIR_2009_138",
        article="Art. 46 §1",
        niveau_obligation=obligation,
        texte_original=(
            "Les entreprises mettent en place un système de contrôle interne "
            "efficace comprenant des procédures administratives et comptables."
        ),
        texte_normalise=(
            "Système de contrôle interne effectif : procédures administratives, "
            "comptables, cadre de contrôle et reporting interne."
        ),
        texte_verification=(
            "Le rapport décrit-il le système de contrôle interne de l'entreprise, "
            "notamment les procédures administratives, comptables et le cadre de "
            "contrôle interne à tous les niveaux hiérarchiques ?"
        ),
        mots_cles=["contrôle interne", "procédures", "administratives"],
    )


def _creer_engine_et_store() -> tuple:
    """Crée un moteur d'embedding et un VectorStore de test (répertoire unique)."""
    corpus = [
        "Le système de contrôle interne comprend les procédures administratives "
        "et comptables ainsi que le cadre de contrôle à tous les niveaux.",
        "Les fonds propres éligibles couvrent le SCR et le MCR de solvabilité.",
        "Les provisions techniques Best Estimate sont calculées actuariellement.",
        "La gouvernance comprend le conseil d'administration et les comités.",
        "Le profil de risque couvre souscription marché crédit liquidité opérationnel.",
        "La fonction actuarielle émet un avis sur la politique de souscription.",
        "La valorisation prudentielle des actifs respecte les normes IFRS.",
        "Le SCR est calculé selon la formule standard avec les modules de risque.",
        "La politique de rémunération distingue composante fixe et variable.",
        "Le rapport SFCR est publié dans un délai de quatorze semaines.",
    ]
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(corpus)

    test_dir = _dir_test_unique()
    store = VectorStore(persist_dir=test_dir)
    store.initialiser(engine)
    return engine, store, corpus


def _indexer_sfcr_test(store, engine, corpus, doc_id="doc-test-p4") -> str:
    """Indexe un SFCR de test dans le VectorStore."""
    import uuid as _uuid
    from src.ingestion.models import ChunkDocument

    sections = [
        "B_GOUVERNANCE", "E_GESTION_CAPITAL", "D_VALORISATION",
        "B_GOUVERNANCE", "C_PROFIL_RISQUE", "B_GOUVERNANCE",
        "D_VALORISATION", "E_GESTION_CAPITAL", "B_GOUVERNANCE",
        "A_ACTIVITE_RESULTATS",
    ]
    chunks = [
        ChunkDocument(
            chunk_id=str(_uuid.uuid4()),
            doc_id=doc_id,
            type_doc="SFCR",
            texte=corpus[i],
            page_debut=i + 1,
            page_fin=i + 1,
            section=sections[i],
            position_dans_doc=i,
            nb_tokens=compter_tokens(corpus[i]),
        )
        for i in range(len(corpus))
    ]
    embeddings = engine.embed_documents([c.texte for c in chunks])
    store.indexer_sfcr(chunks, embeddings.tolist(), "Test SA", 2023)
    return doc_id


# ---------------------------------------------------------------------------
# Tests LLMEngine
# ---------------------------------------------------------------------------

def test_interface_abstraite_llm():
    """LLMEngine est bien abstraite."""
    try:
        LLMEngine()
        raise AssertionError("LLMEngine devrait être abstraite")
    except TypeError:
        pass
    print("  ✓ LLMEngine : classe abstraite correcte")


def test_analyse_llm_dataclass():
    """AnalyseLLM valide et contraint ses champs."""
    # Score capped entre 0 et 1
    a = AnalyseLLM(
        score=1.5,        # → sera réduit à 1.0
        justification="Test",
        elements_trouves=["gouvernance"],
        elements_manquants=[],
        confiance=2.0,    # → sera réduit à 1.0
    )
    assert a.score == 1.0
    assert a.confiance == 1.0

    b = AnalyseLLM(
        score=-0.3,       # → sera relevé à 0.0
        justification="Test négatif",
        elements_trouves=[],
        elements_manquants=["contrôle"],
        confiance=-0.1,   # → 0.0
    )
    assert b.score == 0.0
    assert b.confiance == 0.0
    print("  ✓ AnalyseLLM : contraintes score ∈ [0,1] et confiance ∈ [0,1]")


def test_rule_based_passage_pertinent():
    """
    RuleBasedLLM donne un score élevé pour un passage
    qui couvre clairement l'exigence.
    """
    llm = RuleBasedLLM()
    exigence = _exigence_test()

    passage_pertinent = (
        "L'entreprise a mis en place un système de contrôle interne robuste "
        "comprenant des procédures administratives et comptables strictes. "
        "Ce cadre de contrôle interne s'applique à tous les niveaux hiérarchiques "
        "et fait l'objet d'un reporting régulier à la direction générale. "
        "La fonction de vérification de la conformité supervise l'ensemble "
        "du dispositif de contrôle permanent."
    )

    analyse = llm.analyser_conformite(
        texte_verification=exigence.texte_verification,
        passage_sfcr=passage_pertinent,
        id_exigence=exigence.id_exigence,
        niveau_obligation="SHALL",
    )

    assert analyse.score >= 0.50, (
        f"Score trop bas pour passage pertinent : {analyse.score}"
    )
    assert len(analyse.elements_trouves) > 0
    assert analyse.confiance > 0.3
    print(
        f"  ✓ RuleBasedLLM passage pertinent : score={analyse.score:.3f}, "
        f"trouvés={analyse.elements_trouves[:3]}"
    )


def test_rule_based_passage_hors_sujet():
    """
    RuleBasedLLM donne un score faible pour un passage
    qui ne couvre pas l'exigence.
    """
    llm = RuleBasedLLM()
    exigence = _exigence_test()

    passage_hors_sujet = (
        "Les primes brutes émises ont augmenté de 3,2% par rapport à l'exercice "
        "précédent pour atteindre 1,2 milliard d'euros. Cette performance reflète "
        "la dynamique commerciale positive sur l'ensemble des marchés géographiques."
    )

    analyse = llm.analyser_conformite(
        texte_verification=exigence.texte_verification,
        passage_sfcr=passage_hors_sujet,
        id_exigence=exigence.id_exigence,
        niveau_obligation="SHALL",
    )

    assert analyse.score < 0.50, (
        f"Score trop élevé pour passage hors sujet : {analyse.score}"
    )
    print(
        f"  ✓ RuleBasedLLM passage hors sujet : score={analyse.score:.3f} "
        f"(< 0.50 attendu)"
    )


def test_rule_based_coherence_relative():
    """
    Le score passage pertinent doit être strictement supérieur
    au score passage hors sujet — test de cohérence relative.
    """
    llm = RuleBasedLLM()
    exigence = _exigence_test()

    passage_bon = (
        "Le système de contrôle interne de l'entreprise repose sur des "
        "procédures administratives et comptables documentées, un cadre de "
        "contrôle interne validé annuellement et un reporting structuré à "
        "tous les niveaux de l'organisation."
    )
    passage_mauvais = (
        "Le ratio de couverture du SCR s'établit à 187% au 31 décembre 2023, "
        "témoignant de la solidité financière de l'entreprise."
    )

    score_bon     = llm.analyser_conformite(exigence.texte_verification,
                                            passage_bon,      exigence.id_exigence).score
    score_mauvais = llm.analyser_conformite(exigence.texte_verification,
                                            passage_mauvais,  exigence.id_exigence).score

    assert score_bon > score_mauvais, (
        f"Incohérence : score_bon={score_bon:.3f} ≤ score_mauvais={score_mauvais:.3f}"
    )
    print(
        f"  ✓ Cohérence relative : pertinent={score_bon:.3f} > "
        f"hors-sujet={score_mauvais:.3f}"
    )


def test_rule_based_backend_nom():
    """Le backend a un nom non vide."""
    llm = RuleBasedLLM()
    assert llm.nom_backend
    assert "offline" in llm.nom_backend.lower() or "rule" in llm.nom_backend.lower()
    print(f"  ✓ nom_backend : '{llm.nom_backend}'")


def test_factory_llm_mode_local():
    """La factory crée bien un RuleBasedLLM en mode local."""
    llm = creer_llm_engine(mode="local")
    assert isinstance(llm, RuleBasedLLM)
    print("  ✓ Factory LLM mode='local' → RuleBasedLLM")


def test_factory_llm_mode_auto_sans_cle():
    """Sans clé API et sans Ollama, le mode auto tombe sur RuleBasedLLM."""
    import os
    # S'assurer qu'il n'y a pas de clé dans l'environnement
    old_key = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        llm = creer_llm_engine(mode="auto")
        assert isinstance(llm, RuleBasedLLM)
        print("  ✓ Factory LLM mode='auto' sans clé → RuleBasedLLM")
    finally:
        if old_key:
            os.environ["ANTHROPIC_API_KEY"] = old_key


# ---------------------------------------------------------------------------
# Tests RAGPipeline
# ---------------------------------------------------------------------------

def test_rag_pipeline_initialisation():
    """Le RAGPipeline vérifie la somme des poids à l'initialisation."""
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()

    # Poids valides
    pipeline = RAGPipeline(engine, store, llm,
                           poids_cosinus=0.4, poids_llm=0.6)
    assert pipeline.poids_cosinus == 0.4
    assert pipeline.poids_llm == 0.6

    # Poids invalides → AssertionError
    try:
        RAGPipeline(engine, store, llm, poids_cosinus=0.5, poids_llm=0.3)
        raise AssertionError("Devrait lever une AssertionError")
    except AssertionError:
        pass

    print("  ✓ RAGPipeline : validation poids cosinus+llm=1.0")


def test_calcul_score_final():
    """Le calcul du score final combine correctement les deux signaux."""
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    pipeline = RAGPipeline(engine, store, llm)

    # Cas 1 : score_cosinus=0.8, score_llm=0.9
    # → 0.4×0.8 + 0.6×0.9 = 0.32 + 0.54 = 0.86
    score = pipeline._calculer_score_final(0.8, 0.9)
    assert abs(score - 0.86) < 0.01, f"Score inattendu : {score} (attendu ~0.86)"

    # Cas 2 : score_cosinus=0.9, score_llm=0.1
    # → 0.4×0.9 + 0.6×0.1 = 0.36 + 0.06 = 0.42
    score2 = pipeline._calculer_score_final(0.9, 0.1)
    assert abs(score2 - 0.42) < 0.01, f"Score inattendu : {score2} (attendu ~0.42)"

    # Cas 3 : les deux à 0 → 0
    score3 = pipeline._calculer_score_final(0.0, 0.0)
    assert score3 == 0.0

    print(
        f"  ✓ Score final : "
        f"(0.8cos,0.9llm)={score:.2f} | "
        f"(0.9cos,0.1llm)={score2:.2f} | "
        f"(0,0)={score3:.2f}"
    )


def test_mapping_theme_section():
    """Le mapping thème → section SFCR est cohérent."""
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    pipeline = RAGPipeline(engine, store, llm)

    assert pipeline._theme_vers_section("T04") == "B_GOUVERNANCE"
    assert pipeline._theme_vers_section("T06") == "B_GOUVERNANCE"
    assert pipeline._theme_vers_section("T02") == "E_GESTION_CAPITAL"
    assert pipeline._theme_vers_section("T01") == "D_VALORISATION"
    assert pipeline._theme_vers_section("T08") is None   # Transversal
    assert pipeline._theme_vers_section("T99") is None   # Inconnu

    print("  ✓ Mapping thème → section : T04→B, T02→E, T01→D, T08→None")


def test_analyser_exigence_avec_passages():
    """
    L'analyse d'une exigence avec des passages pertinents produit
    un ResultatAnalyseExigence valide avec un score > 0.
    """
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    doc_id = _indexer_sfcr_test(store, engine, corpus)
    pipeline = RAGPipeline(engine, store, llm, top_k=3)

    exigence = _exigence_test()
    res = pipeline._analyser_exigence(exigence, doc_id)

    assert isinstance(res, ResultatAnalyseExigence)
    assert res.matching.id_exigence == exigence.id_exigence
    assert 0.0 <= res.matching.score_final <= 1.0
    assert 0.0 <= res.matching.score_similarite <= 1.0
    assert 0.0 <= res.matching.score_llm <= 1.0
    assert res.matching.statut in [
        StatutConformite.COUVERT,
        StatutConformite.PARTIEL,
        StatutConformite.ABSENT,
    ]
    assert res.nb_candidats >= 0
    assert res.temps_analyse_s >= 0

    print(
        f"  ✓ Analyse exigence : score_final={res.matching.score_final:.3f}, "
        f"statut={res.matching.statut.value}, "
        f"candidats={res.nb_candidats}"
    )


def test_analyser_exigence_sans_sfcr_indexe():
    """
    Quand aucun SFCR n'est indexé, l'exigence est marquée ABSENT
    sans erreur levée.
    """
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    pipeline = RAGPipeline(engine, store, llm)

    exigence = _exigence_test()
    # doc_id inexistant dans le store
    res = pipeline._analyser_exigence(exigence, "doc-inexistant-xyz")

    assert res.matching.statut == StatutConformite.ABSENT
    assert res.matching.score_final == 0.0
    assert res.nb_candidats == 0

    print("  ✓ Exigence sans SFCR indexé → ABSENT proprement (pas d'exception)")


def test_statut_depuis_score_final():
    """La classification COUVERT/PARTIEL/ABSENT est correcte."""
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
            f"Score {score} → {statut.value} ≠ attendu {statut_attendu.value}"
        )
    print("  ✓ Classification statuts : COUVERT≥0.75, PARTIEL≥0.45, ABSENT<0.45")


def test_analyser_sfcr_complet():
    """
    L'analyse complète d'un SFCR produit des résultats pour
    chaque exigence, sans erreur, avec des statistiques cohérentes.
    """
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    doc_id = _indexer_sfcr_test(store, engine, corpus)
    pipeline = RAGPipeline(engine, store, llm, top_k=3)

    # Créer un petit jeu d'exigences de test (3 exigences)
    exigences = [
        _exigence_test("SII-P2-DIR2009-Art46-§1", "T06"),
        _exigence_test("SII-P2-DIR2009-Art47-§1", "T06"),
        _exigence_test("SII-P3-REG2015-Art298-§1", "T08"),
    ]

    resultat = pipeline.analyser_sfcr(
        exigences=exigences,
        doc_id=doc_id,
        entreprise="Test SA",
        annee=2023,
    )

    assert isinstance(resultat, ResultatAnalyseSFCR)
    assert resultat.nb_exigences_analysees == 3
    assert len(resultat.resultats_exigences) == 3
    assert resultat.nb_erreurs == 0
    assert resultat.temps_total_s >= 0

    # Vérifier que tous les matchings ont des champs valides
    for r in resultat.resultats_exigences:
        assert 0.0 <= r.matching.score_final <= 1.0
        assert r.matching.statut in list(StatutConformite)
        assert r.matching.id_exigence in [e.id_exigence for e in exigences]

    # matchings() retourne la liste plate
    matchings = resultat.matchings
    assert len(matchings) == 3

    print(
        f"  ✓ Analyse SFCR complète : {len(exigences)} exigences analysées, "
        f"0 erreur, {resultat.temps_total_s:.2f}s"
    )


def test_analyser_sfcr_avec_filtre_theme():
    """Le filtre par thème réduit les exigences analysées."""
    engine, store, corpus = _creer_engine_et_store()
    llm = RuleBasedLLM()
    doc_id = _indexer_sfcr_test(store, engine, corpus)
    pipeline = RAGPipeline(engine, store, llm)

    exigences = [
        _exigence_test("SII-P2-DIR2009-Art46-§1", "T06"),
        _exigence_test("SII-P2-DIR2009-Art47-§1", "T06"),
        _exigence_test("SII-P3-REG2015-Art298-§1", "T08"),
    ]

    # Analyser uniquement T06
    resultat = pipeline.analyser_sfcr(
        exigences=exigences,
        doc_id=doc_id,
        entreprise="Test SA",
        annee=2023,
        filtre_themes=["T06"],
    )

    assert resultat.nb_exigences_analysees == 2
    assert all(r.exigence.theme_id == "T06"
               for r in resultat.resultats_exigences)

    print(
        f"  ✓ Filtre thème T06 : {resultat.nb_exigences_analysees}/3 "
        f"exigences analysées"
    )


def test_pipeline_e2e_referentiel_reel():
    """
    Test end-to-end avec le référentiel réel (71 exigences).
    Valide l'intégration Phase 1 + Phase 2 + Phase 3 + Phase 4.
    """
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    from src.ingestion.referentiel_loader import ReferentielLoader

    # Charger le référentiel réel
    loader = ReferentielLoader()
    loader.charger(strict=True)
    exigences = loader.get_toutes_exigences()

    # Fit du moteur sur le corpus réel
    textes_fit = (
        [e.texte_normalise for e in exigences] +
        [e.texte_verification for e in exigences]
    )
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(textes_fit)

    # Construire le VectorStore
    if VECTORSTORE_TEST_DIR.exists():
        shutil.rmtree(VECTORSTORE_TEST_DIR)
    store = VectorStore(persist_dir=VECTORSTORE_TEST_DIR)
    store.initialiser(engine)

    # Indexer un SFCR synthétique
    from src.ingestion.models import ChunkDocument

    sfcr_synthetique = [
        ("Le système de gouvernance comprend le conseil d'administration, la "
         "direction générale et les quatre fonctions clés : gestion des risques, "
         "conformité, audit interne et fonction actuarielle indépendante.",
         "B_GOUVERNANCE"),
        ("Le système de contrôle interne repose sur des procédures administratives "
         "et comptables documentées, un cadre de contrôle permanent à trois niveaux "
         "et un reporting structuré à tous les niveaux hiérarchiques.",
         "B_GOUVERNANCE"),
        ("L'évaluation interne des risques et de la solvabilité (ORSA) a été "
         "réalisée en 2023. Les besoins globaux de solvabilité ont été évalués "
         "en tenant compte du profil de risque spécifique de l'entreprise.",
         "C_PROFIL_RISQUE"),
        ("Les provisions techniques Best Estimate ont été calculées conformément "
         "aux guidelines EIOPA. La Risk Margin est calculée par la méthode du "
         "coût du capital avec un taux de 6%.",
         "D_VALORISATION"),
        ("Le SCR s'élève à 450 M€, calculé selon la formule standard. "
         "Les fonds propres éligibles atteignent 820 M€, soit un ratio de "
         "couverture de 182%. Le MCR est couvert à 485%.",
         "E_GESTION_CAPITAL"),
        ("La politique de rémunération distingue une composante fixe et une "
         "composante variable pour les dirigeants effectifs et les détenteurs "
         "de fonctions clés, conformément aux exigences Solvabilité II.",
         "B_GOUVERNANCE"),
    ]

    doc_id = str(uuid.uuid4())
    chunks = [
        ChunkDocument(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            type_doc="SFCR",
            texte=texte,
            page_debut=i + 1,
            page_fin=i + 1,
            section=section,
            position_dans_doc=i,
            nb_tokens=compter_tokens(texte),
        )
        for i, (texte, section) in enumerate(sfcr_synthetique)
    ]

    embs_sfcr = engine.embed_documents([c.texte for c in chunks])
    store.indexer_sfcr(chunks, embs_sfcr.tolist(), "Groupe Test", 2023)

    # Analyser avec le pipeline RAG (thème T06 uniquement pour rapidité)
    llm = RuleBasedLLM()
    pipeline = RAGPipeline(engine, store, llm, top_k=3)

    exigences_t06 = loader.get_exigences_par_theme("T06")
    resultat = pipeline.analyser_sfcr(
        exigences=exigences_t06,
        doc_id=doc_id,
        entreprise="Groupe Test",
        annee=2023,
    )

    assert resultat.nb_exigences_analysees == len(exigences_t06)
    assert len(resultat.matchings) == len(exigences_t06)
    assert resultat.nb_erreurs == 0

    # Au moins une exigence doit être COUVERTE ou PARTIELLE
    # (le SFCR synthétique couvre explicitement le contrôle interne)
    non_absentes = [
        r for r in resultat.resultats_exigences
        if r.matching.statut != StatutConformite.ABSENT
    ]
    assert len(non_absentes) >= 1, (
        "Aucune exigence T06 couverte dans le SFCR synthétique — "
        "vérifier le pipeline de retrieval"
    )

    scores = [r.matching.score_final for r in resultat.resultats_exigences]
    score_moyen = sum(scores) / len(scores)

    print(
        f"  ✓ E2E complet : {len(exigences_t06)} exigences T06 analysées | "
        f"score_moyen={score_moyen:.3f} | "
        f"non_absentes={len(non_absentes)}/{len(exigences_t06)}"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # LLM Engine
        ("Interface abstraite LLMEngine",           test_interface_abstraite_llm),
        ("AnalyseLLM contraintes score/confiance",  test_analyse_llm_dataclass),
        ("RuleBasedLLM passage pertinent",          test_rule_based_passage_pertinent),
        ("RuleBasedLLM passage hors sujet",         test_rule_based_passage_hors_sujet),
        ("RuleBasedLLM cohérence relative",         test_rule_based_coherence_relative),
        ("RuleBasedLLM nom backend",                test_rule_based_backend_nom),
        ("Factory LLM mode='local'",               test_factory_llm_mode_local),
        ("Factory LLM mode='auto' sans clé",        test_factory_llm_mode_auto_sans_cle),
        # RAGPipeline
        ("RAGPipeline initialisation poids",        test_rag_pipeline_initialisation),
        ("Calcul score final 40/60",                test_calcul_score_final),
        ("Mapping thème → section SFCR",            test_mapping_theme_section),
        ("Analyse exigence avec passages",          test_analyser_exigence_avec_passages),
        ("Analyse exigence sans SFCR indexé",       test_analyser_exigence_sans_sfcr_indexe),
        ("Classification COUVERT/PARTIEL/ABSENT",   test_statut_depuis_score_final),
        ("Analyse SFCR complète",                   test_analyser_sfcr_complet),
        ("Filtre par thème",                        test_analyser_sfcr_avec_filtre_theme),
        # End-to-end
        ("Pipeline E2E référentiel réel + SFCR",    test_pipeline_e2e_referentiel_reel),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 4 — PIPELINE RAG CORE")
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

    # Nettoyage final
    if VECTORSTORE_TEST_DIR.exists():
        shutil.rmtree(VECTORSTORE_TEST_DIR)

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)
    if nb_ko == 0:
        print("\n✓ Phase 4 validée — prêt pour la Phase 5 (moteur de scoring)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
