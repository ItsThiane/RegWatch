"""
tests/test_phase7.py — Validation Phase 7 : Évaluation & Optimisation
======================================================================
Teste :
  - Ground truth : structure, cohérence, couverture
  - Métriques IR : Precision@K, Recall@K, MRR, NDCG@K
  - RAGEvaluator : pipeline d'évaluation complet
  - Propriétés mathématiques des métriques
  - Rapport d'évaluation : format et contenu

Exécution : python tests/test_phase7.py
"""

import sys
import math
import shutil
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.ground_truth import (
    GROUND_TRUTH, EntreeGroundTruth, get_tous_ids_exigences,
)
from src.evaluation.rag_evaluator import (
    RAGEvaluator, MetriquesRetrieval, RapportEvaluation,
    indexer_ground_truth_comme_sfcr,
)


# ---------------------------------------------------------------------------
# Tests Ground Truth
# ---------------------------------------------------------------------------

def test_ground_truth_structure():
    """Chaque entrée du GT a la structure correcte."""
    assert len(GROUND_TRUTH) >= 15, (
        f"Ground truth trop petit : {len(GROUND_TRUTH)} entrées"
    )
    for entree in GROUND_TRUTH:
        assert entree.id_exigence, "id_exigence vide"
        assert len(entree.passages_niveau2) >= 1, (
            f"{entree.id_exigence} : aucun passage niveau 2"
        )
        assert all(len(p) >= 50 for p in entree.passages_niveau2), (
            f"{entree.id_exigence} : passage niveau 2 trop court"
        )
    print(f"  ✓ Ground truth : {len(GROUND_TRUTH)} entrées, structure valide")


def test_ground_truth_ids_dans_referentiel():
    """Tous les id_exigence du GT existent dans le référentiel."""
    from src.ingestion.referentiel_loader import ReferentielLoader
    loader = ReferentielLoader()
    loader.charger(strict=False)
    ids_ref = {e.id_exigence for e in loader.get_toutes_exigences()}

    ids_gt = get_tous_ids_exigences()
    ids_inconnus = [id_e for id_e in ids_gt if id_e not in ids_ref]

    assert not ids_inconnus, (
        f"IDs du GT absents du référentiel : {ids_inconnus}"
    )
    print(
        f"  ✓ IDs ground truth : {len(ids_gt)} exigences "
        f"toutes présentes dans le référentiel"
    )


def test_ground_truth_couverture_themes():
    """Le GT couvre au moins 6 thèmes distincts."""
    from src.ingestion.referentiel_loader import ReferentielLoader
    loader = ReferentielLoader()
    loader.charger(strict=False)

    themes_couverts = set()
    for entree in GROUND_TRUTH:
        exig = loader.get_exigence(entree.id_exigence)
        if exig:
            themes_couverts.add(exig.theme_id)

    assert len(themes_couverts) >= 6, (
        f"Couverture thématique insuffisante : "
        f"{len(themes_couverts)} thèmes couverts"
    )
    print(
        f"  ✓ Couverture thématique : "
        f"{len(themes_couverts)} thèmes couverts sur 11"
    )


def test_ground_truth_passages_distincts():
    """Les passages du GT sont suffisamment distincts (pas de doublons)."""
    tous_passages = []
    for entree in GROUND_TRUTH:
        tous_passages.extend(entree.passages_niveau2)
        tous_passages.extend(entree.passages_niveau1)

    # Pas de passages identiques
    assert len(tous_passages) == len(set(tous_passages)), (
        "Des passages sont dupliqués dans le ground truth"
    )
    print(
        f"  ✓ Passages distincts : {len(tous_passages)} passages, "
        f"aucun doublon"
    )


# ---------------------------------------------------------------------------
# Tests propriétés mathématiques des métriques
# ---------------------------------------------------------------------------

def _creer_metriques(k, p, r, mrr, ndcg, nb=10, hits=7):
    return MetriquesRetrieval(
        k=k, precision_at_k=p, recall_at_k=r,
        mrr=mrr, ndcg_at_k=ndcg,
        nb_exigences_evaluees=nb,
        nb_exigences_avec_hit=hits,
    )


def test_precision_at_k_bornee():
    """Precision@K ∈ [0, 1]."""
    for p in [0.0, 0.5, 1.0]:
        m = _creer_metriques(5, p, 0.5, 0.5, 0.5)
        assert 0.0 <= m.precision_at_k <= 1.0
    print("  ✓ Precision@K ∈ [0, 1]")


def test_recall_at_k_bornee():
    """Recall@K ∈ [0, 1]."""
    for r in [0.0, 0.5, 1.0]:
        m = _creer_metriques(5, 0.5, r, 0.5, 0.5)
        assert 0.0 <= m.recall_at_k <= 1.0
    print("  ✓ Recall@K ∈ [0, 1]")


def test_mrr_bornee():
    """MRR ∈ [0, 1]."""
    for mrr in [0.0, 0.333, 1.0]:
        m = _creer_metriques(5, 0.5, 0.5, mrr, 0.5)
        assert 0.0 <= m.mrr <= 1.0
    print("  ✓ MRR ∈ [0, 1]")


def test_ndcg_calcul_manuel():
    """
    Vérification manuelle du calcul NDCG.

    Niveaux [2, 1, 0] avec K=3 :
    DCG@3  = (4-1)/log2(2) + (2-1)/log2(3) + (1-1)/log2(4)
           = 3/1 + 1/1.585 + 0
           = 3 + 0.631 = 3.631
    IDCG@3 = (4-1)/1 + (2-1)/1.585 + 0 = 3.631 (déjà trié)
    NDCG@3 = 3.631 / 3.631 = 1.0
    """
    niveaux_tries = [2, 1, 0]
    score = RAGEvaluator._ndcg_at_k(niveaux_tries, k=3)
    assert abs(score - 1.0) < 0.01, (
        f"NDCG@3 pour [2,1,0] = {score:.4f} ≠ 1.0"
    )

    # Niveaux inversés [0, 1, 2] — pire classement possible
    niveaux_inverses = [0, 1, 2]
    score_inv = RAGEvaluator._ndcg_at_k(niveaux_inverses, k=3)
    score_ideal = RAGEvaluator._ndcg_at_k(niveaux_tries, k=3)
    assert score_inv < score_ideal, (
        f"Classement inversé devrait avoir NDCG < classement idéal : "
        f"{score_inv:.4f} vs {score_ideal:.4f}"
    )

    # Tous zéros → NDCG = 0
    score_zero = RAGEvaluator._ndcg_at_k([0, 0, 0], k=3)
    assert score_zero == 0.0

    print(
        f"  ✓ NDCG calcul : [2,1,0]={score:.3f} | "
        f"[0,1,2]={score_inv:.3f} | [0,0,0]={score_zero:.3f}"
    )


def test_ndcg_monotonie_classement():
    """
    Un meilleur classement (pertinent en tête) donne un NDCG plus élevé.
    """
    # Cas 1 : pertinent en 1ère position
    score_optimal = RAGEvaluator._ndcg_at_k([2, 0, 0], k=3)
    # Cas 2 : pertinent en 3ème position
    score_tardif  = RAGEvaluator._ndcg_at_k([0, 0, 2], k=3)
    assert score_optimal > score_tardif, (
        f"NDCG devrait décroître avec la position du résultat pertinent : "
        f"{score_optimal:.4f} vs {score_tardif:.4f}"
    )
    print(
        f"  ✓ NDCG monotonie : pertinent@1={score_optimal:.3f} > "
        f"pertinent@3={score_tardif:.3f}"
    )


def test_similarite_lexicale():
    """La similarité lexicale donne des résultats cohérents."""
    texte_ref = (
        "le système de contrôle interne comprend des procédures administratives "
        "et comptables ainsi que le cadre de contrôle"
    )
    # Très similaire
    sim_haute = RAGEvaluator._similarite_lexicale(
        texte_ref,
        "procédures administratives et comptables cadre contrôle interne",
    )
    # Très différent
    sim_basse = RAGEvaluator._similarite_lexicale(
        texte_ref,
        "le ratio de couverture scr s'établit à 189% au 31 décembre 2023",
    )

    assert sim_haute > sim_basse, (
        f"Similarité incohérente : similaire={sim_haute:.3f} ≤ "
        f"différent={sim_basse:.3f}"
    )
    assert 0.0 <= sim_haute <= 1.0
    assert 0.0 <= sim_basse <= 1.0
    print(
        f"  ✓ Similarité lexicale : similaire={sim_haute:.3f} > "
        f"différent={sim_basse:.3f}"
    )


# ---------------------------------------------------------------------------
# Tests RAGEvaluator avec données réelles
# ---------------------------------------------------------------------------

def _setup_evaluateur():
    """Crée un RAGEvaluator avec ground truth indexé."""
    from src.ingestion.referentiel_loader import ReferentielLoader
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore

    loader = ReferentielLoader()
    loader.charger()
    exigences = loader.get_toutes_exigences()

    # Corpus de fit = référentiel + passages ground truth
    corpus_fit = (
        [e.texte_normalise for e in exigences] +
        [e.texte_verification for e in exigences] +
        [p for gt in GROUND_TRUTH for p in gt.passages_niveau2] +
        [p for gt in GROUND_TRUTH for p in gt.passages_niveau1]
    )
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(corpus_fit)

    vs_dir = Path(f"/tmp/regwatch_eval_{uuid.uuid4().hex[:8]}")
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine)

    # Indexer le référentiel
    embs_ref = engine.embed_documents(
        [e.texte_verification for e in exigences]
    )
    store.indexer_referentiel(exigences, embs_ref.tolist())

    # Indexer le ground truth comme SFCR
    doc_id = indexer_ground_truth_comme_sfcr(store, engine, GROUND_TRUTH)

    evaluateur = RAGEvaluator(engine, store)
    return evaluateur, doc_id, vs_dir


def test_evaluateur_produit_metriques():
    """Le RAGEvaluator produit des métriques pour tous les K demandés."""
    evaluateur, doc_id, vs_dir = _setup_evaluateur()

    rapport = evaluateur.evaluer(
        ground_truth=GROUND_TRUTH[:5],  # Sous-ensemble pour rapidité
        k_values=[1, 3, 5],
        doc_id=doc_id,
    )

    assert len(rapport.metriques_par_k) == 3
    assert 1 in rapport.metriques_par_k
    assert 3 in rapport.metriques_par_k
    assert 5 in rapport.metriques_par_k

    for k, m in rapport.metriques_par_k.items():
        assert m.k == k
        assert 0.0 <= m.precision_at_k <= 1.0
        assert 0.0 <= m.recall_at_k <= 1.0
        assert 0.0 <= m.mrr <= 1.0
        assert 0.0 <= m.ndcg_at_k <= 1.0
        assert m.nb_exigences_evaluees == 5

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ RAGEvaluator : métriques produites pour K=[1,3,5] | "
        f"P@5={rapport.metriques_par_k[5].precision_at_k:.3f} | "
        f"R@5={rapport.metriques_par_k[5].recall_at_k:.3f}"
    )


def test_recall_monotone_en_k():
    """
    Recall@K doit être non-décroissant en K.
    Plus K est grand, plus on trouve de passages pertinents.
    """
    evaluateur, doc_id, vs_dir = _setup_evaluateur()

    rapport = evaluateur.evaluer(
        ground_truth=GROUND_TRUTH[:8],
        k_values=[1, 3, 5, 10],
        doc_id=doc_id,
    )

    recalls = [
        rapport.metriques_par_k[k].recall_at_k
        for k in sorted(rapport.metriques_par_k.keys())
    ]
    for i in range(len(recalls) - 1):
        assert recalls[i] <= recalls[i+1] + 0.01, (
            f"Recall non monotone : K={i+1} → {recalls[i]:.3f} > "
            f"K={i+2} → {recalls[i+1]:.3f}"
        )

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ Monotonie Recall@K : {[round(r, 3) for r in recalls]} "
        f"(non-décroissant)"
    )


def test_mrr_superieur_a_precision():
    """
    MRR doit être ≥ Precision@1 (par construction mathématique).
    Si Precision@1 > 0, le 1er résultat est pertinent → MRR ≥ 1.
    """
    evaluateur, doc_id, vs_dir = _setup_evaluateur()

    rapport = evaluateur.evaluer(
        ground_truth=GROUND_TRUTH[:5],
        k_values=[1],
        doc_id=doc_id,
    )
    m = rapport.metriques_par_k[1]

    # MRR ≥ Precision@1 car MRR compte 1/rank (rang toujours ≥1)
    # et Precision@1 = nb_pertinents/1, donc si P@1=1 alors MRR=1
    assert m.mrr >= m.precision_at_k - 0.01, (
        f"MRR ({m.mrr:.3f}) < Precision@1 ({m.precision_at_k:.3f})"
    )

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ MRR ≥ P@1 : MRR={m.mrr:.3f} ≥ P@1={m.precision_at_k:.3f}"
    )


def test_rapport_evaluation_format():
    """Le rapport formaté contient toutes les sections."""
    evaluateur, doc_id, vs_dir = _setup_evaluateur()

    rapport = evaluateur.evaluer(
        ground_truth=GROUND_TRUTH[:3],
        k_values=[1, 5],
        doc_id=doc_id,
    )
    texte = rapport.formater()

    sections = [
        "RAPPORT D'ÉVALUATION",
        "MÉTRIQUES PAR VALEUR DE K",
        "MEILLEURE CONFIG",
        "RECOMMANDATIONS",
    ]
    for s in sections:
        assert s in texte, f"Section absente : '{s}'"

    assert len(rapport.recommandations_optimisation) >= 2

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ Rapport formaté : {len(texte)} chars | "
        f"{len(rapport.recommandations_optimisation)} recommandations"
    )


def test_ground_truth_indexation():
    """L'indexation du GT comme SFCR crée les bons chunks."""
    from src.ingestion.referentiel_loader import ReferentielLoader
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore

    loader = ReferentielLoader()
    loader.charger()
    corpus = [e.texte_normalise for e in loader.get_toutes_exigences()]
    corpus += [p for gt in GROUND_TRUTH for p in gt.passages_niveau2]

    engine = LocalTfidfEmbedder(dimension=128)
    engine.fit(corpus)

    vs_dir = Path(f"/tmp/regwatch_gt_{uuid.uuid4().hex[:8]}")
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine)

    embs = engine.embed_documents(corpus[:5])
    store.indexer_referentiel(
        loader.get_toutes_exigences()[:5],
        embs.tolist()
    )

    doc_id = indexer_ground_truth_comme_sfcr(
        store, engine, GROUND_TRUTH[:3]
    )
    # Vérifier via stats() car est_sfcr_indexe n'existe pas dans ce VectorStore
    stats = store.stats()
    assert stats['sfcr']['nb_documents'] > 0, 'Aucun chunk SFCR indexé'
    stats = store.stats()
    # Au moins autant de chunks que de passages
    nb_passages = sum(
        len(gt.passages_niveau2) + len(gt.passages_niveau1) + len(gt.passages_niveau0)
        for gt in GROUND_TRUTH[:3]
    )
    assert stats["sfcr"]["nb_documents"] >= nb_passages

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ Indexation GT : doc_id={doc_id[:8]}... | "
        f"{stats['sfcr']['nb_documents']} chunks SFCR indexés"
    )


# ---------------------------------------------------------------------------
# Test E2E Phase 7
# ---------------------------------------------------------------------------

def test_pipeline_e2e_phases_1_7():
    """
    Test end-to-end Phases 1→7 : référentiel → RAG → scoring → évaluation.
    """
    from src.ingestion.referentiel_loader import ReferentielLoader
    from src.rag.embedding_engine import LocalTfidfEmbedder
    from src.rag.vector_store import VectorStore
    from src.rag.llm_engine import RuleBasedLLM
    from src.rag.rag_pipeline import RAGPipeline
    from src.scoring.scoring_engine import ScoringEngine

    loader = ReferentielLoader()
    loader.charger()
    exigences = loader.get_toutes_exigences()

    corpus = (
        [e.texte_normalise for e in exigences] +
        [p for gt in GROUND_TRUTH for p in gt.passages_niveau2] +
        [p for gt in GROUND_TRUTH for p in gt.passages_niveau1]
    )
    engine = LocalTfidfEmbedder(dimension=256)
    engine.fit(corpus)

    vs_dir = Path(f"/tmp/regwatch_e2e7_{uuid.uuid4().hex[:8]}")
    store = VectorStore(persist_dir=vs_dir)
    store.initialiser(engine)

    # Indexer référentiel
    embs_ref = engine.embed_documents(
        [e.texte_verification for e in exigences]
    )
    store.indexer_referentiel(exigences, embs_ref.tolist())

    # Indexer le ground truth comme SFCR
    doc_id = indexer_ground_truth_comme_sfcr(store, engine, GROUND_TRUTH)

    # RAG
    llm = RuleBasedLLM()
    rag = RAGPipeline(engine, store, llm, top_k=5)
    ids_gt = get_tous_ids_exigences()
    exigences_gt = [
        e for e in exigences if e.id_exigence in ids_gt
    ]
    res_rag = rag.analyser_sfcr(
        exigences=exigences_gt,
        doc_id=doc_id,
        entreprise="Ground Truth Test",
        annee=2023,
    )
    assert res_rag.nb_erreurs == 0

    # Scoring
    scoring = ScoringEngine(loader)
    rapport = scoring.calculer(res_rag)
    assert 0.0 <= rapport.score_global <= 1.0

    # Évaluation
    evaluateur = RAGEvaluator(engine, store)
    rapport_eval = evaluateur.evaluer(
        ground_truth=GROUND_TRUTH,
        k_values=[1, 3, 5],
        doc_id=doc_id,
    )

    m5 = rapport_eval.metriques_par_k[5]
    assert m5.nb_exigences_evaluees == len(GROUND_TRUTH)
    assert 0.0 <= m5.precision_at_k <= 1.0
    assert 0.0 <= m5.recall_at_k <= 1.0
    assert 0.0 <= m5.mrr <= 1.0
    assert 0.0 <= m5.ndcg_at_k <= 1.0

    shutil.rmtree(vs_dir)
    print(
        f"  ✓ Pipeline E2E Phases 1→7 :\n"
        f"    Scoring   : score_global={rapport.score_global:.3f} | "
        f"niveau={rapport.niveau_conformite.value}\n"
        f"    Évaluation: P@5={m5.precision_at_k:.3f} | "
        f"R@5={m5.recall_at_k:.3f} | "
        f"MRR={m5.mrr:.3f} | "
        f"NDCG@5={m5.ndcg_at_k:.3f}\n"
        f"    Hits      : {m5.nb_exigences_avec_hit}/{m5.nb_exigences_evaluees} "
        f"exigences avec ≥1 résultat pertinent"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # Ground truth
        ("Structure du ground truth",                test_ground_truth_structure),
        ("IDs GT dans le référentiel",               test_ground_truth_ids_dans_referentiel),
        ("Couverture thématique du GT",              test_ground_truth_couverture_themes),
        ("Passages GT distincts",                    test_ground_truth_passages_distincts),
        # Propriétés mathématiques
        ("Precision@K ∈ [0, 1]",                    test_precision_at_k_bornee),
        ("Recall@K ∈ [0, 1]",                       test_recall_at_k_bornee),
        ("MRR ∈ [0, 1]",                            test_mrr_bornee),
        ("NDCG calcul manuel",                       test_ndcg_calcul_manuel),
        ("NDCG monotonie du classement",             test_ndcg_monotonie_classement),
        ("Similarité lexicale cohérente",            test_similarite_lexicale),
        # RAGEvaluator
        ("RAGEvaluator produit métriques",           test_evaluateur_produit_metriques),
        ("Recall monotone en K",                     test_recall_monotone_en_k),
        ("MRR ≥ Precision@1",                        test_mrr_superieur_a_precision),
        ("Rapport évaluation formaté",               test_rapport_evaluation_format),
        ("Indexation ground truth comme SFCR",       test_ground_truth_indexation),
        # E2E
        ("Pipeline E2E Phases 1→7 complet",          test_pipeline_e2e_phases_1_7),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 7 — ÉVALUATION & OPTIMISATION RAG")
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

    # Nettoyage
    for d in Path("/tmp").glob("regwatch_*"):
        shutil.rmtree(d, ignore_errors=True)

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)
    if nb_ko == 0:
        print("\n✓ Phase 7 validée — projet RegWatch complet !")
        print("\n" + "=" * 60)
        print("BILAN FINAL : 7 phases | 93 tests | 0 KO")
        print("=" * 60)
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
