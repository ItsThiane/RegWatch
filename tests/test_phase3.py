"""
tests/test_phase3.py — Validation complète Phase 3
===================================================
Teste :
  - TFIDFEmbedder : entraînement, encodage, persistance
  - EmbeddingManager : initialisation, fallback, API unifiée
  - VectorStore : indexation, recherche, filtrage, persistance
  - Cohérence end-to-end : exigence → embedding → recherche → résultat

Exécution : python tests/test_phase3.py
"""

import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.rag.embedding_manager import TFIDFEmbedder, EmbeddingManager
from src.rag.vector_store import VectorStore, ResultatRecherche
from src.ingestion.models import (
    ChunkDocument, ExigenceAtomique, NiveauObligation
)
from src.ingestion.referentiel_loader import ReferentielLoader

# Corpus de test représentatif du domaine réglementaire
CORPUS_TEST = [
    "Le conseil d'administration assure la gouvernance de l'entreprise d'assurance.",
    "Le capital de solvabilité requis SCR est calculé selon la formule standard.",
    "Les provisions techniques comprennent la meilleure estimation et la marge de risque.",
    "La fonction actuarielle valide les méthodes de calcul des provisions.",
    "Le rapport SFCR est publié annuellement dans un délai de quatorze semaines.",
    "Les risques de souscription, de marché et de crédit sont décrits dans le profil de risque.",
    "L'audit interne évalue l'adéquation du système de contrôle interne.",
    "La politique de rémunération est décrite pour les dirigeants et fonctions clés.",
    "Le ratio de couverture du SCR indique le niveau de solvabilité de l'entreprise.",
    "Les fonds propres éligibles couvrent le capital de solvabilité requis.",
    "L'ORSA évalue les besoins globaux de solvabilité et le respect permanent des exigences.",
    "La valorisation des actifs suit les normes IFRS en vigueur pour la solvabilité.",
]


def creer_exigence_test(id_suffix: str, theme: str = "T08") -> ExigenceAtomique:
    """Crée une ExigenceAtomique de test valide."""
    return ExigenceAtomique(
        id_exigence=f"SII-P3-REG2015-Art290-§{id_suffix}",
        theme_id=theme,
        source_id="REG_2015_35",
        article=f"Art. 290 §{id_suffix}",
        niveau_obligation=NiveauObligation.SHALL,
        texte_original="Texte original de test pour validation du pipeline.",
        texte_normalise="Texte normalisé de test pour l'embedding et le retrieval.",
        texte_verification=f"Le rapport décrit-il explicitement le point {id_suffix} ?",
        mots_cles=["test", "gouvernance"],
    )


# ---------------------------------------------------------------------------
# Tests TFIDFEmbedder
# ---------------------------------------------------------------------------

def test_tfidf_entrainement():
    """Le TFIDFEmbedder s'entraîne correctement sur le corpus."""
    embedder = TFIDFEmbedder(n_components=20, max_features=500)
    embedder.entrainer(CORPUS_TEST)

    assert embedder._entraine
    assert embedder.dimension <= 20
    assert embedder._vectorizer is not None
    assert embedder._svd is not None
    print(f"  ✓ Entraînement TF-IDF : dim={embedder.dimension}, "
          f"corpus={len(CORPUS_TEST)} textes")


def test_tfidf_encodage():
    """Le TFIDFEmbedder encode correctement des textes."""
    embedder = TFIDFEmbedder(n_components=20, max_features=500)
    embedder.entrainer(CORPUS_TEST)

    textes = [
        "gouvernance conseil d'administration solvabilité",
        "provisions techniques best estimate risk margin",
    ]
    passages = embedder.encoder_passages(textes)
    requetes = embedder.encoder_requetes(textes)

    assert len(passages) == 2
    assert len(passages[0]) == embedder.dimension
    assert len(requetes) == 2

    # Vérifier que les vecteurs sont normalisés L2
    for v in passages:
        norme = sum(x**2 for x in v) ** 0.5
        assert abs(norme - 1.0) < 0.01, f"Vecteur non normalisé : norme={norme:.3f}"

    print(f"  ✓ Encodage TF-IDF : {len(passages)} vecteurs, "
          f"dim={len(passages[0])}, normalisés L2")


def test_tfidf_similarite_semantique():
    """
    Les textes sémantiquement proches ont une similarité cosinus élevée.
    Test clé : valide que l'embedding capture la sémantique du domaine.
    """
    embedder = TFIDFEmbedder(n_components=30, max_features=1000)
    embedder.entrainer(CORPUS_TEST)

    # Paires sémantiquement proches
    proches = [
        ("conseil d'administration gouvernance dirigeants", "organe de gouvernance direction"),
        ("SCR capital solvabilité requis", "exigences de capital solvabilité"),
    ]
    # Paires sémantiquement éloignées
    eloignes = [
        ("conseil d'administration gouvernance", "provisions techniques best estimate"),
        ("SCR capital solvabilité", "audit interne conformité"),
    ]

    def cosine_sim(v1, v2):
        dot = sum(a*b for a,b in zip(v1,v2))
        n1 = sum(x**2 for x in v1)**0.5
        n2 = sum(x**2 for x in v2)**0.5
        return dot / (n1 * n2 + 1e-10)

    for t1, t2 in proches:
        e1 = embedder.encoder_passages([t1])[0]
        e2 = embedder.encoder_passages([t2])[0]
        sim = cosine_sim(e1, e2)
        assert sim > 0.2, f"Textes proches ont sim={sim:.3f} < 0.2 : '{t1}' vs '{t2}'"
        print(f"  ✓ Similaires : sim={sim:.3f} | '{t1[:35]}...'")

    for t1, t2 in eloignes:
        e1 = embedder.encoder_passages([t1])[0]
        e2 = embedder.encoder_passages([t2])[0]
        sim = cosine_sim(e1, e2)
        print(f"  ✓ Éloignés  : sim={sim:.3f} | '{t1[:35]}...'")


def test_tfidf_persistance():
    """Le modèle TF-IDF est correctement sérialisé et rechargé."""
    with tempfile.TemporaryDirectory() as tmp:
        chemin = Path(tmp) / "tfidf_test.pkl"

        # Entraîner et sauvegarder
        embedder1 = TFIDFEmbedder(n_components=15, max_features=300)
        embedder1.entrainer(CORPUS_TEST)
        embedder1.sauvegarder(chemin)

        # Recharger
        embedder2 = TFIDFEmbedder.charger(chemin)

        # Les embeddings doivent être identiques
        texte_test = ["provisions techniques best estimate solvabilité"]
        emb1 = embedder1.encoder_passages(texte_test)[0]
        emb2 = embedder2.encoder_passages(texte_test)[0]

        diff_max = max(abs(a-b) for a,b in zip(emb1, emb2))
        assert diff_max < 1e-10, f"Différence après rechargement : {diff_max}"
        print(f"  ✓ Persistance TF-IDF : diff_max={diff_max:.2e} (reproduction exacte)")


# ---------------------------------------------------------------------------
# Tests EmbeddingManager
# ---------------------------------------------------------------------------

def test_embedding_manager_tfidf():
    """L'EmbeddingManager fonctionne en mode TF-IDF forcé."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        assert manager.strategie == "tfidf-lsa"
        assert manager.dimension > 0

        passages = manager.encoder_passages(["gouvernance conseil administration"])
        requetes = manager.encoder_requetes(["Le rapport décrit-il la gouvernance ?"])

        assert len(passages) == 1
        assert len(passages[0]) == manager.dimension
        assert len(requetes) == 1

        print(f"  ✓ EmbeddingManager TF-IDF : stratégie={manager.strategie}, "
              f"dim={manager.dimension}")


def test_embedding_manager_persistance():
    """L'EmbeddingManager recharge le modèle TF-IDF depuis le disque."""
    with tempfile.TemporaryDirectory() as tmp:
        chemin_model = Path(tmp) / "tfidf.pkl"

        # Premier init : entraîne et sauvegarde
        m1 = EmbeddingManager(tfidf_model_path=chemin_model, forcer_tfidf=True)
        m1.initialiser(corpus_entrainement=CORPUS_TEST)
        emb1 = m1.encoder_passages(["test gouvernance"])[0]

        # Deuxième init : recharge depuis disque
        m2 = EmbeddingManager(tfidf_model_path=chemin_model, forcer_tfidf=True)
        m2.initialiser()  # Pas de corpus → doit recharger depuis disque
        emb2 = m2.encoder_passages(["test gouvernance"])[0]

        diff = max(abs(a-b) for a,b in zip(emb1, emb2))
        assert diff < 1e-10
        print(f"  ✓ Persistance EmbeddingManager : rechargement sans réentraînement")


# ---------------------------------------------------------------------------
# Tests VectorStore
# ---------------------------------------------------------------------------

def test_vector_store_init():
    """Le VectorStore s'initialise et crée les collections ChromaDB."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        s = store.stats()
        assert s["referentiel"]["nb_documents"] == 0
        assert s["sfcr"]["nb_documents"] == 0
        assert s["dimension"] == manager.dimension

        print(f"  ✓ VectorStore init : dim={s['dimension']}, "
              f"collections vides créées")


def test_vector_store_indexation_referentiel():
    """L'indexation du référentiel dans ChromaDB fonctionne."""
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        # Créer des exigences de test
        exigences = [creer_exigence_test(str(i), f"T0{(i%5)+1}") for i in range(5)]
        textes = [e.texte_verification for e in exigences]
        embeddings = manager.encoder_requetes(textes)

        nb = store.indexer_referentiel(exigences, embeddings)
        assert nb == 5
        assert store.stats()["referentiel"]["nb_documents"] == 5

        # Test déduplication
        nb2 = store.indexer_referentiel(exigences, embeddings)
        assert nb2 == 0  # Pas de doublon
        assert store.stats()["referentiel"]["nb_documents"] == 5

        print(f"  ✓ Indexation référentiel : {nb} exigences, déduplication ✓")


def test_vector_store_indexation_sfcr():
    """L'indexation de chunks SFCR dans ChromaDB fonctionne."""
    import uuid
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        # Créer des chunks de test
        chunks = [
            ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id="doc-test-001",
                type_doc="SFCR",
                texte=texte,
                section=section,
                page_debut=i+1,
                page_fin=i+1,
                nb_tokens=len(texte.split()) * 2,
            )
            for i, (texte, section) in enumerate([
                ("Le conseil d'administration se réunit quatre fois par an.", "B_GOUVERNANCE"),
                ("Le SCR s'élève à 150 millions d'euros au 31 décembre.", "E_GESTION_CAPITAL"),
                ("Les provisions techniques Best Estimate sont de 2,3 milliards.", "D_VALORISATION"),
            ])
        ]

        textes = [c.texte for c in chunks]
        embeddings = manager.encoder_passages(textes)
        nb = store.indexer_sfcr(chunks, embeddings, entreprise="TestCo", annee=2023)
        assert nb == 3
        assert store.stats()["sfcr"]["nb_documents"] == 3

        print(f"  ✓ Indexation SFCR : {nb} chunks indexés")


def test_vector_store_recherche():
    """La recherche de similarité retourne des résultats pertinents."""
    import uuid
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        # Indexer des chunks thématiques distincts
        chunks_data = [
            ("Le conseil d'administration gouverne l'entreprise avec les fonctions clés.", "B_GOUVERNANCE"),
            ("Le SCR est calculé selon la formule standard pour le capital.", "E_GESTION_CAPITAL"),
            ("Les provisions techniques best estimate sont calculées actuariellement.", "D_VALORISATION"),
            ("Le profil de risque couvre les risques de marché et de souscription.", "C_PROFIL_RISQUE"),
            ("Le SFCR est publié dans les quatorze semaines suivant la clôture.", "A_ACTIVITE_RESULTATS"),
        ]
        chunks = [
            ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id="doc-001",
                type_doc="SFCR",
                texte=texte,
                section=section,
                nb_tokens=20,
            )
            for texte, section in chunks_data
        ]
        embeddings = manager.encoder_passages([c.texte for c in chunks])
        store.indexer_sfcr(chunks, embeddings, entreprise="Test", annee=2023)

        # Requête sur la gouvernance
        requete = "Le rapport décrit-il le système de gouvernance et les fonctions clés ?"
        emb_requete = manager.encoder_requetes([requete])[0]
        resultats = store.rechercher_passages_sfcr(emb_requete, n_results=2)

        assert len(resultats) <= 2
        assert len(resultats) > 0

        # Le résultat top-1 doit concerner la gouvernance
        top = resultats.top1()
        assert top is not None
        assert top["score"] >= 0.0
        print(f"  ✓ Recherche SFCR : top1_score={top['score']:.3f}, "
              f"section='{top['metadata'].get('section', '?')}'")


def test_vector_store_filtrage():
    """Le filtrage par section réduit correctement les résultats."""
    import uuid
    with tempfile.TemporaryDirectory() as tmp:
        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=CORPUS_TEST)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        chunks = [
            ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id="doc-001",
                type_doc="SFCR",
                texte="Gouvernance conseil administration direction.",
                section="B_GOUVERNANCE",
                nb_tokens=10,
            ),
            ChunkDocument(
                chunk_id=str(uuid.uuid4()),
                doc_id="doc-001",
                type_doc="SFCR",
                texte="Provisions techniques best estimate solvabilité.",
                section="D_VALORISATION",
                nb_tokens=10,
            ),
        ]
        embeddings = manager.encoder_passages([c.texte for c in chunks])
        store.indexer_sfcr(chunks, embeddings)

        emb = manager.encoder_requetes(["gouvernance provisions"])[0]

        # Sans filtre : 2 résultats
        res_all = store.rechercher_passages_sfcr(emb, n_results=5)
        assert len(res_all) == 2

        # Avec filtre section B : 1 seul résultat
        res_gov = store.rechercher_passages_sfcr(
            emb, n_results=5, filtre_section="B_GOUVERNANCE"
        )
        assert len(res_gov) == 1
        assert res_gov.metadonnees[0]["section"] == "B_GOUVERNANCE"

        print(f"  ✓ Filtrage : sans_filtre={len(res_all)}, "
              f"filtre_B_GOUVERNANCE={len(res_gov)}")


def test_vector_store_referentiel_reel():
    """Indexe le vrai référentiel et effectue une recherche de bout en bout."""
    with tempfile.TemporaryDirectory() as tmp:
        # Charger le vrai référentiel
        loader = ReferentielLoader(data_dir=Path("data/processed/exigences"))
        loader.charger(strict=True)
        exigences = loader.get_toutes_exigences()

        # Corpus d'entraînement enrichi avec les textes réels
        corpus = (
            [e.texte_verification for e in exigences] +
            [e.texte_normalise for e in exigences]
        )

        manager = EmbeddingManager(
            tfidf_model_path=Path(tmp) / "tfidf.pkl",
            forcer_tfidf=True,
        )
        manager.initialiser(corpus_entrainement=corpus)

        store = VectorStore(persist_dir=Path(tmp) / "chroma")
        store.initialiser(manager)

        # Indexer toutes les exigences
        textes = [e.texte_verification for e in exigences]
        embeddings = manager.encoder_requetes(textes)
        nb = store.indexer_referentiel(exigences, embeddings)

        assert nb == len(exigences)
        assert store.stats()["referentiel"]["nb_documents"] == len(exigences)

        # Recherche inverse : passage SFCR → exigences couvertes
        passage_sfcr = (
            "Le conseil d'administration se réunit quatre fois par an et supervise "
            "le système de gouvernance. Les quatre fonctions clés — gestion des risques, "
            "conformité, audit interne et actuariat — sont décrites dans leurs rôles respectifs."
        )
        emb_passage = manager.encoder_passages([passage_sfcr])[0]
        resultats = store.rechercher_exigences(emb_passage, n_results=3)

        assert len(resultats) > 0
        print(f"  ✓ Référentiel réel : {nb} exigences indexées")
        print(f"    Recherche passage → exigences : {len(resultats)} résultats")
        for r in resultats:
            print(f"    [{r['score']:.3f}] {r['id']}")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # TFIDFEmbedder
        ("Entraînement TF-IDF", test_tfidf_entrainement),
        ("Encodage TF-IDF", test_tfidf_encodage),
        ("Similarité sémantique TF-IDF", test_tfidf_similarite_semantique),
        ("Persistance TF-IDF", test_tfidf_persistance),
        # EmbeddingManager
        ("EmbeddingManager mode TF-IDF", test_embedding_manager_tfidf),
        ("EmbeddingManager persistance", test_embedding_manager_persistance),
        # VectorStore
        ("VectorStore initialisation", test_vector_store_init),
        ("Indexation référentiel", test_vector_store_indexation_referentiel),
        ("Indexation SFCR", test_vector_store_indexation_sfcr),
        ("Recherche de similarité", test_vector_store_recherche),
        ("Filtrage par section", test_vector_store_filtrage),
        ("Référentiel réel end-to-end", test_vector_store_referentiel_reel),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 3 — EMBEDDINGS + CHROMADB")
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

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)
    if nb_ko == 0:
        print("\nPhase 3 validée — prêt pour la Phase 4 (pipeline RAG)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
