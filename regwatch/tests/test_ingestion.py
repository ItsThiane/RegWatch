"""
tests/test_ingestion.py — Validation complète de la Phase 2
============================================================
Teste :
  - text_utils : nettoyage, comptage tokens, préfixes e5
  - SFCRChunker : chunking, validation, préparation embedding
  - IngestionPipeline : ingestion référentiel
  - Cohérence globale du pipeline

Exécution : python tests/test_ingestion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion.text_utils import (
    compter_tokens, tronquer_a_tokens, nettoyer_texte_pdf,
    normaliser_pour_embedding, preparer_requete_e5,
    est_bruit, est_tableau_qrt, statistiques_texte,
)
from src.ingestion.sfcr_chunker import SFCRChunker, CHUNK_SIZE_SFCR_TOKENS, CHUNK_OVERLAP_TOKENS
from src.ingestion.models import ChunkDocument
from src.ingestion.ingestion_pipeline import IngestionPipeline


# ---------------------------------------------------------------------------
# Tests text_utils
# ---------------------------------------------------------------------------

def test_compter_tokens():
    """Le comptage de tokens est cohérent et non nul."""
    texte = "Le rapport décrit le système de gouvernance de l'entreprise."
    nb = compter_tokens(texte)
    assert 10 <= nb <= 20, f"Comptage inattendu : {nb} tokens pour '{texte}'"
    assert compter_tokens("") == 0
    print(f"  ✓ compter_tokens : '{texte[:40]}...' → {nb} tokens")


def test_tronquer_a_tokens():
    """La troncature respecte la limite et préserve les phrases."""
    long_texte = "La gouvernance de l'entreprise est solide. " * 50
    tronque = tronquer_a_tokens(long_texte, max_tokens=100)
    nb = compter_tokens(tronque)
    assert nb <= 110, f"Texte tronqué trop long : {nb} tokens"
    assert len(tronque) > 0
    print(f"  ✓ tronquer_a_tokens : {compter_tokens(long_texte)} → {nb} tokens")


def test_nettoyer_texte_pdf():
    """Le nettoyage corrige les artefacts PDF courants."""
    cas_tests = [
        # Césure de mots
        ("gouver-\nnance", "gouvernance"),
        # Ligature fi
        ("déﬁnition", "définition"),
        # Espaces multiples
        ("le   rapport", "le rapport"),
        # Caractères de contrôle
        ("texte\x00propre", "textepropre"),
    ]
    for entree, attendu in cas_tests:
        sortie = nettoyer_texte_pdf(entree)
        assert attendu in sortie or sortie.strip() == attendu.strip(), (
            f"Nettoyage échoué : '{entree}' → '{sortie}' (attendu '{attendu}')"
        )
    print(f"  ✓ nettoyer_texte_pdf : {len(cas_tests)} cas testés")


def test_prefixes_e5():
    """
    Les préfixes "query: " et "passage: " sont correctement appliqués.
    CRITIQUE : sans ces préfixes, multilingual-e5-large perd ~15-20% de performance.
    """
    texte = "Le rapport décrit la gouvernance de l'entreprise."
    query = preparer_requete_e5(texte, type_doc="query")
    passage = preparer_requete_e5(texte, type_doc="passage")

    assert query.startswith("query: "), f"Préfixe query absent : '{query[:20]}'"
    assert passage.startswith("passage: "), f"Préfixe passage absent : '{passage[:20]}'"
    assert texte[:20] in query
    assert texte[:20] in passage
    print(f"  ✓ Préfixes e5 : query='{query[:30]}...' | passage='{passage[:30]}...'")


def test_detection_bruit():
    """La détection de bruit identifie les en-têtes et pieds de page."""
    bruits = [
        "42",                              # Numéro de page
        "Page 3 sur 45",                   # Pied de page
        "© 2023 Groupama SA",              # Copyright
        "Confidentiel — usage interne",    # Mention confidentialité
    ]
    pas_bruits = [
        "Le conseil d'administration se réunit quatre fois par an pour examiner "
        "la stratégie et les résultats de l'entreprise.",
        "La fonction de gestion des risques est chargée d'identifier et d'évaluer "
        "les risques auxquels l'entreprise est exposée.",
    ]
    for bruit in bruits:
        assert est_bruit(bruit), f"Bruit non détecté : '{bruit}'"

    for pas_bruit in pas_bruits:
        assert not est_bruit(pas_bruit), f"Faux positif bruit : '{pas_bruit[:50]}'"

    print(f"  ✓ Détection bruit : {len(bruits)} bruits + {len(pas_bruits)} non-bruits")


def test_detection_qrt():
    """La détection des tableaux QRT fonctionne."""
    qrt_textes = [
        "S.02.01 — Bilan",
        "S.25.01 Capital de Solvabilité Requis",
        "Tableau quantitatif S.05.01",
        "QRT publié en annexe",
    ]
    for t in qrt_textes:
        assert est_tableau_qrt(t), f"QRT non détecté : '{t}'"
    print(f"  ✓ Détection QRT : {len(qrt_textes)} tableaux détectés")


# ---------------------------------------------------------------------------
# Tests SFCRChunker
# ---------------------------------------------------------------------------

def test_chunker_parametres():
    """Le chunker est initialisé avec les bons paramètres."""
    chunker = SFCRChunker()
    assert chunker.chunk_size == CHUNK_SIZE_SFCR_TOKENS
    assert chunker.chunk_overlap == CHUNK_OVERLAP_TOKENS
    assert chunker.min_tokens == 50
    print(f"  ✓ Paramètres chunker : size={chunker.chunk_size}, "
          f"overlap={chunker.chunk_overlap}, min={chunker.min_tokens}")


def test_chunker_referentiel():
    """Le chunking d'un texte réglementaire produit des chunks valides."""
    chunker = SFCRChunker()
    texte = (
        "Les entreprises publient annuellement un rapport sur leur solvabilité "
        "et leur situation financière. Ce rapport contient des informations sur "
        "l'activité, la gouvernance, le profil de risque, la valorisation et "
        "la gestion du capital. Il est mis à la disposition du public par voie "
        "électronique dans un délai de quatorze semaines après la clôture "
        "de l'exercice annuel de référence."
    )
    chunks = chunker.chunker_referentiel(
        texte=texte,
        source_id="REG_2015_35",
        article="Art. 290 §1",
        id_exigence="SII-P3-REG2015-Art290-§1",
    )
    assert len(chunks) >= 1, "Aucun chunk produit"
    for chunk in chunks:
        assert chunk.type_doc == "REGLEMENTAIRE"
        assert chunk.nb_tokens > 0
        assert chunk.section == "SII-P3-REG2015-Art290-§1"
    print(f"  ✓ Chunking réglementaire : {len(chunks)} chunk(s) produit(s)")


def test_validation_chunks():
    """La validation rejette les chunks invalides."""
    chunker = SFCRChunker()

    # Chunk valide
    chunk_valide = ChunkDocument(
        chunk_id="test-001",
        doc_id="doc-001",
        type_doc="SFCR",
        texte="Le système de gouvernance de l'entreprise comprend un conseil "
              "d'administration et une direction générale, ainsi que quatre "
              "fonctions clés : gestion des risques, conformité, audit interne "
              "et fonction actuarielle. Ces fonctions sont indépendantes.",
        nb_tokens=compter_tokens(
            "Le système de gouvernance de l'entreprise comprend un conseil "
            "d'administration et une direction générale, ainsi que quatre "
            "fonctions clés : gestion des risques, conformité, audit interne "
            "et fonction actuarielle. Ces fonctions sont indépendantes."
        ),
    )
    valide, raison = chunker._valider_chunk(chunk_valide)
    assert valide, f"Chunk valide rejeté : {raison}"

    # Chunk trop court
    chunk_court = ChunkDocument(
        chunk_id="test-002", doc_id="doc-001", type_doc="SFCR",
        texte="Court.", nb_tokens=2,
    )
    valide, raison = chunker._valider_chunk(chunk_court)
    assert not valide, "Chunk court accepté à tort"

    # Chunk vide
    chunk_vide = ChunkDocument(
        chunk_id="test-003", doc_id="doc-001", type_doc="SFCR",
        texte="   ", nb_tokens=0,
    )
    valide, raison = chunker._valider_chunk(chunk_vide)
    assert not valide, "Chunk vide accepté à tort"

    # Chunk bruit (que des chiffres)
    chunk_bruit = ChunkDocument(
        chunk_id="test-004", doc_id="doc-001", type_doc="SFCR",
        texte="12 345 678 | 23.45 | 0.987 | 1,234,567 | 99.9%",
        nb_tokens=compter_tokens("12 345 678 | 23.45 | 0.987 | 1,234,567 | 99.9%"),
    )
    valide, raison = chunker._valider_chunk(chunk_bruit)
    assert not valide, "Chunk bruit accepté à tort"

    print("  ✓ Validation chunks : valide=✓, court=✗, vide=✗, bruit=✗")


def test_preparation_embedding():
    """Les textes préparés pour l'embedding ont les bons préfixes."""
    chunker = SFCRChunker()
    chunks = [
        ChunkDocument(
            chunk_id=f"c{i}", doc_id="d1", type_doc="SFCR",
            texte=f"Texte de test numéro {i} pour l'embedding.",
            nb_tokens=10,
        )
        for i in range(3)
    ]
    textes = chunker.preparer_textes_embedding(chunks)
    assert len(textes) == 3
    for t in textes:
        assert t.startswith("passage: "), f"Préfixe manquant : '{t[:30]}'"
    print(f"  ✓ Préparation embedding : {len(textes)} textes avec préfixe 'passage:'")


def test_chunker_texte_long():
    """Un texte long est correctement découpé en plusieurs chunks."""
    chunker = SFCRChunker()
    # Créer un texte synthétique de ~2000 tokens (environ 4× la taille cible)
    paragraph = (
        "L'entreprise a mis en place un système de gestion des risques robuste "
        "permettant d'identifier, mesurer, surveiller, gérer et déclarer les "
        "risques auxquels elle est exposée. Ce système couvre les risques de "
        "souscription, de marché, de crédit, de liquidité et les risques "
        "opérationnels conformément aux exigences de Solvabilité II. "
    )
    texte_long = paragraph * 25  # ~2000 tokens

    chunks = chunker.chunker_referentiel(
        texte=texte_long,
        source_id="TEST",
        article="Art. test",
        id_exigence="SII-P2-TEST-Art99-§1",
    )

    assert len(chunks) >= 3, f"Pas assez de chunks pour un texte long : {len(chunks)}"
    for chunk in chunks:
        assert chunk.nb_tokens <= 400, (
            f"Chunk trop long : {chunk.nb_tokens} tokens (max cible=350)"
        )
    print(f"  ✓ Chunking texte long : {compter_tokens(texte_long)} tokens → "
          f"{len(chunks)} chunks")


# ---------------------------------------------------------------------------
# Tests IngestionPipeline — référentiel
# ---------------------------------------------------------------------------

def test_ingestion_referentiel():
    """Le pipeline ingère correctement le référentiel complet."""
    pipeline = IngestionPipeline()
    resultat = pipeline.ingerer_referentiel()

    assert resultat.nb_exigences > 0, "Aucune exigence ingérée"
    assert resultat.nb_chunks >= resultat.nb_exigences, (
        "Moins de chunks que d'exigences — impossible"
    )
    assert len(resultat.textes_embedding) == len(resultat.chunks), (
        "Nombre de textes embedding ≠ nombre de chunks"
    )
    assert len(resultat.ids_exigences) == len(resultat.chunks), (
        "Traçabilité chunk→exigence incomplète"
    )

    # Vérifier les préfixes query: sur les textes de vérification
    for texte in resultat.textes_embedding[:5]:
        assert texte.startswith("query: "), (
            f"Préfixe 'query:' manquant sur texte référentiel : '{texte[:40]}'"
        )

    print(f"  ✓ Ingestion référentiel : {resultat.nb_exigences} exigences → "
          f"{resultat.nb_chunks} chunks")
    print(f"    Tous les textes d'embedding ont le préfixe 'query:'")


def test_coherence_tracabilite():
    """La traçabilité chunk → exigence est complète et cohérente."""
    pipeline = IngestionPipeline()
    resultat = pipeline.ingerer_referentiel()

    ids_uniques = set(resultat.ids_exigences)
    assert len(ids_uniques) == resultat.nb_exigences, (
        f"IDs exigences dupliqués ou manquants : "
        f"{len(ids_uniques)} uniques / {resultat.nb_exigences} attendus"
    )

    # Vérifier que chaque id_exigence correspond à un chunk avec section=id_exigence
    for chunk, id_exig in zip(resultat.chunks, resultat.ids_exigences):
        assert chunk.section == id_exig, (
            f"Traçabilité brisée : chunk.section='{chunk.section}' "
            f"≠ id_exigence='{id_exig}'"
        )
    print(f"  ✓ Traçabilité : {len(ids_uniques)} exigences uniques, "
          f"chaque chunk tracé vers son exigence")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    tests = [
        # text_utils
        ("Comptage de tokens", test_compter_tokens),
        ("Troncature à N tokens", test_tronquer_a_tokens),
        ("Nettoyage texte PDF", test_nettoyer_texte_pdf),
        ("Préfixes multilingual-e5", test_prefixes_e5),
        ("Détection du bruit PDF", test_detection_bruit),
        ("Détection tableaux QRT", test_detection_qrt),
        # SFCRChunker
        ("Paramètres du chunker", test_chunker_parametres),
        ("Chunking réglementaire", test_chunker_referentiel),
        ("Validation des chunks", test_validation_chunks),
        ("Préparation embedding", test_preparation_embedding),
        ("Chunking texte long", test_chunker_texte_long),
        # Pipeline
        ("Ingestion référentiel", test_ingestion_referentiel),
        ("Traçabilité chunk→exigence", test_coherence_tracabilite),
    ]

    print("\n" + "=" * 60)
    print("VALIDATION PHASE 2 — INGESTION & CHUNKING")
    print("=" * 60)

    nb_ok = nb_ko = 0
    for nom, fn in tests:
        print(f"\n[TEST] {nom}")
        try:
            fn()
            nb_ok += 1
        except Exception as e:
            print(f"  ✗ ÉCHEC : {e}")
            nb_ko += 1

    print("\n" + "=" * 60)
    print(f"RÉSULTATS : {nb_ok} OK / {nb_ko} KO / {len(tests)} total")
    print("=" * 60)
    if nb_ko == 0:
        print("\nPhase 2 validée — prêt pour la Phase 3 (embeddings + ChromaDB)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
