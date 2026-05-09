"""
scripts/build_vectorstore.py — Construction des bases vectorielles
==================================================================
Script à exécuter UNE FOIS pour initialiser ChromaDB avec :
  1. Le référentiel Solvabilité II (71 exigences atomiques)
  2. Optionnellement un SFCR de test

Usage :
    # Indexer uniquement le référentiel
    python scripts/build_vectorstore.py

    # Indexer référentiel + un SFCR
    python scripts/build_vectorstore.py --sfcr path/to/rapport.pdf
                                        --entreprise "AXA France"
                                        --annee 2023

    # Réindexer depuis zéro
    python scripts/build_vectorstore.py --reset
"""

import sys
import logging
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_vectorstore")


def build_referentiel(store, engine, loader, forcer: bool = False) -> int:
    """Construit et indexe le référentiel réglementaire."""
    import numpy as np
    from src.ingestion.ingestion_pipeline import IngestionPipeline
    from src.ingestion.text_utils import preparer_requete_e5

    logger.info("=" * 55)
    logger.info("ÉTAPE 1 — Ingestion du référentiel réglementaire")
    logger.info("=" * 55)

    pipeline = IngestionPipeline()
    resultat = pipeline.ingerer_referentiel()
    logger.info(resultat.rapport())

    logger.info("ÉTAPE 2 — Génération des embeddings référentiel")
    t0 = time.time()

    textes = resultat.textes_embedding
    embeddings = engine.embed_documents(textes)
    logger.info(
        f"Embeddings générés : {embeddings.shape} "
        f"en {time.time() - t0:.1f}s"
    )

    logger.info("ÉTAPE 3 — Indexation ChromaDB référentiel")
    # Construire le mapping id_exigence → ExigenceAtomique pour métadonnées
    loader.charger(strict=True)
    exigences_map = {
        e.id_exigence: e
        for e in loader.get_toutes_exigences()
    }

    nb = store.indexer_referentiel(
        chunks=resultat.chunks,
        embeddings=embeddings,
        ids_exigences=resultat.ids_exigences,
        exigences_map=exigences_map,
        forcer_reimport=forcer,
    )
    logger.info(f"✓ Référentiel indexé : {nb} chunks")
    return nb


def build_sfcr(store, engine, pdf_path: Path, entreprise: str, annee: int) -> int:
    """Ingère et indexe un rapport SFCR."""
    import numpy as np
    from src.ingestion.ingestion_pipeline import IngestionPipeline

    logger.info("=" * 55)
    logger.info(f"INDEXATION SFCR — {entreprise} {annee}")
    logger.info("=" * 55)

    pipeline = IngestionPipeline()
    resultat = pipeline.ingerer_sfcr(pdf_path, entreprise, annee)

    if not resultat.pret_pour_indexation:
        logger.error(f"SFCR non indexable : {resultat.erreurs}")
        return 0

    logger.info(resultat.rapport())

    logger.info("Génération des embeddings SFCR...")
    t0 = time.time()
    embeddings = engine.embed_documents(resultat.textes_embedding)
    logger.info(f"Embeddings générés : {embeddings.shape} en {time.time() - t0:.1f}s")

    nb = store.indexer_sfcr(
        chunks=resultat.chunks,
        embeddings=embeddings,
        doc_id=resultat.doc_sfcr.id,
        entreprise=entreprise,
        annee=annee,
    )
    logger.info(f"✓ SFCR indexé : {nb} chunks")
    return nb


def main():
    parser = argparse.ArgumentParser(
        description="Construction des bases vectorielles RegWatch"
    )
    parser.add_argument("--sfcr", type=Path, help="Chemin vers un PDF SFCR à indexer")
    parser.add_argument("--entreprise", type=str, default="Entreprise test")
    parser.add_argument("--annee", type=int, default=2023)
    parser.add_argument("--reset", action="store_true",
                        help="Réinitialise toutes les collections avant indexation")
    parser.add_argument("--mode-embedding", choices=["auto", "local", "production"],
                        default="auto", help="Moteur d'embedding à utiliser")
    parser.add_argument("--vectorstore-dir", type=Path,
                        default=Path("data/vectorstore"))
    args = parser.parse_args()

    from src.rag.embedding_engine import creer_embedding_engine
    from src.rag.vector_store import VectorStore
    from src.ingestion.referentiel_loader import ReferentielLoader

    # ── Initialisation ────────────────────────────────────────────────
    logger.info("Initialisation du VectorStore...")
    store = VectorStore(persist_dir=args.vectorstore_dir)

    # Pour le mode local : corpus de fit = textes du référentiel
    loader = ReferentielLoader()
    loader.charger(strict=True)
    corpus_fit = [e.texte_normalise for e in loader.get_toutes_exigences()]
    corpus_fit += [e.texte_verification for e in loader.get_toutes_exigences()]

    engine = creer_embedding_engine(
        mode=args.mode_embedding,
        dimension=256,
        corpus_fit=corpus_fit,
    )

    store.initialiser(engine)

    if args.reset:
        logger.warning("Reset complet du VectorStore...")
        store.reset_complet()
        store.initialiser(engine)

    # ── Indexation référentiel ────────────────────────────────────────
    if store.est_referentiel_indexe() and not args.reset:
        logger.info("Référentiel déjà indexé — skip (utiliser --reset pour réindexer)")
    else:
        build_referentiel(store, engine, loader, forcer=args.reset)

    # ── Indexation SFCR optionnel ─────────────────────────────────────
    if args.sfcr:
        if not args.sfcr.exists():
            logger.error(f"Fichier PDF introuvable : {args.sfcr}")
            sys.exit(1)
        build_sfcr(store, engine, args.sfcr, args.entreprise, args.annee)

    # ── Rapport final ─────────────────────────────────────────────────
    logger.info("\n" + store.afficher_stats())
    logger.info("✓ VectorStore prêt — Phase 3 terminée")


if __name__ == "__main__":
    main()
