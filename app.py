"""
app.py — Interface Streamlit RegWatch
======================================
Interface utilisateur complète pour l'analyse de conformité SFCR.

Lancement :
    streamlit run app.py

Architecture de l'interface :
  Sidebar  → Upload PDF + paramètres + déclencheur d'analyse
  Tab 1    → Vue d'ensemble : KPIs, radar, barres piliers
  Tab 2    → Détail par thème : tableau interactif, drill-down exigences
  Tab 3    → Recommandations + export JSON/CSV
  Tab 4    → À propos : architecture, choix techniques
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import uuid
import shutil
import tempfile
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

# ── Configuration Streamlit ────────────────────────────────────────────────
st.set_page_config(
    page_title="RegWatch — Conformité Solvabilité II",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS personnalisé ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Palette couleurs conformité */
    .niveau-A { background:#e8f5e9; color:#1b5e20;
                border-left:4px solid #2e7d32; padding:8px 12px;
                border-radius:4px; font-weight:bold; }
    .niveau-B { background:#fff8e1; color:#e65100;
                border-left:4px solid #f9a825; padding:8px 12px;
                border-radius:4px; font-weight:bold; }
    .niveau-C { background:#fff3e0; color:#bf360c;
                border-left:4px solid #ef6c00; padding:8px 12px;
                border-radius:4px; font-weight:bold; }
    .niveau-D { background:#ffebee; color:#b71c1c;
                border-left:4px solid #c62828; padding:8px 12px;
                border-radius:4px; font-weight:bold; }
    /* Métriques KPI */
    .kpi-box { background:#f8f9fa; border-radius:8px;
               padding:16px; text-align:center;
               border:1px solid #e0e0e0; }
    .kpi-val { font-size:2rem; font-weight:bold; color:#1a237e; }
    .kpi-lab { font-size:0.85rem; color:#666; margin-top:4px; }
    /* Header */
    .main-header { background:linear-gradient(135deg,#1a237e,#283593);
                   color:white; padding:20px 24px; border-radius:8px;
                   margin-bottom:20px; }
    /* Statut badges */
    .badge-couvert  { background:#c8e6c9; color:#1b5e20;
                      padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-partiel  { background:#fff9c4; color:#f57f17;
                      padding:2px 8px; border-radius:12px; font-size:0.8rem; }
    .badge-absent   { background:#ffcdd2; color:#b71c1c;
                      padding:2px 8px; border-radius:12px; font-size:0.8rem; }
</style>
""", unsafe_allow_html=True)


# ── Initialisation session state ───────────────────────────────────────────
def init_session():
    defaults = {
        "rapport": None,
        "scores_exig": None,
        "scores_themes": None,
        "scores_piliers": None,
        "resultat_rag": None,
        "analyse_lancee": False,
        "entreprise": "",
        "annee": 2023,
        "vs_dir": None,
        "engine_emb": None,
        "store": None,
        "loader": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Chargement des ressources (mis en cache) ───────────────────────────────
@st.cache_resource(show_spinner="Chargement du référentiel Solvabilité II...")
def charger_referentiel():
    from src.ingestion.referentiel_loader import ReferentielLoader
    loader = ReferentielLoader()
    loader.charger(strict=True)
    return loader


@st.cache_resource(show_spinner="Initialisation du moteur d'embedding...")
def initialiser_engine(_loader):
    from src.rag.embedding_manager import EmbeddingManager
    exigences = _loader.get_toutes_exigences()
    textes_fit = (
        [e.texte_normalise for e in exigences] +
        [e.texte_verification for e in exigences]
    )
    manager = EmbeddingManager()
    manager.initialiser(corpus_entrainement=textes_fit)
    return manager


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8rem;">🔍 RegWatch</h1>
    <p style="margin:4px 0 0 0;opacity:0.85;font-size:1rem;">
        Analyse automatique de conformité Solvabilité II · Architecture RAG
    </p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Paramètres d'analyse")

    uploaded_file = st.file_uploader(
        "Rapport SFCR (PDF)",
        type=["pdf"],
        help="Déposez le rapport SFCR au format PDF. "
             "Le système extrait automatiquement les sections A-E.",
    )

    entreprise = st.text_input(
        "Entreprise",
        placeholder="Ex : AXA France Vie",
        help="Nom de l'entreprise d'assurance",
    )

    annee = st.number_input(
        "Exercice de référence",
        min_value=2016,
        max_value=2030,
        value=2023,
        step=1,
    )

    st.divider()

    st.subheader("⚙️ Configuration LLM")
    mode_llm = st.selectbox(
        "Moteur d'analyse",
        options=["local (développement)", "anthropic (Claude)", "ollama (Mistral)", "groq (Llama-3.3-70B)"],
        index=0,
        help=(
            "**local** : RuleBasedLLM offline, gratuit, déterministe\n"
            "**anthropic** : Claude Sonnet, meilleure qualité (clé API requise)\n"
            "**ollama** : Mistral-7B local (Ollama installé requis)"
            "**llama3** : groq (Llama-3.3-70B)"
        ),
    )

    api_key = ""
    if "anthropic" in mode_llm:
        api_key = st.text_input(
            "Clé API Anthropic",
            type="password",
            placeholder="sk-ant-...",
        )

    top_k = st.slider(
        "Passages candidats (top-K)",
        min_value=1, max_value=10, value=5,
        help="Nombre de passages SFCR évalués par exigence. "
             "Plus = meilleure couverture, plus lent.",
    )

    st.divider()

    filtre_themes = st.multiselect(
        "Filtrer par thème (optionnel)",
        options=[
            "T01 — Valorisation actifs/passifs",
            "T02 — SCR / MCR",
            "T03 — Fonds propres",
            "T04 — Gouvernance",
            "T05 — Gestion risques / ORSA",
            "T06 — Contrôle interne / Audit",
            "T07 — Fonction actuarielle",
            "T08 — Structure SFCR",
            "T09 — QRT",
            "T10 — Transparence",
            "T11 — Délais publication",
        ],
        help="Laissez vide pour analyser les 3 piliers complets (71 exigences)",
    )

    themes_selectionnes = (
        [t[:3] for t in filtre_themes] if filtre_themes else None
    )

    st.divider()

    lancer = st.button(
        "🚀 Lancer l'analyse",
        type="primary",
        use_container_width=True,
        disabled=not (uploaded_file and entreprise),
    )

    if not uploaded_file:
        st.info("⬆️ Déposez un PDF SFCR pour commencer")
    elif not entreprise:
        st.warning("Renseignez le nom de l'entreprise")


# ── Pipeline d'analyse ─────────────────────────────────────────────────────
if lancer and uploaded_file and entreprise:
    with st.spinner("🔄 Analyse en cours — veuillez patienter..."):
        try:
            from src.ingestion.ingestion_pipeline import IngestionPipeline
            from src.rag.vector_store import VectorStore
            from src.rag.llm_engine import creer_llm_engine
            from src.rag.rag_pipeline import RAGPipeline
            from src.scoring.scoring_engine import ScoringEngine

            # Charger le référentiel et l'engine
            loader = charger_referentiel()
            engine_emb = initialiser_engine(loader)

            # Sauvegarder le PDF uploadé temporairement
            with tempfile.NamedTemporaryFile(
                suffix=".pdf", delete=False
            ) as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = Path(tmp.name)

            # Initialiser ChromaDB (répertoire temporaire unique)
            if st.session_state.vs_dir:
                shutil.rmtree(st.session_state.vs_dir, ignore_errors=True)
            vs_dir = Path(tempfile.mkdtemp(prefix="regwatch_"))
            st.session_state.vs_dir = vs_dir

            store = VectorStore(persist_dir=vs_dir)
            store.initialiser(engine_emb)

            # Indexer le référentiel
            exigences = loader.get_toutes_exigences()
            embs_ref = engine_emb.encoder_requetes(
                [e.texte_verification for e in exigences]
            )
            store.indexer_referentiel(exigences, embs_ref)

            # Ingérer le SFCR
            st.info("📄 Parsing du PDF en cours...")
            pipeline_ing = IngestionPipeline()
            res_ing = pipeline_ing.ingerer_sfcr(
                pdf_path=pdf_path,
                entreprise=entreprise,
                annee=int(annee),
            )

            if not res_ing.pret_pour_indexation:
                st.error(
                    f"❌ Erreur d'ingestion : {res_ing.erreurs}\n"
                    "Vérifiez que le PDF est un rapport SFCR valide "
                    "et non protégé."
                )
                st.stop()

            st.info(
                f"📊 SFCR ingéré : {res_ing.resultat_chunking.nb_chunks} chunks "
                f"| {res_ing.doc_parse.nb_pages} pages"
            )

            embs_sfcr = engine_emb.encoder_passages(
                [t.replace("passage: ", "") for t in res_ing.textes_embedding]
            )
            store.indexer_sfcr(
                res_ing.chunks,
                embs_sfcr,
                entreprise,
                int(annee),
            )

            # Configurer le LLM
            mode_map = {
                "local (développement)": "local",
                "anthropic (Claude)": "anthropic",
                "ollama (Mistral)": "ollama",
                "groq (Llama-3.3-70B)": "groq"
            }
            llm = creer_llm_engine(
                mode=mode_map[mode_llm],
                api_key=api_key or None,
            )

            # RAG
            st.info(
                f"🤖 Analyse RAG : "
                f"{len(exigences) if not themes_selectionnes else len([e for e in exigences if e.theme_id in themes_selectionnes])} "
                f"exigences × top-{top_k} passages..."
            )
            # RAGPipeline attend un objet avec embed_query() et embed_documents()
            # EmbeddingManager expose encoder_requetes/encoder_passages
            # On crée un adaptateur léger
            class _EngineAdapter:
                def __init__(self, mgr):
                    self._mgr = mgr
                @property
                def dimension(self): return self._mgr.dimension
                @property
                def nom_modele(self): return self._mgr.nom_modele
                def embed_query(self, texte):
                    import numpy as np
                    return np.array(self._mgr.encoder_requetes([texte])[0])
                def embed_documents(self, textes):
                    import numpy as np
                    return np.array(self._mgr.encoder_passages(textes))
                def similarite_cosinus(self, a, b):
                    return float(a @ b)
            rag = RAGPipeline(_EngineAdapter(engine_emb), store, llm, top_k=top_k)
            res_rag = rag.analyser_sfcr(
                exigences=exigences,
                doc_id=res_ing.doc_sfcr.id,
                entreprise=entreprise,
                annee=int(annee),
                filtre_themes=themes_selectionnes,
            )

            # Scoring
            scoring = ScoringEngine(loader)
            rapport = scoring.calculer(res_rag)
            scores_exig = scoring._calculer_scores_exigences(res_rag.matchings)
            scores_themes = scoring._calculer_scores_themes(scores_exig)
            scores_piliers = scoring._calculer_scores_piliers(scores_themes)

            # Sauvegarder dans session state
            st.session_state.rapport = rapport
            st.session_state.scores_exig = scores_exig
            st.session_state.scores_themes = scores_themes
            st.session_state.scores_piliers = scores_piliers
            st.session_state.resultat_rag = res_rag
            st.session_state.analyse_lancee = True
            st.session_state.entreprise = entreprise
            st.session_state.annee = annee

            pdf_path.unlink(missing_ok=True)
            st.success("✅ Analyse terminée !")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erreur inattendue : {e}")
            import traceback
            st.code(traceback.format_exc())


# ── Affichage des résultats ────────────────────────────────────────────────
if st.session_state.analyse_lancee and st.session_state.rapport:
    rapport        = st.session_state.rapport
    scores_exig    = st.session_state.scores_exig
    scores_themes  = st.session_state.scores_themes
    scores_piliers = st.session_state.scores_piliers

    from src.ingestion.referentiel_solvabilite2 import PILIERS, THEMES
    from src.ingestion.models import StatutConformite, NiveauObligation

    tabs = st.tabs([
        "📊 Vue d'ensemble",
        "🔍 Détail par thème",
        "💡 Recommandations",
        "ℹ️ À propos",
    ])

    # ── TAB 1 : Vue d'ensemble ─────────────────────────────────────────
    with tabs[0]:
        # Niveau et score global
        nv = rapport.niveau_conformite.value
        niveau_labels = {
            "A": ("🟢 Niveau A — Conforme", "niveau-A"),
            "B": ("🟡 Niveau B — Satisfaisant", "niveau-B"),
            "C": ("🟠 Niveau C — Insuffisant", "niveau-C"),
            "D": ("🔴 Niveau D — Non conforme", "niveau-D"),
        }
        label_nv, classe_nv = niveau_labels[nv]
        st.markdown(
            f'<div class="{classe_nv}" style="font-size:1.2rem;">'
            f'{label_nv} &nbsp;·&nbsp; '
            f'Score global : <strong>{rapport.score_global:.3f} / 1.000</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        kpis = [
            (c1, f"{rapport.score_global:.1%}", "Score global"),
            (c2, str(rapport.nb_exigences_couvertes), "✅ Couvertes"),
            (c3, str(rapport.nb_exigences_partielles), "⚠️ Partielles"),
            (c4, str(rapport.nb_exigences_absentes), "❌ Absentes"),
            (c5, str(rapport.nb_exigences_analysees), "Total analysées"),
        ]
        for col, val, lab in kpis:
            col.metric(label=lab, value=val)

        st.divider()

        # Graphes
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("Scores par pilier")

            piliers_noms = [
                f"P1 — {PILIERS['P1'].intitule[:20]}",
                f"P2 — {PILIERS['P2'].intitule[:20]}",
                f"P3 — {PILIERS['P3'].intitule[:20]}",
            ]
            piliers_scores = [
                rapport.score_pilier_1,
                rapport.score_pilier_2,
                rapport.score_pilier_3,
            ]
            couleurs_barres = [
                "#4CAF50" if s >= 0.80 else
                "#FFC107" if s >= 0.65 else
                "#FF9800" if s >= 0.45 else "#f44336"
                for s in piliers_scores
            ]

            fig_barres = go.Figure(go.Bar(
                x=piliers_noms,
                y=piliers_scores,
                marker_color=couleurs_barres,
                text=[f"{s:.1%}" for s in piliers_scores],
                textposition="outside",
            ))
            fig_barres.add_hline(
                y=0.75, line_dash="dash", line_color="green",
                annotation_text="Seuil COUVERT (75%)",
                annotation_position="top right",
            )
            fig_barres.add_hline(
                y=0.45, line_dash="dash", line_color="orange",
                annotation_text="Seuil PARTIEL (45%)",
                annotation_position="top right",
            )
            fig_barres.update_layout(
                yaxis_range=[0, 1.15],
                yaxis_title="Score de conformité",
                showlegend=False,
                height=350,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_barres, use_container_width=True)

        with col_right:
            st.subheader("Radar par thème")

            themes_actifs = [
                (tid, st_th)
                for tid, st_th in scores_themes.items()
                if st_th.nb_exigences_total > 0
            ]
            if themes_actifs:
                theta = [f"{t[0]}" for t in themes_actifs]
                r     = [t[1].score_brut for t in themes_actifs]
                theta_closed = theta + [theta[0]]
                r_closed     = r + [r[0]]

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_closed,
                    theta=theta_closed,
                    fill="toself",
                    fillcolor="rgba(33,150,243,0.15)",
                    line_color="#2196F3",
                    name="Score conformité",
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=[0.75] * (len(theta) + 1),
                    theta=theta_closed,
                    mode="lines",
                    line_color="green",
                    line_dash="dash",
                    name="Seuil COUVERT (75%)",
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True, range=[0, 1],
                            tickformat=".0%",
                        )
                    ),
                    showlegend=True,
                    height=350,
                    margin=dict(t=20, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # Distribution des statuts
        st.subheader("Distribution des statuts par thème")
        rows = []
        for tid, st_th in sorted(scores_themes.items()):
            if st_th.nb_exigences_total == 0:
                continue
            rows.append({
                "Thème": f"[{tid}] {st_th.libelle_theme[:35]}",
                "Score": st_th.score_brut,
                "✅ Couvertes": st_th.nb_couvertes,
                "⚠️ Partielles": st_th.nb_partielles,
                "❌ Absentes": st_th.nb_absentes,
                "Total": st_th.nb_exigences_total,
                "Niveau": (
                    "A" if st_th.score_brut >= 0.80 else
                    "B" if st_th.score_brut >= 0.65 else
                    "C" if st_th.score_brut >= 0.45 else "D"
                ),
            })

        if rows:
            df = pd.DataFrame(rows)
            fig_stack = px.bar(
                df,
                x="Thème",
                y=["✅ Couvertes", "⚠️ Partielles", "❌ Absentes"],
                color_discrete_map={
                    "✅ Couvertes": "#4CAF50",
                    "⚠️ Partielles": "#FFC107",
                    "❌ Absentes": "#f44336",
                },
                labels={"value": "Nb exigences", "variable": "Statut"},
                height=380,
            )
            fig_stack.update_layout(
                xaxis_tickangle=-35,
                margin=dict(t=20, b=100),
                legend_title="Statut",
            )
            st.plotly_chart(fig_stack, use_container_width=True)

    # ── TAB 2 : Détail par thème ───────────────────────────────────────
    with tabs[1]:
        st.subheader("Détail par thème réglementaire")

        # Tableau synthèse
        data_table = []
        for tid, st_th in sorted(scores_themes.items()):
            if st_th.nb_exigences_total == 0:
                continue
            niv = (
                "🟢 A" if st_th.score_brut >= 0.80 else
                "🟡 B" if st_th.score_brut >= 0.65 else
                "🟠 C" if st_th.score_brut >= 0.45 else "🔴 D"
            )
            data_table.append({
                "ID": tid,
                "Thème": st_th.libelle_theme,
                "Score": f"{st_th.score_brut:.3f}",
                "Niveau": niv,
                "✅": st_th.nb_couvertes,
                "⚠️": st_th.nb_partielles,
                "❌": st_th.nb_absentes,
                "SHA absents": len(st_th.exigences_critiques_absentes),
            })

        df_table = pd.DataFrame(data_table)
        st.dataframe(
            df_table,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score",
                    min_value=0,
                    max_value=1,
                    format="%.3f",
                ),
                "SHA absents": st.column_config.NumberColumn(
                    "SHALL absents",
                    help="Exigences SHALL non couvertes — non-conformités critiques",
                ),
            },
        )

        st.divider()
        st.subheader("Drill-down par exigence")

        # Sélection du thème
        themes_dispo = [
            f"{tid} — {scores_themes[tid].libelle_theme}"
            for tid in sorted(scores_themes.keys())
            if scores_themes[tid].nb_exigences_total > 0
        ]
        theme_choisi = st.selectbox(
            "Sélectionner un thème",
            options=themes_dispo,
        )
        tid_choisi = theme_choisi[:3]

        # Filtre statut
        filtre_statut = st.radio(
            "Filtrer par statut",
            ["Tous", "✅ COUVERT", "⚠️ PARTIEL", "❌ ABSENT"],
            horizontal=True,
        )

        # Affichage des exigences du thème
        exig_theme = [
            se for se in scores_exig
            if se.theme_id == tid_choisi
        ]
        if filtre_statut != "Tous":
            statut_map = {
                "✅ COUVERT": "COUVERT",
                "⚠️ PARTIEL": "PARTIEL",
                "❌ ABSENT":  "ABSENT",
            }
            exig_theme = [
                se for se in exig_theme
                if se.statut.value == statut_map[filtre_statut]
            ]

        if not exig_theme:
            st.info("Aucune exigence correspondant aux filtres sélectionnés.")
        else:
            for se in sorted(exig_theme, key=lambda x: x.score_brut):
                emoji = {
                    "COUVERT": "✅",
                    "PARTIEL": "⚠️",
                    "ABSENT":  "❌",
                }[se.statut.value]
                oblig_color = {
                    "SHALL":  "🔴",
                    "SHOULD": "🟡",
                    "MAY":    "🟢",
                }[se.niveau_obligation.value]

                with st.expander(
                    f"{emoji} {se.id_exigence} "
                    f"| {oblig_color} {se.niveau_obligation.value} "
                    f"| Score : {se.score_brut:.3f}",
                    expanded=False,
                ):
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric("Score brut", f"{se.score_brut:.3f}")
                    col_b.metric("Score pondéré", f"{se.score_pondere:.3f}")
                    col_c.metric("Statut", se.statut.value)

                    if se.passage_trouve:
                        st.markdown("**📄 Passage SFCR le plus pertinent :**")
                        st.info(se.passage_trouve[:600])

                    st.markdown("**🤖 Justification LLM :**")
                    st.write(se.justification)

    # ── TAB 3 : Recommandations + Export ──────────────────────────────
    with tabs[2]:
        st.subheader("💡 Recommandations")

        for i, rec in enumerate(rapport.recommandations, 1):
            if "CRITIQUE" in rec or "🔴" in rec:
                st.error(f"**{i}.** {rec}")
            elif "INSUFFISANT" in rec or "🟠" in rec:
                st.warning(f"**{i}.** {rec}")
            elif "PARTIEL" in rec or "🟡" in rec:
                st.info(f"**{i}.** {rec}")
            else:
                st.success(f"**{i}.** {rec}")

        st.divider()
        st.subheader("📥 Export des résultats")

        from src.scoring.rapport_exporter import RapportExporter
        from src.scoring.scoring_engine import ScoringEngine
        from src.ingestion.referentiel_loader import ReferentielLoader

        col_exp1, col_exp2 = st.columns(2)

        # Export JSON
        with col_exp1:
            st.markdown("**JSON — Rapport complet**")
            st.caption(
                "Structure complète : métadonnées, scores par pilier/thème/"
                "exigence, passages trouvés, justifications LLM."
            )

            loader_exp = charger_referentiel()
            scoring_exp = ScoringEngine(loader_exp)
            sc_piliers = scoring_exp._calculer_scores_piliers(scores_themes)

            payload_json = {
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
                    "nb_analysees": rapport.nb_exigences_analysees,
                    "nb_couvertes": rapport.nb_exigences_couvertes,
                    "nb_partielles": rapport.nb_exigences_partielles,
                    "nb_absentes": rapport.nb_exigences_absentes,
                },
                "piliers": [
                    {
                        "id": pid,
                        "score": sp.score_brut,
                        "niveau": sp.niveau,
                    }
                    for pid, sp in sc_piliers.items()
                ],
                "themes": [
                    {
                        "id": tid,
                        "libelle": st_th.libelle_theme,
                        "score": st_th.score_brut,
                        "couvertes": st_th.nb_couvertes,
                        "partielles": st_th.nb_partielles,
                        "absentes": st_th.nb_absentes,
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

            st.download_button(
                label="⬇️ Télécharger JSON",
                data=json.dumps(payload_json, ensure_ascii=False, indent=2),
                file_name=f"regwatch_{rapport.entreprise.replace(' ','_')}_{rapport.annee_rapport}.json",
                mime="application/json",
                use_container_width=True,
            )

        # Export CSV
        with col_exp2:
            st.markdown("**CSV — Tableau par exigence**")
            st.caption(
                "Format Excel pour les actuaires : "
                "une ligne par exigence avec statut, score et justification."
            )

            rows_csv = []
            for se in sorted(scores_exig, key=lambda x: x.theme_id):
                st_th = scores_themes.get(se.theme_id)
                rows_csv.append({
                    "id_exigence": se.id_exigence,
                    "theme_id": se.theme_id,
                    "libelle_theme": st_th.libelle_theme if st_th else "",
                    "niveau_obligation": se.niveau_obligation.value,
                    "poids_obligation": se.poids_obligation,
                    "statut": se.statut.value,
                    "score_brut": round(se.score_brut, 4),
                    "score_pondere": round(se.score_pondere, 4),
                    "score_theme": round(st_th.score_brut, 4) if st_th else 0,
                    "justification": se.justification[:200],
                })

            df_csv = pd.DataFrame(rows_csv)
            st.download_button(
                label="⬇️ Télécharger CSV",
                data=df_csv.to_csv(index=False, encoding="utf-8-sig"),
                file_name=f"regwatch_{rapport.entreprise.replace(' ','_')}_{rapport.annee_rapport}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    # ── TAB 4 : À propos ──────────────────────────────────────────────
    with tabs[3]:
        st.subheader("Architecture RegWatch")

        col_arch1, col_arch2 = st.columns(2)

        with col_arch1:
            st.markdown("""
**Stack technologique**

| Couche | Outil | Justification |
|---|---|---|
| Parsing PDF | PyMuPDF | Le plus rapide, conserve la typographie |
| Chunking | LangChain RecursiveCharacterTextSplitter | Respecte les frontières naturelles |
| Embeddings | `multilingual-e5-large` | Top-3 MTEB, natif FR/EN |
| Base vectorielle | ChromaDB | Persistant + filtrage métadonnées |
| LLM | Claude Sonnet / Mistral / RuleBased | Interface découplée |
| Interface | Streamlit | Standard Data Science |
""")

        with col_arch2:
            st.markdown("""
**Formule de scoring**

```
score_final = 0.40 × cosinus + 0.60 × LLM

score_thème = Σ(score × poids_obligation)
            / Σ(poids_obligation)

score_pilier = Σ(score_thème × poids_thème)

score_global = Σ(score_pilier × poids_pilier)
             P1×0.30 + P2×0.35 + P3×0.35

Niveau A ≥ 0.80 | B ≥ 0.65 | C ≥ 0.45 | D < 0.45
```
""")

        st.divider()
        st.markdown("""
**Référentiel Solvabilité II**
- **71 exigences atomiques** extraites de 3 sources officielles
- Directive 2009/138/CE | Règlement délégué 2015/35 | EIOPA-BoS-15/109
- 3 piliers × 11 thèmes × SHALL/SHOULD/MAY

**Pipeline RAG**
1. Embedding `texte_verification` avec préfixe `query:`
2. Recherche top-K dans ChromaDB (distance cosinus)
3. Analyse LLM par passage candidat → score sémantique
4. Score final = 40% vectoriel + 60% sémantique
5. Statut : COUVERT ≥ 0.75 | PARTIEL ≥ 0.45 | ABSENT < 0.45
""")


# ── État initial (aucune analyse) ──────────────────────────────────────────
else:
    st.markdown("""
### Comment utiliser RegWatch

1. **Déposez** votre rapport SFCR (PDF) dans la barre latérale gauche
2. **Renseignez** l'entreprise et l'exercice de référence
3. **Choisissez** le moteur LLM (local pour le développement)
4. **Cliquez** sur *Lancer l'analyse*

L'outil analyse automatiquement la conformité du rapport aux
**71 exigences réglementaires** Solvabilité II réparties sur 3 piliers et 11 thèmes.
""")

    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.info("**Pilier 1** — Exigences quantitatives\nValorisation · SCR · MCR · Fonds propres")
    with col_d2:
        st.info("**Pilier 2** — Gouvernance & Contrôle\nGouvernance · ORSA · Audit · Actuariat")
    with col_d3:
        st.info("**Pilier 3** — Reporting & Transparence\nSFCR · QRT · Délais · Communication")
