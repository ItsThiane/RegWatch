"""
llm_engine.py — Moteur LLM découplé pour l'analyse de conformité
================================================================
Architecture à trois backends interchangeables :

  LLMEngine (interface abstraite)
      ├── AnthropicLLM   ← Production (API Claude Sonnet)
      ├── OllamaLLM      ← Production alternative (Mistral-7B local)
      └── RuleBasedLLM   ← Développement offline (règles déterministes)

Rôle du LLM dans le pipeline RAG :
  Pour chaque paire (exigence, passage_SFCR), le LLM répond à trois
  questions précises formulées dans texte_verification :
    1. Le passage couvre-t-il l'exigence ? (score 0.0 → 1.0)
    2. Quelle est la qualité de la couverture ?
    3. Justification en langage naturel

Pourquoi Claude Sonnet en production ?
  - Meilleur raisonnement juridique/réglementaire parmi les LLM du marché
  - Compréhension native FR/EN sans dégradation de qualité
  - Réponse structurée fiable (JSON strict)
  - Contexte 200K tokens → peut analyser des passages longs
  - Alternative : Mistral-7B via Ollama (gratuit, local, 80% des perf)

Pourquoi un RuleBasedLLM en développement ?
  - Zéro latence, zéro coût, zéro réseau
  - Déterministe → tests reproductibles
  - Suffit pour valider le pipeline complet sans LLM réel
  - Score basé sur des heuristiques lexicales calibrées sur Solvabilité II
"""

from __future__ import annotations
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structure de réponse LLM
# ---------------------------------------------------------------------------

@dataclass
class AnalyseLLM:
    """
    Résultat de l'analyse LLM pour une paire (exigence, passage).

    score         : Niveau de couverture [0.0, 1.0]
                    1.0 = couverture complète et explicite
                    0.5 = couverture partielle ou implicite
                    0.0 = non couvert
    justification : Explication en langage naturel (1-3 phrases)
    elements_trouves : Éléments spécifiques de l'exigence trouvés
    elements_manquants: Éléments de l'exigence absents du passage
    confiance     : Niveau de confiance dans l'analyse [0.0, 1.0]
    tokens_utilises: Nombre de tokens consommés (monitoring coût)
    """
    score: float
    justification: str
    elements_trouves: list[str]
    elements_manquants: list[str]
    confiance: float = 1.0
    tokens_utilises: int = 0
    backend_utilise: str = "inconnu"

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))
        self.confiance = max(0.0, min(1.0, self.confiance))

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "justification": self.justification,
            "elements_trouves": self.elements_trouves,
            "elements_manquants": self.elements_manquants,
            "confiance": round(self.confiance, 4),
            "tokens_utilises": self.tokens_utilises,
            "backend_utilise": self.backend_utilise,
        }


# ---------------------------------------------------------------------------
# Interface abstraite
# ---------------------------------------------------------------------------

class LLMEngine(ABC):
    """
    Interface abstraite pour le moteur LLM d'analyse de conformité.

    Tout backend doit implémenter analyser_conformite() qui prend
    une exigence réglementaire et un passage SFCR et retourne
    une AnalyseLLM structurée.
    """

    @abstractmethod
    def analyser_conformite(
        self,
        texte_verification: str,
        passage_sfcr: str,
        id_exigence: str,
        niveau_obligation: str = "SHALL",
    ) -> AnalyseLLM:
        """
        Analyse si un passage SFCR couvre une exigence réglementaire.

        Args:
            texte_verification : Question précise sur l'exigence
                                 (ex: "Le rapport décrit-il la gouvernance ?")
            passage_sfcr       : Extrait du rapport SFCR à analyser
            id_exigence        : Identifiant pour le logging
            niveau_obligation  : SHALL / SHOULD / MAY

        Returns:
            AnalyseLLM avec score, justification et éléments identifiés
        """
        ...

    @property
    @abstractmethod
    def nom_backend(self) -> str:
        """Nom du backend pour la documentation."""
        ...

    def _construire_prompt(
        self,
        texte_verification: str,
        passage_sfcr: str,
        niveau_obligation: str,
    ) -> str:
        """
        Construit le prompt d'analyse de conformité.

        Format optimisé pour une réponse JSON structurée.
        Instruit le LLM à se comporter comme un expert Solvabilité II.
        """
        return f"""Tu es un expert en conformité réglementaire Solvabilité II et en audit des rapports SFCR (Solvency and Financial Condition Report).

Ta mission : analyser si le passage suivant d'un rapport SFCR couvre l'exigence réglementaire posée.

EXIGENCE À VÉRIFIER (niveau {niveau_obligation}) :
{texte_verification}

PASSAGE DU RAPPORT SFCR :
\"\"\"
{passage_sfcr}
\"\"\"

INSTRUCTIONS :
1. Lis attentivement l'exigence et le passage.
2. Évalue si le passage couvre explicitement ou implicitement l'exigence.
3. Identifie les éléments présents et manquants.
4. Attribue un score de couverture strict (pas de générosité excessive).

Réponds UNIQUEMENT en JSON valide, sans texte avant ni après :
{{
  "score": <float entre 0.0 et 1.0>,
  "justification": "<1-3 phrases expliquant le score en français>",
  "elements_trouves": ["<élément 1 de l'exigence présent dans le passage>", ...],
  "elements_manquants": ["<élément 1 de l'exigence absent du passage>", ...],
  "confiance": <float entre 0.0 et 1.0, ta confiance dans l'analyse>
}}

Règles de scoring :
- 0.90-1.00 : Exigence couverte explicitement et complètement
- 0.65-0.89 : Exigence couverte partiellement ou implicitement
- 0.40-0.64 : Éléments évoqués mais couverture insuffisante
- 0.00-0.39 : Exigence non couverte dans ce passage"""


# ---------------------------------------------------------------------------
# Backend 1 : AnthropicLLM (Production)
# ---------------------------------------------------------------------------

class AnthropicLLM(LLMEngine):
    """
    Backend LLM basé sur l'API Claude (Anthropic).

    Modèle : claude-sonnet-4-20250514
    Pourquoi Sonnet et non Haiku ?
      - Haiku : plus rapide et moins cher, mais raisonnement juridique moins fiable
      - Sonnet : meilleur équilibre qualité/coût pour l'analyse réglementaire
      - Opus : inutilement puissant pour cette tâche structurée

    Prérequis :
        pip install anthropic
        export ANTHROPIC_API_KEY="sk-ant-..."
    """

    MODEL = "claude-sonnet-4-20250514"
    MAX_TOKENS = 1000

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Clé API Anthropic manquante. "
                "Définir ANTHROPIC_API_KEY ou passer api_key au constructeur."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    @property
    def nom_backend(self) -> str:
        return f"anthropic/{self.MODEL}"

    def analyser_conformite(
        self,
        texte_verification: str,
        passage_sfcr: str,
        id_exigence: str,
        niveau_obligation: str = "SHALL",
    ) -> AnalyseLLM:
        prompt = self._construire_prompt(
            texte_verification, passage_sfcr, niveau_obligation
        )
        try:
            t0 = time.time()
            response = self._get_client().messages.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            latence = time.time() - t0
            texte = response.content[0].text.strip()
            tokens = response.usage.input_tokens + response.usage.output_tokens

            resultat = self._parser_reponse_json(texte)
            resultat.tokens_utilises = tokens
            resultat.backend_utilise = self.nom_backend

            logger.debug(
                f"{id_exigence} | score={resultat.score:.2f} | "
                f"tokens={tokens} | latence={latence:.1f}s"
            )
            return resultat

        except Exception as e:
            logger.error(f"Erreur API Anthropic pour {id_exigence}: {e}")
            return self._reponse_erreur(str(e))

    def _parser_reponse_json(self, texte: str) -> AnalyseLLM:
        """Parse la réponse JSON du LLM avec tolérance aux erreurs."""
        try:
            # Extraire le JSON si entouré de ```
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texte, re.DOTALL)
            json_str = match.group(1) if match else texte
            data = json.loads(json_str)
            return AnalyseLLM(
                score=float(data.get("score", 0.0)),
                justification=str(data.get("justification", "")),
                elements_trouves=list(data.get("elements_trouves", [])),
                elements_manquants=list(data.get("elements_manquants", [])),
                confiance=float(data.get("confiance", 0.8)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"JSON LLM invalide : {e} | texte={texte[:100]}")
            return self._reponse_erreur(f"JSON invalide : {e}")

    def _reponse_erreur(self, message: str) -> AnalyseLLM:
        return AnalyseLLM(
            score=0.0,
            justification=f"Erreur d'analyse : {message}",
            elements_trouves=[],
            elements_manquants=["Analyse impossible"],
            confiance=0.0,
            backend_utilise=self.nom_backend,
        )


# ---------------------------------------------------------------------------
# Backend 2 : OllamaLLM (Production alternative — Mistral-7B local)
# ---------------------------------------------------------------------------

class OllamaLLM(LLMEngine):
    """
    Backend LLM basé sur Ollama (Mistral-7B en local).

    Pourquoi Mistral-7B ?
      - Meilleur open-source pour le raisonnement structuré en français
      - Licence Apache 2.0 → pas de restrictions commerciales
      - 7B paramètres → tourne sur 8 Go de RAM (CPU) ou 6 Go VRAM (GPU)
      - Instruction-tuned (v0.3) → suit les consignes JSON correctement

    Installation :
        curl -fsSL https://ollama.ai/install.sh | sh
        ollama pull mistral
        ollama serve  # dans un terminal séparé

    Usage :
        engine = OllamaLLM(model="mistral", base_url="http://ollama:11434")
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://ollama:11434",
        timeout: int = 120,
    ):
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    @property
    def nom_backend(self) -> str:
        return f"ollama/{self._model}"

    def analyser_conformite(
        self,
        texte_verification: str,
        passage_sfcr: str,
        id_exigence: str,
        niveau_obligation: str = "SHALL",
    ) -> AnalyseLLM:
        import urllib.request

        prompt = self._construire_prompt(
            texte_verification, passage_sfcr, niveau_obligation
        )
        payload = json.dumps({
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,     # Basse température → réponses déterministes
                "num_predict": 800,
            },
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self._base_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as r:
                data = json.loads(r.read())
                texte = data.get("response", "").strip()

            resultat = AnthropicLLM._parser_reponse_json(self, texte)
            resultat.backend_utilise = self.nom_backend
            return resultat

        except Exception as e:
            logger.error(f"Erreur Ollama pour {id_exigence}: {e}")
            return AnalyseLLM(
                score=0.0,
                justification=f"Erreur Ollama : {e}",
                elements_trouves=[],
                elements_manquants=["Analyse Ollama impossible"],
                confiance=0.0,
                backend_utilise=self.nom_backend,
            )


# ---------------------------------------------------------------------------
# Backend 3 : RuleBasedLLM (Développement offline)
# ---------------------------------------------------------------------------

class RuleBasedLLM(LLMEngine):
    """
    Backend LLM déterministe basé sur des règles lexicales.

    Conçu pour le développement offline — valide le pipeline complet
    sans appel LLM. Les scores sont calculés par analyse de fréquence
    de mots-clés réglementaires entre l'exigence et le passage.

    Algorithme :
      1. Extraction des mots significatifs de texte_verification
      2. Recherche de ces mots dans passage_sfcr (avec lemmatisation simple)
      3. Score = ratio mots trouvés / mots total (pondéré par position)
      4. Bonus si le passage est long et structuré (signe de complétude)
      5. Malus si le passage est générique (mots sans substance)

    Limites vs LLM réel :
      - Ne comprend pas la paraphrase sémantique
      - Sensible aux variations lexicales (SCR vs capital requis)
      - Pas de raisonnement sur la suffisance d'une réponse
      → Scores calibrés pour être conservateurs (légèrement sous-estimés)
    """

    # Mots vides à ignorer pour le matching
    STOP_WORDS = {
        "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou",
        "en", "dans", "sur", "par", "pour", "avec", "sans", "est", "sont",
        "il", "elle", "ils", "elles", "ce", "cet", "cette", "ces",
        "que", "qui", "quoi", "dont", "où", "si", "mais", "car", "donc",
        "or", "ni", "a", "au", "aux", "leur", "leurs", "se", "sa", "son",
        "rapport", "entreprise", "assurance", "réassurance",
    }

    # Termes réglementaires de haute valeur Solvabilité II
    TERMES_HAUTS_VALEUR = {
        "scr", "mcr", "sfcr", "orsa", "best estimate", "risk margin",
        "solvabilité ii", "eiopa", "provisions techniques", "fonds propres",
        "gouvernance", "fonctions clés", "audit interne", "actuariel",
        "conformité", "contrôle interne", "réassurance", "valorisation",
    }

    @property
    def nom_backend(self) -> str:
        return "rule-based (développement offline)"

    def analyser_conformite(
        self,
        texte_verification: str,
        passage_sfcr: str,
        id_exigence: str,
        niveau_obligation: str = "SHALL",
    ) -> AnalyseLLM:
        """
        Analyse lexicale de la couverture d'une exigence par un passage.
        """
        # Extraction des mots-clés de l'exigence
        mots_exigence = self._extraire_mots_cles(texte_verification)
        mots_passage = self._extraire_mots_cles(passage_sfcr)

        if not mots_exigence:
            return AnalyseLLM(
                score=0.0,
                justification="Exigence vide ou non analysable.",
                elements_trouves=[],
                elements_manquants=[],
                confiance=0.3,
                backend_utilise=self.nom_backend,
            )

        # Matching des mots-clés (avec racines communes)
        trouves, manquants = self._matcher_mots_cles(
            mots_exigence, mots_passage, passage_sfcr.lower()
        )

        # Score de base : ratio mots trouvés
        ratio_base = len(trouves) / len(mots_exigence)

        # Bonus pour termes réglementaires de haute valeur
        bonus_hv = self._bonus_termes_haute_valeur(
            texte_verification.lower(), passage_sfcr.lower()
        )

        # Bonus longueur du passage (>200 tokens → contenu substantiel)
        nb_mots_passage = len(passage_sfcr.split())
        bonus_longueur = min(0.05, nb_mots_passage / 4000)

        # Score final capped à 0.88 (le RuleBased ne peut pas donner 1.0)
        # car il ne comprend pas la sémantique profonde
        score_brut = ratio_base * 0.80 + bonus_hv * 0.15 + bonus_longueur
        score = round(min(0.88, score_brut), 4)

        # Confiance : inversement proportionnelle au nombre de mots manquants
        confiance = max(0.4, 1.0 - (len(manquants) / max(len(mots_exigence), 1)) * 0.5)

        # Justification
        justification = self._generer_justification(
            score, trouves, manquants, id_exigence, passage_sfcr
        )

        return AnalyseLLM(
            score=score,
            justification=justification,
            elements_trouves=[m.replace("_", " ") for m in trouves[:5]],
            elements_manquants=[m.replace("_", " ") for m in manquants[:5]],
            confiance=round(confiance, 3),
            tokens_utilises=0,
            backend_utilise=self.nom_backend,
        )

    def _extraire_mots_cles(self, texte: str) -> list[str]:
        """Extrait les mots significatifs d'un texte."""
        texte_lower = texte.lower()
        # Supprimer la ponctuation
        texte_clean = re.sub(r"[^\w\s\-]", " ", texte_lower)
        mots = texte_clean.split()
        # Filtrer stop words et mots très courts
        return [m for m in mots if m not in self.STOP_WORDS and len(m) >= 4]

    def _matcher_mots_cles(
        self,
        mots_exigence: list[str],
        mots_passage: list[str],
        passage_lower: str,
    ) -> tuple[list[str], list[str]]:
        """
        Match les mots-clés de l'exigence dans le passage.
        Utilise le début des mots (racine) pour tolérer les variations
        morphologiques françaises (gouvernance → gouverner, etc.)
        """
        trouves, manquants = [], []
        for mot in mots_exigence:
            # Match exact
            if mot in mots_passage:
                trouves.append(mot)
                continue
            # Match par racine (5 premiers caractères)
            racine = mot[:5] if len(mot) >= 5 else mot
            if any(m.startswith(racine) for m in mots_passage):
                trouves.append(mot)
                continue
            # Match partiel dans le texte complet (bigrammes, termes composés)
            if len(mot) >= 6 and mot[:6] in passage_lower:
                trouves.append(mot)
                continue
            manquants.append(mot)
        return trouves, manquants

    def _bonus_termes_haute_valeur(
        self, exigence_lower: str, passage_lower: str
    ) -> float:
        """Bonus si des termes réglementaires clés de l'exigence sont dans le passage."""
        bonus = 0.0
        for terme in self.TERMES_HAUTS_VALEUR:
            if terme in exigence_lower and terme in passage_lower:
                bonus += 0.15
        return min(1.0, bonus)

    def _generer_justification(
        self,
        score: float,
        trouves: list[str],
        manquants: list[str],
        id_exigence: str,
        passage: str,
    ) -> str:
        """Génère une justification textuelle basée sur le score."""
        nb_mots = len(passage.split())
        if score >= 0.75:
            return (
                f"Le passage traite explicitement des éléments clés de "
                f"l'exigence {id_exigence} "
                f"({len(trouves)} termes pertinents détectés sur {len(passage.split())} mots). "
                f"La couverture est considérée comme substantielle."
            )
        elif score >= 0.45:
            manq_str = ", ".join(manquants[:3]) if manquants else "aucun"
            return (
                f"Le passage aborde partiellement l'exigence {id_exigence}. "
                f"{len(trouves)} éléments correspondent mais certains aspects "
                f"semblent insuffisamment développés : {manq_str}."
            )
        else:
            return (
                f"Le passage ne semble pas couvrir l'exigence {id_exigence} "
                f"de façon suffisante ({len(trouves)} correspondances sur "
                f"{len(trouves)+len(manquants)} termes attendus). "
                f"Éléments manquants : {', '.join(manquants[:3]) or 'non détectés'}."
            )


#----------------
# BACKEND 4: GROQ 
#----------------
class GroqLLM(LLMEngine):
    """
    Backend LLM basé sur l'API Groq (LLaMA 3.3 70B).

    Modèle : llama-3.3-70b-versatile
    Pourquoi LLaMA 3.3 70B et non Mistral ?
      - LLaMA 3.3 70B : 70 milliards de paramètres, meilleur raisonnement
        juridique et structuré que Mistral 7B
      - Groq LPU : 500+ tokens/sec, latence très faible vs Ollama CPU
      - Free tier : 14 400 requêtes/jour, largement suffisant pour le projet

    Prérequis :
        pip install groq
        export GROQ_API_KEY="gsk_..."
    """

    MODEL = "llama-3.3-70b-versatile"
    MAX_TOKENS = 300

    def __init__(self, api_key: Optional[str] = None):
        import os
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Clé API Groq manquante. "
                "Définir GROQ_API_KEY ou passer api_key au constructeur. "
                "Clé gratuite sur : https://console.groq.com"
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    @property
    def nom_backend(self) -> str:
        return f"groq/{self.MODEL}"

    def analyser_conformite(
        self,
        texte_verification: str,
        passage_sfcr: str,
        id_exigence: str,
        niveau_obligation: str = "SHALL",
    ) -> AnalyseLLM:
        prompt = self._construire_prompt(
            texte_verification, passage_sfcr, niveau_obligation
        )
        try:
            t0 = time.time()
            response = self._get_client().chat.completions.create(
                model=self.MODEL,
                max_tokens=self.MAX_TOKENS,
                temperature=0.1,  # Faible température = réponses déterministes
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en conformité réglementaire "
                                   "Solvabilité II. Réponds UNIQUEMENT en JSON valide."
                    },
                    {"role": "user", "content": prompt}
                ],
            )
            latence = time.time() - t0

            # Différence avec Anthropic : response.choices[0].message.content
            texte = response.choices[0].message.content.strip()

            # Groq expose les tokens via usage (même structure qu'OpenAI)
            tokens = response.usage.prompt_tokens + response.usage.completion_tokens

            resultat = self._parser_reponse_json(texte)
            resultat.tokens_utilises = tokens
            resultat.backend_utilise = self.nom_backend

            logger.debug(
                f"{id_exigence} | score={resultat.score:.2f} | "
                f"tokens={tokens} | latence={latence:.1f}s"
            )
            return resultat

        except Exception as e:
            logger.error(f"Erreur API Groq pour {id_exigence}: {e}")
            return self._reponse_erreur(str(e))

    def _parser_reponse_json(self, texte: str) -> AnalyseLLM:
        """Parse la réponse JSON du LLM avec tolérance aux erreurs."""
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", texte, re.DOTALL)
            json_str = match.group(1) if match else texte
            data = json.loads(json_str)
            return AnalyseLLM(
                score=float(data.get("score", 0.0)),
                justification=str(data.get("justification", "")),
                elements_trouves=list(data.get("elements_trouves", [])),
                elements_manquants=list(data.get("elements_manquants", [])),
                confiance=float(data.get("confiance", 0.8)),
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"JSON LLM invalide : {e} | texte={texte[:100]}")
            # Fallback : extraire le score seul par regex
            score_match = re.search(r'"score"\s*:\s*([0-9.]+)', texte)
            if score_match:
                return AnalyseLLM(
                    score=float(score_match.group(1)),
                    justification="Parsing partiel — JSON malformé",
                    elements_trouves=[],
                    elements_manquants=[],
                    confiance=0.5,
                )
            return self._reponse_erreur(f"JSON invalide : {e}")

    def _reponse_erreur(self, message: str) -> AnalyseLLM:
        return AnalyseLLM(
            score=0.0,
            justification=f"Erreur d'analyse : {message}",
            elements_trouves=[],
            elements_manquants=["Analyse impossible"],
            confiance=0.0,
            backend_utilise=self.nom_backend,
        )

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def creer_llm_engine(
    mode: str = "auto",
    api_key: Optional[str] = None,
    ollama_url: str = "http://ollama:11434",
    ollama_model: str = "mistral",
) -> LLMEngine:
    """
    Factory pour créer le bon moteur LLM selon l'environnement.

    Args:
        mode       : "anthropic"   → API Claude (nécessite ANTHROPIC_API_KEY)
                     "ollama"      → Mistral local (nécessite Ollama installé)
                     "local"       → RuleBasedLLM (offline, développement)
                     "auto"        → Essaie Anthropic, puis Ollama, puis local
        api_key    : Clé API Anthropic (sinon depuis ANTHROPIC_API_KEY)
        ollama_url : URL du serveur Ollama
        ollama_model: Modèle Ollama à utiliser

    Returns:
        LLMEngine prêt à l'emploi

    Usage recommandé :
        # Développement :
        engine = creer_llm_engine("local")

        # Production avec API :
        engine = creer_llm_engine("anthropic", api_key="sk-ant-...")

        # Production locale :
        engine = creer_llm_engine("ollama")
    """
    import os

    if mode == "groq":
        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if not key:
            raise ValueError("GROQ_API_KEY non défini")

        logger.info("LLM : GroqLLM")
        return GroqLLM(api_key=key)

    if mode == "ollama":
        logger.info(f"LLM : OllamaLLM ({ollama_model})")
        return OllamaLLM(model=ollama_model, base_url=ollama_url)

    if mode == "local":
        logger.info("LLM : RuleBasedLLM (développement offline)")
        return RuleBasedLLM()
    
    if mode == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY non défini")
        logger.info("LLM : AnthropicLLM (Claude Sonnet)")
        return AnthropicLLM(api_key=key)

    # Mode "auto" : cascade de priorité
    # 1. Anthropic si clé disponible
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        logger.info("Mode auto → AnthropicLLM")
        return AnthropicLLM(api_key=key)

    # 2. Ollama si serveur accessible
    try:
        import urllib.request
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=2)
        logger.info(f"Mode auto → OllamaLLM ({ollama_model})")
        return OllamaLLM(model=ollama_model, base_url=ollama_url)
    except Exception:
        pass

    # 3. Fallback local
    logger.warning(
        "Mode auto → RuleBasedLLM (offline). "
        "Définir ANTHROPIC_API_KEY ou lancer Ollama pour le mode production."
    )
    return RuleBasedLLM()
