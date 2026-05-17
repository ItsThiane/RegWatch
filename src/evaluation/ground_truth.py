"""
ground_truth.py — Jeu de vérité terrain pour l'évaluation RAG
==============================================================
Définit les paires (exigence → passages SFCR attendus) qui servent
de référence pour mesurer la qualité du retrieval.

Construction du ground truth :
  Chaque entrée associe un id_exigence à une liste de passages textuels
  qui constituent une réponse correcte à la question de vérification.
  Ces passages sont construits à partir des textes réglementaires réels
  et de SFCR publics (AXA, Groupama, Covéa...).

Niveaux de pertinence (inspirés de TREC/BEIR) :
  2 — Très pertinent : couvre l'exigence explicitement et complètement
  1 — Pertinent      : couvre partiellement ou implicitement
  0 — Non pertinent  : hors sujet

Métriques calculées sur ce ground truth :
  Precision@K : proportion de passages pertinents dans le top-K
  Recall@K    : proportion d'exigences pour lesquelles ≥1 passage pertinent
               figure dans le top-K
  MRR         : Mean Reciprocal Rank — position moyenne du 1er bon résultat
  NDCG@K      : Normalized Discounted Cumulative Gain — qualité du classement
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class EntreeGroundTruth:
    """
    Une entrée du jeu de vérité terrain.

    id_exigence      : Identifiant de l'exigence testée
    passages_niveau2 : Passages très pertinents (réponse idéale)
    passages_niveau1 : Passages partiellement pertinents
    passages_niveau0 : Passages non pertinents (distracteurs)
    notes            : Commentaires sur la difficulté / particularités
    """
    id_exigence: str
    passages_niveau2: list[str]
    passages_niveau1: list[str] = field(default_factory=list)
    passages_niveau0: list[str] = field(default_factory=list)
    notes: str = ""

    def tous_passages(self) -> list[tuple[str, int]]:
        """Retourne tous les passages avec leur niveau de pertinence."""
        result = []
        for p in self.passages_niveau2:
            result.append((p, 2))
        for p in self.passages_niveau1:
            result.append((p, 1))
        for p in self.passages_niveau0:
            result.append((p, 0))
        return result

    def passages_pertinents(self) -> list[str]:
        """Retourne tous les passages pertinents (niveau >= 1)."""
        return self.passages_niveau2 + self.passages_niveau1


# ---------------------------------------------------------------------------
# Jeu de vérité terrain — 20 exigences représentatives
# ---------------------------------------------------------------------------

GROUND_TRUTH: list[EntreeGroundTruth] = [

    # ── T04 : Gouvernance ────────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art292-§2",
        passages_niveau2=[
            "L'entreprise a mis en place les quatre fonctions clés requises par "
            "Solvabilité II : la fonction de gestion des risques, la fonction de "
            "vérification de la conformité, la fonction d'audit interne et la "
            "fonction actuarielle. Chacune est dotée de ressources suffisantes "
            "et opère de manière indépendante.",
            "Les quatre fonctions clés Solvabilité II — gestion des risques, "
            "conformité, audit interne et actuariat — sont opérationnelles et "
            "indépendantes des fonctions opérationnelles qu'elles supervisent. "
            "Leurs responsables rendent compte directement au conseil d'administration.",
        ],
        passages_niveau1=[
            "La direction des risques, la conformité et l'audit interne constituent "
            "les piliers du dispositif de contrôle de l'entreprise. La fonction "
            "actuarielle complète ce dispositif en supervisant les provisions.",
            "Notre organisation comprend quatre fonctions de contrôle indépendantes "
            "conformément aux exigences de la directive Solvabilité II.",
        ],
        passages_niveau0=[
            "Le chiffre d'affaires consolidé atteint 12,4 milliards d'euros en 2023, "
            "en progression de 4,2% par rapport à l'exercice précédent.",
            "Le ratio de couverture du SCR s'établit à 189% au 31 décembre 2023.",
        ],
        notes="Exigence centrale — les quatre fonctions clés doivent toutes être citées",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art292-§1b",
        passages_niveau2=[
            "La politique de rémunération de l'entreprise distingue une composante "
            "fixe et une composante variable pour les dirigeants effectifs et les "
            "détenteurs de fonctions clés. La part variable est plafonnée à 100% "
            "de la rémunération fixe et est conditionnée à l'atteinte d'objectifs "
            "quantitatifs et qualitatifs définis annuellement.",
            "Notre politique de rémunération est structurée en trois catégories : "
            "les dirigeants effectifs, les responsables de fonctions clés et les "
            "autres collaborateurs. Pour chaque catégorie, la rémunération fixe "
            "et la rémunération variable sont clairement définies et documentées.",
        ],
        passages_niveau1=[
            "La rémunération des dirigeants comprend une part fixe et une part "
            "variable liée aux performances individuelles et collectives.",
            "Le comité des rémunérations supervise la politique de rémunération "
            "applicable aux membres de la direction et aux fonctions clés.",
        ],
        passages_niveau0=[
            "Les provisions techniques sont calculées conformément aux guidelines "
            "EIOPA et aux hypothèses économiques en vigueur.",
        ],
        notes="Distinction fixe/variable ET ventilation par catégorie requises",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art293-§1",
        passages_niveau2=[
            "L'entreprise applique une politique de compétence et d'honorabilité "
            "(fit & proper) à l'ensemble des personnes qui dirigent effectivement "
            "l'entreprise et aux détenteurs de fonctions clés. Cette évaluation "
            "est réalisée avant la prise de fonction et renouvelée annuellement. "
            "Elle couvre les qualifications professionnelles, l'expérience, "
            "la réputation et l'absence de conflits d'intérêts.",
        ],
        passages_niveau1=[
            "Les membres du conseil d'administration et les responsables de fonctions "
            "clés font l'objet d'une évaluation de leur compétence et honorabilité "
            "conformément aux exigences Solvabilité II.",
            "Un processus d'évaluation fit & proper est en place pour tous les "
            "dirigeants et responsables de fonctions de contrôle.",
        ],
        passages_niveau0=[
            "Le taux de sinistralité s'établit à 68,3% pour l'exercice 2023.",
        ],
        notes="Fit & proper doit couvrir dirigeants ET fonctions clés",
    ),

    # ── T05 : Gestion des risques / ORSA ────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P2-DIR2009-Art45-§1",
        passages_niveau2=[
            "L'entreprise a conduit son évaluation interne des risques et de la "
            "solvabilité (ORSA) en 2023. Cette évaluation a permis de déterminer "
            "les besoins globaux de solvabilité en tenant compte du profil de risque "
            "spécifique de l'entreprise, de ses limites de tolérance au risque et "
            "de sa stratégie commerciale. Le rapport ORSA a été approuvé par le "
            "conseil d'administration en juin 2023.",
            "L'ORSA 2023 conclut que les besoins globaux de solvabilité de "
            "l'entreprise sont couverts avec un ratio de 165% dans le scénario "
            "central et restent positifs dans l'ensemble des scénarios adverses "
            "testés, y compris le scénario de stress combiné.",
        ],
        passages_niveau1=[
            "L'évaluation interne des risques et de la solvabilité a été réalisée "
            "conformément à l'article 45 de la directive Solvabilité II.",
            "Notre ORSA annuel évalue la suffisance de notre capital par rapport "
            "à notre profil de risque spécifique et nos objectifs stratégiques.",
        ],
        passages_niveau0=[
            "L'entreprise a lancé une nouvelle offre d'assurance-vie en unités de "
            "compte en mars 2023, avec un objectif de collecte de 500 M€.",
        ],
        notes="ORSA = évaluation interne risques ET solvabilité, approuvé par CA",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P2-REG2015-Art295-§4",
        passages_niveau2=[
            "L'entreprise a réalisé en 2023 plusieurs stress tests réglementaires "
            "et internes. Le choc taux (-100 pb) réduit le ratio SCR de 189% à 152%. "
            "Le choc actions (-30%) le ramène à 163%. Le scénario catastrophe "
            "naturelle (ouragan européen) aboutit à un ratio de 147%. Dans tous "
            "les scénarios, le ratio reste supérieur à 100%.",
            "Les analyses de sensibilité réalisées dans le cadre de l'ORSA "
            "montrent que le ratio de couverture SCR demeure supérieur à 130% "
            "dans l'ensemble des scénarios adverses individuels testés "
            "(choc taux, choc actions, choc crédit, choc mortalité).",
        ],
        passages_niveau1=[
            "Des stress tests ont été conduits sur les principaux facteurs de risque. "
            "Les résultats démontrent la résilience du bilan de l'entreprise.",
            "L'entreprise effectue des analyses de sensibilité trimestrielles "
            "sur les principaux risques auxquels elle est exposée.",
        ],
        passages_niveau0=[
            "La fonction actuarielle a émis un avis favorable sur la politique "
            "de souscription 2023.",
        ],
        notes="Résultats chiffrés des stress tests requis pour niveau 2",
    ),

    # ── T06 : Contrôle interne / Audit ──────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P2-DIR2009-Art46-§1",
        passages_niveau2=[
            "Le système de contrôle interne de l'entreprise comprend l'ensemble "
            "des procédures administratives et comptables, un cadre de contrôle "
            "interne structuré sur trois niveaux de défense, et des dispositifs "
            "de reporting à tous les niveaux de l'organisation. La fonction de "
            "vérification de la conformité assure le respect permanent des "
            "obligations réglementaires.",
            "Notre dispositif de contrôle interne repose sur des procédures "
            "documentées couvrant l'ensemble des processus opérationnels. "
            "Il inclut des contrôles de premier niveau intégrés aux opérations, "
            "des contrôles de deuxième niveau assurés par les fonctions risques "
            "et conformité, et un audit interne indépendant en troisième niveau.",
        ],
        passages_niveau1=[
            "L'entreprise dispose d'un solide dispositif de contrôle interne "
            "validé annuellement par la direction générale et l'audit interne.",
            "Les procédures de contrôle interne font l'objet d'une mise à jour "
            "annuelle et sont communiquées à l'ensemble des collaborateurs.",
        ],
        passages_niveau0=[
            "Le Best Estimate des provisions vie s'établit à 18,4 Md€.",
        ],
        notes="Trois éléments requis : procédures, cadre de contrôle, reporting",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P2-DIR2009-Art47-§1",
        passages_niveau2=[
            "La fonction d'audit interne est organisée de manière indépendante "
            "des fonctions opérationnelles. Elle évalue l'adéquation et l'efficacité "
            "du système de contrôle interne et du système de gouvernance. "
            "Son plan d'audit annuel est approuvé par le comité d'audit du conseil "
            "d'administration. Ses conclusions et recommandations sont communiquées "
            "directement au conseil d'administration.",
        ],
        passages_niveau1=[
            "L'audit interne, rattaché directement au conseil d'administration, "
            "conduit des missions sur l'ensemble des processus de l'entreprise "
            "selon un plan annuel fondé sur les risques.",
            "La fonction d'audit interne évalue régulièrement l'efficacité du "
            "dispositif de contrôle interne et formule des recommandations "
            "d'amélioration suivies de plans d'action.",
        ],
        passages_niveau0=[
            "Les primes brutes acquises s'élèvent à 8,2 Md€ pour l'exercice 2023.",
        ],
        notes="Indépendance et rattachement au CA sont des éléments clés",
    ),

    # ── T07 : Fonction actuarielle ───────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P2-DIR2009-Art48-§1a",
        passages_niveau2=[
            "La fonction actuarielle coordonne le calcul des provisions techniques "
            "de l'entreprise et en valide les méthodologies, modèles et hypothèses. "
            "Elle s'assure du caractère approprié des méthodes actuarielles utilisées "
            "au regard des exigences de Solvabilité II et des guidelines EIOPA. "
            "Elle rend compte annuellement au conseil d'administration de la "
            "fiabilité et de l'adéquation des provisions techniques.",
        ],
        passages_niveau1=[
            "La fonction actuarielle supervise et valide les calculs de provisions "
            "techniques réalisés par les équipes techniques de l'entreprise.",
            "Les hypothèses actuarielles sont revues et validées annuellement "
            "par la fonction actuarielle conformément aux exigences réglementaires.",
        ],
        passages_niveau0=[
            "Le ratio de solvabilité de 189% reflète la solidité financière "
            "de l'entreprise et sa capacité à absorber des chocs adverses.",
        ],
        notes="Distinction coordination/validation vs calcul lui-même",
    ),

    # ── T08 : Structure SFCR ─────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art298-§1",
        passages_niveau2=[
            "Le capital de solvabilité requis (SCR) s'élève à 312 millions d'euros "
            "au 31 décembre 2023, calculé selon la formule standard. Les fonds "
            "propres éligibles atteignent 589 millions d'euros, soit un ratio de "
            "couverture du SCR de 189%. Le minimum de capital requis (MCR) s'établit "
            "à 78 millions d'euros et est couvert à hauteur de 755%.",
            "Section E — Gestion du capital : notre politique de gestion du capital "
            "vise à maintenir le ratio de couverture SCR entre 150% et 200%. "
            "Au 31/12/2023, le SCR est de 312 M€ (formule standard), les fonds "
            "propres éligibles de 589 M€ (ratio 189%) et le MCR de 78 M€ (ratio 755%).",
        ],
        passages_niveau1=[
            "Notre ratio de couverture SCR de 189% témoigne de la solidité "
            "financière de l'entreprise et dépasse largement l'objectif interne.",
            "Les fonds propres sont constitués à 95% d'éléments Tier 1 de la "
            "meilleure qualité, assurant une couverture robuste du SCR et du MCR.",
        ],
        passages_niveau0=[
            "L'entreprise exerce ses activités dans 12 pays européens et emploie "
            "plus de 8 000 collaborateurs.",
        ],
        notes="SCR + MCR + fonds propres + ratio couverture = les 4 éléments clés",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art291-§1b",
        passages_niveau2=[
            "Les résultats de souscription 2023 se ventilent comme suit par ligne "
            "d'activité : assurance-vie individuelle (primes 4,2 Md€, ratio combiné "
            "98,2%), assurance santé (primes 1,8 Md€, ratio 94,5%), assurance IARD "
            "(primes 2,3 Md€, ratio 96,8%). Ces résultats sont en amélioration "
            "par rapport à l'exercice précédent sur toutes les lignes d'activité.",
        ],
        passages_niveau1=[
            "Les primes brutes émises par ligne d'activité Solvabilité II montrent "
            "une progression de 4,2% sur l'ensemble du portefeuille.",
            "La ventilation du résultat technique par ligne d'activité fait "
            "apparaître une performance homogène sur l'ensemble des branches.",
        ],
        passages_niveau0=[
            "Le Best Estimate des provisions vie inclut les engagements épargne, "
            "prévoyance et retraite de l'entreprise.",
        ],
        notes="Ventilation par LoB SII ET résultats chiffrés nécessaires",
    ),

    # ── T01 : Valorisation ───────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P1-REG2015-Art261-§1",
        passages_niveau2=[
            "Les provisions techniques au sens de Solvabilité II s'élèvent à "
            "21,4 milliards d'euros au 31 décembre 2023, décomposées en Best "
            "Estimate de 20,8 Md€ et Risk Margin de 0,6 Md€. Le Best Estimate "
            "est calculé comme la valeur actuelle probable des flux de trésorerie "
            "futurs en utilisant la courbe des taux sans risque publiée par l'EIOPA. "
            "Les principales hypothèses actuarielles sont : taux de mortalité (tables "
            "réglementaires), taux de rachat (modèle comportemental), frais de "
            "gestion (projection sur base historique).",
        ],
        passages_niveau1=[
            "Le Best Estimate des provisions vie est calculé selon une approche "
            "stochastique pour les contrats avec options et garanties financières.",
            "La Risk Margin est calculée par la méthode du coût du capital avec "
            "un taux de 6% conformément aux guidelines EIOPA.",
        ],
        passages_niveau0=[
            "L'entreprise a distribué un dividende de 2,50€ par action au titre "
            "de l'exercice 2022.",
        ],
        notes="BE + RM + méthodes + hypothèses principales toutes requises",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P1-REG2015-Art260-§1b",
        passages_niveau2=[
            "Le bilan Solvabilité II diffère du bilan comptable sur plusieurs "
            "points majeurs. Les actifs sont valorisés à la juste valeur (mark-to-"
            "market) contre la valeur historique en comptabilité locale. Les "
            "provisions techniques passent de 22,1 Md€ (normes françaises) à "
            "21,4 Md€ (Solvabilité II), soit un écart de -700 M€ principalement "
            "dû à l'actualisation des flux futurs et à la suppression des marges "
            "de prudence implicites.",
        ],
        passages_niveau1=[
            "Un tableau de passage entre le bilan comptable et le bilan Solvabilité II "
            "est présenté en annexe. Les principaux retraitements portent sur la "
            "valorisation des actifs et le calcul des provisions techniques.",
            "Les différences entre valorisation Solvabilité II et comptabilité "
            "locale sont expliquées dans la section D du présent rapport.",
        ],
        passages_niveau0=[
            "Notre portefeuille d'investissement est principalement composé "
            "d'obligations (72%), d'actions (15%) et d'immobilier (8%).",
        ],
        notes="Différences quantitatives ET qualitatives entre SII et comptable",
    ),

    # ── T09 : QRT ────────────────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art304-§1",
        passages_niveau2=[
            "Les tableaux quantitatifs réglementaires (QRT) suivants sont publiés "
            "en annexe du présent rapport : S.02.01 (Bilan), S.05.01 (Primes, "
            "sinistres et dépenses par ligne d'activité), S.05.02 (Primes, "
            "sinistres et dépenses par pays), S.17.01 (Provisions techniques "
            "non-vie), S.25.01 (Capital de solvabilité requis — formule standard), "
            "S.23.01 (Fonds propres).",
        ],
        passages_niveau1=[
            "Les QRT obligatoires sont publiés conformément au règlement délégué "
            "2015/35 et aux instructions de l'EIOPA.",
            "Les tableaux quantitatifs S.02, S.05, S.23 et S.25 sont disponibles "
            "en annexe du présent rapport SFCR.",
        ],
        passages_niveau0=[
            "La fonction de vérification de la conformité a conduit 12 missions "
            "au cours de l'exercice 2023.",
        ],
        notes="Liste explicite des QRT publiés avec leurs codes S.xx.xx",
    ),

    # ── T10 : Transparence ───────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art289-§3",
        passages_niveau2=[
            "Au cours de l'exercice 2023, les changements significatifs suivants "
            "sont intervenus : (1) acquisition de la société XYZ Assurances en "
            "janvier 2023, intégrée dans le périmètre SFCR à compter du "
            "1er avril 2023 ; (2) révision de la politique d'investissement avec "
            "une réduction de l'exposition actions de 18% à 15% ; (3) changement "
            "de méthode de calcul du Best Estimate vie suite aux nouvelles "
            "guidelines EIOPA publiées en juin 2023.",
        ],
        passages_niveau1=[
            "L'exercice 2023 a été marqué par plusieurs évolutions significatives "
            "dans la gouvernance et le profil de risque de l'entreprise.",
            "Les principaux changements intervenus en 2023 par rapport à 2022 "
            "sont décrits dans chaque section du présent rapport.",
        ],
        passages_niveau0=[
            "Le résultat net part du groupe s'élève à 842 millions d'euros.",
        ],
        notes="Changements dans CHAQUE section doivent être signalés",
    ),

    # ── T11 : Délais ─────────────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-DIR2009-Art51-§1",
        passages_niveau2=[
            "Le présent rapport sur la solvabilité et la situation financière "
            "a été approuvé par le conseil d'administration du 15 mars 2024 "
            "et publié le 22 mars 2024, soit dans un délai de 11 semaines après "
            "la clôture de l'exercice au 31 décembre 2023, conformément à "
            "l'exigence réglementaire de 14 semaines.",
        ],
        passages_niveau1=[
            "Ce rapport SFCR est publié conformément à l'article 51 de la "
            "directive Solvabilité II dans le délai réglementaire applicable.",
            "Date de clôture : 31 décembre 2023. Date d'approbation : 15 mars 2024. "
            "Date de publication : 22 mars 2024.",
        ],
        passages_niveau0=[
            "Le résultat opérationnel s'établit à 1,2 milliard d'euros en 2023.",
        ],
        notes="Date de publication explicite permettant de vérifier le délai de 14 semaines",
    ),

    # ── T02 : Capital ────────────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art298-§2",
        passages_niveau2=[
            "Le capital de solvabilité requis est calculé selon la formule standard "
            "définie par le règlement délégué 2015/35. La décomposition du SCR "
            "par module de risque est la suivante : risque de marché 185 M€ (59%), "
            "risque de souscription vie 98 M€ (31%), risque de contrepartie 22 M€ "
            "(7%), risque opérationnel 15 M€ (5%), ajustement pour capacité "
            "d'absorption des pertes -8 M€. SCR net total : 312 M€.",
        ],
        passages_niveau1=[
            "L'entreprise utilise la formule standard pour le calcul du SCR. "
            "Le module de risque de marché constitue le principal contributeur.",
            "La formule standard est appliquée sans mesures transitoires. "
            "Aucun modèle interne n'est utilisé.",
        ],
        passages_niveau0=[
            "Notre réseau de distribution comprend 3 200 agents généraux "
            "et 850 courtiers partenaires.",
        ],
        notes="Formule standard vs modèle interne + décomposition par module",
    ),

    # ── T03 : Fonds propres ──────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P3-REG2015-Art304-§3",
        passages_niveau2=[
            "Les fonds propres éligibles de l'entreprise s'élèvent à 589 millions "
            "d'euros et se décomposent comme suit : Tier 1 non restreint 558 M€ "
            "(95%), Tier 1 restreint 0 M€, Tier 2 31 M€ (5%), Tier 3 0 M€. "
            "Le tableau S.23.01 détaillant la structure des fonds propres est "
            "présenté en annexe du présent rapport.",
        ],
        passages_niveau1=[
            "La quasi-totalité de nos fonds propres est constituée d'éléments "
            "Tier 1 de la meilleure qualité (capital et réserves).",
            "Le tableau quantitatif S.23.01 relatif aux fonds propres est "
            "publié conformément aux exigences réglementaires.",
        ],
        passages_niveau0=[
            "L'entreprise a procédé au remboursement d'une émission obligataire "
            "subordonnée de 300 M€ en septembre 2023.",
        ],
        notes="Décomposition Tier 1/2/3 + référence au QRT S.23.01",
    ),

    # ── T05 : Risques ────────────────────────────────────────────────────

    EntreeGroundTruth(
        id_exigence="SII-P2-REG2015-Art295-§2a",
        passages_niveau2=[
            "Le système de gestion des risques de l'entreprise s'articule autour "
            "d'un Risk Appetite Framework approuvé par le conseil d'administration. "
            "Ce cadre définit les limites de tolérance au risque par catégorie "
            "(ratio SCR minimum 130%, VaR crédit 99,5%, perte maximale actions "
            "400 M€). Les risques sont identifiés, mesurés et suivis trimestriellement "
            "via un tableau de bord risques présenté au comité des risques.",
        ],
        passages_niveau1=[
            "La politique de gestion des risques définit les processus "
            "d'identification, d'évaluation et de surveillance des risques "
            "auxquels l'entreprise est exposée.",
            "Le système de gestion des risques est intégré aux processus "
            "décisionnels de l'entreprise et fait l'objet d'un reporting "
            "trimestriel à la direction générale.",
        ],
        passages_niveau0=[
            "L'entreprise a lancé une nouvelle offre d'épargne retraite "
            "en septembre 2023.",
        ],
        notes="Risk Appetite Framework + processus + intégration décisionnelle",
    ),

    EntreeGroundTruth(
        id_exigence="SII-P2-REG2015-Art295-§3",
        passages_niveau2=[
            "Les principales concentrations de risque identifiées sont : (1) risque "
            "de taux — sensibilité du ratio SCR de -15 points pour -100 pb sur les "
            "taux longs ; (2) risque de contrepartie réassurance — 68% des cessions "
            "sont concentrées sur 3 réassureurs (tous notés AA ou mieux) ; "
            "(3) concentration sectorielle — exposition actions limitée à 20% "
            "sur le secteur financier.",
        ],
        passages_niveau1=[
            "L'entreprise surveille ses concentrations de risque, notamment "
            "la concentration sur ses principaux réassureurs et ses expositions "
            "par secteur et par zone géographique.",
            "Les limites de concentration sont définies dans la politique "
            "d'investissement et font l'objet d'un suivi mensuel.",
        ],
        passages_niveau0=[
            "Notre portefeuille immobilier est évalué à 2,1 Md€.",
        ],
        notes="Concentration par type : crédit, géographique, sectorielle, réassurance",
    ),

]


def get_ground_truth_par_theme(theme_id: str) -> list[EntreeGroundTruth]:
    """Retourne les entrées ground truth pour un thème donné."""
    from src.ingestion.referentiel_loader import ReferentielLoader
    loader = ReferentielLoader()
    loader.charger(strict=False)

    ids_theme = {
        e.id_exigence
        for e in loader.get_exigences_par_theme(theme_id)
    }
    return [gt for gt in GROUND_TRUTH if gt.id_exigence in ids_theme]


def get_tous_ids_exigences() -> list[str]:
    """Retourne les id_exigence du ground truth."""
    return [gt.id_exigence for gt in GROUND_TRUTH]
