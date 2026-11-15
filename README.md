# 🚖 NYC Taxi Trip Duration — Kaggle Challenge (Guide Complet & Solution)

Ce projet propose une solution complète, performante et reproductible pour la compétition Kaggle **[New York City Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration)**.

Le projet est conçu selon une approche **100% Notebook** : tout le code (analyse exploratoire, nettoyage des données, ingénierie des caractéristiques, target encoding cross-validé, modélisation 5-fold multi-modèles, optimisation du blend et génération de soumission) est auto-contenu et pré-exécuté dans un **unique notebook Jupyter** à la racine :
👉 **[`NYC_Taxi_Trip_Duration_Master.ipynb`](./NYC_Taxi_Trip_Duration_Master.ipynb)**

---

## 📑 Sommaire
1. [Arborescence du Projet](#1-arborescence-du-projet)
2. [Objectif & Données du Défi](#2-objectif--données-du-défi)
3. [La Métrique d'Évaluation : Le RMSLE](#3-la-métrique-dévaluation--le-rmsle)
4. [Détail de Chaque Étape du Notebook : Ce qu'on fait & Pourquoi on le fait](#4-détail-de-chaque-étape-du-notebook--ce-quon-fait--pourquoi-on-le-fait)
   - [Étape 1 : Configuration de l'Environnement & Constantes Métier](#étape-1--configuration-de-lenvironnement--constantes-métier)
   - [Étape 2 : Chargement & Diagnostic des Données](#étape-2--chargement--diagnostic-des-données)
   - [Étape 3 : Analyse Exploratoire des Données (EDA) & Intuitions Physiques](#étape-3--analyse-exploratoire-des-données-eda--intuitions-physiques)
   - [Étape 4 : Nettoyage des Données & Filtrage des Valeurs Aberrantes](#étape-4--nettoyage-des-données--filtrage-des-valeurs-aberrantes)
   - [Étape 5 : Ingénierie des Caractéristiques Expertes (Feature Engineering)](#étape-5--ingénierie-des-caractéristiques-expertes-feature-engineering)
   - [Étape 6 : Target Encoding Cross-Validé sans Fuite (Zero Data Leakage)](#étape-6--target-encoding-cross-validé-sans-fuite-zero-data-leakage)
   - [Étape 7 : Modélisation 5-Fold Multi-Modèles (LightGBM, XGBoost, CatBoost)](#étape-7--modélisation-5-fold-multi-modèles-lightgbm-xgboost-catboost)
   - [Étape 8 : Analyse d'Importance des Variables & Distribution des Résidus](#étape-8--analyse-dimportance-des-variables--distribution-des-résidus)
   - [Étape 9 : Optimisation du Blend Multi-Modèles](#étape-9--optimisation-du-blend-multi-modèles)
   - [Étape 10 : Inférence & Vérification Stricte de la Soumission Kaggle](#étape-10--inférence--vérification-stricte-de-la-soumission-kaggle)
5. [Tableau Récapitulatif des Performances](#5-tableau-récapitulatif-des-performances)
6. [Comment Utiliser et Re-exécuter le Projet](#6-comment-utiliser-et-re-exécuter-le-projet)

---

## 1. Arborescence du Projet

Le projet ne contient aucun script externe ni sous-dossier superflu. Tous les fichiers nécessaires sont présents directement à la racine :

```text
New-York-City-Taxi-Trip-Duration/
│
├── NYC_Taxi_Trip_Duration_Master.ipynb   # 🌟 NOTEBOOK MAÎTRE UNIQUE (26 cellules, pré-exécuté avec tous les graphiques)
│
├── submission.csv                        # Fichier final officiel prêt pour soumission Kaggle (625 134 lignes)
│
├── train.csv                             # Données d'entraînement brutes fournies par Kaggle (1 458 644 lignes)
├── test.csv                              # Données de test brutes fournies par Kaggle (625 134 lignes)
├── sample_submission.csv                 # Format d'exemple officiel fourni par Kaggle
│
└── README.md                             # Le présent guide explicatif détaillé
```

---

## 2. Objectif & Données du Défi

L'objectif est de prédire la durée totale de chaque trajet de taxi new-yorkais (`trip_duration`, mesurée en secondes).

### Description des colonnes :
- `id` : Identifiant unique de chaque course.
- `vendor_id` : Code désignant la société de taxi (1 ou 2).
- `pickup_datetime` : Date et heure d'embarquement du passager.
- `dropoff_datetime` : Date et heure de débarquement (disponible uniquement dans `train.csv`).
- `passenger_count` : Nombre de passagers déclarés par le chauffeur (0 à 9).
- `pickup_longitude`, `pickup_latitude` : Coordonnées GPS de départ.
- `dropoff_longitude`, `dropoff_latitude` : Coordonnées GPS d'arrivée.
- `store_and_fwd_flag` : Indique si les données ont été stockées en mémoire avant transmission (Y/N).
- `trip_duration` : Variable cible à prédire (en secondes).

---

## 3. La Métrique d'Évaluation : Le RMSLE

La compétition est évaluée avec le **Root Mean Squared Logarithmic Error (RMSLE)** :

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left(\ln(p_i + 1) - \ln(a_i + 1)\right)^2}$$

Où :
- $p_i$ est la durée prédite pour la course $i$.
- $a_i$ est la durée réelle constatée pour la course $i$.
- $\ln$ est le logarithme népérien.

### 🎯 Pourquoi entraîne-t-on le modèle sur $\log1\text{p}(\text{trip\_duration})$ ?
1. **Pénalité relative et proportionnelle** : Dans le calcul du RMSLE, une erreur de 2 minutes sur un trajet de 5 minutes est pénalisée avec la même sévérité qu'une erreur de 20 minutes sur un trajet de 50 minutes. À l'inverse, l'erreur quadratique standard (RMSE) sur les secondes brutes pénaliserait excessivement les longs trajets et ignorerait les petits trajets urbains.
2. **Alignement mathématique parfait** : Si nous définissons la cible d'entraînement comme :
   $$y = \ln(\text{trip\_duration} + 1) = \text{np.log1p}(\text{trip\_duration})$$
   Alors la fonction de perte par défaut de LightGBM, XGBoost et CatBoost (`RMSE`) optimise **exactement et sans approximation** la formule du RMSLE Kaggle !
3. **Inférence inverse** : Lors de la prédiction finale, il suffit de repasser à l'échelle des secondes réelles via :
   $$\hat{y} = \exp(\hat{y}_{\log}) - 1 = \text{np.expm1}(\hat{y}_{\log})$$

---

## 4. Détail de Chaque Étape du Notebook : Ce qu'on fait & Pourquoi on le fait

Voici l'explication pas-à-pas de chacune des sections du notebook [`NYC_Taxi_Trip_Duration_Master.ipynb`](./NYC_Taxi_Trip_Duration_Master.ipynb) :

---

### Étape 1 : Configuration de l'Environnement & Constantes Métier
- **Ce qu'on fait** :
  - Import des librairies : `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `lightgbm`, `xgboost`, `catboost`.
  - Fixation de la graine aléatoire (`RANDOM_STATE = 42`).
  - Définition des constantes géographiques de New York :
    - Bounding Box : Latitude $[40.50, 41.00]$, Longitude $[-74.30, -73.65]$.
    - Angle d'inclinaison de la grille des rues de Manhattan : $28.9^\circ$.
    - Coordonnées des 3 grands aéroports (JFK, LaGuardia, Newark) et des pôles majeurs (Times Square, Wall Street, Grand Central).
    - Calendrier des 10 jours fériés US 2016, de l'Election Day 2016 et des dates du Blizzard Jonas (22–24 janvier 2016).
- **Pourquoi on le fait** :
  - *Reproductibilité* : Garantir que les découpages K-Fold et entraînements produisent exactement les mêmes résultats à chaque exécution.
  - *Contextualisation métier* : Définir ces ancrages géographiques réels permet de créer des descripteurs physiques adaptés aux spécificités de la ville.

---

### Étape 2 : Chargement & Diagnostic des Données
- **Ce qu'on fait** :
  - Lecture des fichiers `train.csv` (1 458 644 lignes) et `test.csv` (625 134 lignes).
  - Contrôle des types de données, de la mémoire consommée, et comptage des valeurs manquantes (`isna().sum()`).
- **Pourquoi on le fait** :
  - *Validation d'intégrité* : Vérifier qu'aucune colonne n'est corrompue et qu'il n'y a aucun `NaN` dans les entrées.
  - *Gain de temps* : Mesurer le volume en RAM pour s'assurer que les calculs tiendront confortablement en mémoire sans saturer la machine.

---

### Étape 3 : Analyse Exploratoire des Données (EDA) & Intuitions Physiques
- **Ce qu'on fait** :
  - **Graphique 1 & 2** : Histogramme comparatif de la durée brute (`trip_duration`) vs la durée transformée en log (`log1p`).
  - **Graphique 3** : Carte de densité 2D hexagonale (`hexbin`) superposée sur les coordonnées de New York avec les aéroports.
  - **Graphique 4** : Carte de chaleur (Heatmap) de la durée médiane selon l'heure (0 à 23h) et le jour de la semaine (lundi au dimanche).
- **Pourquoi on le fait** :
  - *Normalisation de la cible* : En échelle brute, la distribution est écrasée à gauche avec des valeurs aberrantes allant jusqu'à des millions de secondes. Après $\log1\text{p}$, la courbe devient une distribution gaussienne (en cloche) idéale pour les arbres de décision.
  - *Détection des pôles de trafic* : L'hexbin confirme que l'essentiel des courses se concentre dans Manhattan et sur les autoroutes reliant Manhattan aux aéroports JFK et LaGuardia.
  - *Comportement des embouteillages* : La heatmap démontre empiriquement l'existence des heures de pointe : les trajets sont 30% à 40% plus longs à 18h en semaine qu'à 4h du matin.

---

### Étape 4 : Nettoyage des Données & Filtrage des Valeurs Aberrantes
- **Ce qu'on fait** :
  - Suppression des trajets de durée impossible : $< 10$ secondes et $> 22$ heures (79 200 secondes).
  - Suppression des coordonnées GPS situées hors de la région métropolitaine de New York.
  - Calcul de la vitesse moyenne implicite ($v = \text{distance} / \text{durée}$) et suppression des trajets avec $v > 150 \text{ km/h}$ sur plus de 3 km.
  - **Filtrage appliqué uniquement sur `train.csv` (4 657 lignes supprimées, soit 0.32% du dataset).**
- **Pourquoi on le fait** :
  - *Trajets $< 10$ secondes* : Chauffeur ayant engagé le compteur par erreur puis annulé immédiatement la course.
  - *Trajets $> 22$ heures* : Compteurs oubliés pendant des jours voire des mois (durée artificielle sans rapport avec la circulation).
  - *Erreurs GPS* : Relevés erronés indiquant des points en plein océan Atlantique ou sur d'autres continents.
  - *Règle d'or Kaggle* : Le jeu de test (`test.csv`) ne doit **JAMAIS** être filtré, car la plateforme attend exactement une prédiction pour chaque ID présent.

---

### Étape 5 : Ingénierie des Caractéristiques Expertes (Feature Engineering)
La solution s'appuie sur un ensemble complet de variables géométriques et contextuelles :

1. **Distance de Manhattan pivotée à 28.9° (`manhattan_dist_rot`)** :
   - *Pourquoi ?* : Manhattan n'est pas alignée sur les axes cardinaux Nord-Sud. Les avenues sont inclinées d'environ $28.9^\circ$ vers l'Est. En appliquant une rotation de $28.9^\circ$ avant de sommer les écarts absolus, on mesure la **distance réelle le long des avenues et rues**, bien plus fidèle que la distance euclidienne.
2. **Distance de Haversine (`haversine_dist`) & Cap de direction (`bearing`)** :
   - *Pourquoi ?* : Haversine calcule la distance à vol d'oiseau sur la sphère terrestre. Le cap ($-180^\circ$ à $+180^\circ$) indique si le taxi roule vers le Nord (Uptown), le Sud (Downtown/Wall St), ou les arrondissements extérieurs (Queens/Brooklyn).
3. **Log-distances & Interactions Dynamiques (`log_haversine`, `dist_x_hour`, `dist_x_rush`)** :
   - *Pourquoi ?* : 5 km parcourus à 4h du matin ne prennent pas le même temps qu'à 18h en pleine heure de pointe. Les interactions multiplicatives permettent aux arbres d'accéder directement au coût temporel de la distance selon l'heure.
4. **Indicateurs Aéroports & Manhattan (`is_jfk_trip`, `is_lga_trip`, `is_ewr_trip`, `pickup_in_manhattan`, `dropoff_in_manhattan`)** :
   - *Pourquoi ?* : Les courses vers les aéroports empruntent des autoroutes avec des vitesses plus élevées et des grilles tarifaires spécifiques (forfaits régulés TLC).
5. **Clustering Spatial Non-Supervisé (`MiniBatchKMeans`, $k=100$)** :
   - *Pourquoi ?* : Partitionne les coordonnées GPS en 100 quartiers discrets. La combinaison `cluster_pair = pickup_cluster * 100 + dropoff_cluster` encode les flux d'un quartier précis à un autre.

---

### Étape 6 : Target Encoding Cross-Validé sans Fuite (Zero Data Leakage)
- **Ce qu'on fait** :
  - Pour chaque pli de validation croisée, on encode la durée logarithmique moyenne observée pour chaque `cluster_pair`, `pickup_cluster`, `dropoff_cluster`, `hour`, `hour_dow`, et `vendor_hour`.
  - Application d'un **lissage bayésien** ($\alpha = 20$) vers la moyenne globale :
    $$\text{TE}(c) = \frac{\text{mean}(c) \cdot \text{count}(c) + \mu_{\text{global}} \cdot \alpha}{\text{count}(c) + \alpha}$$
- **Pourquoi on le fait** :
  - *Capture de haute dimension* : Permet au modèle d'exploiter la matrice de temps de trajet quartier-à-quartier sans créer des milliers de colonnes One-Hot.
  - *Zéro Data Leakage* : L'encodage est calculé **strictement sur les données d'entraînement du pli en cours**, empêchant tout surapprentissage sur les plis de validation et de test.

---

### Étape 7 : Modélisation 5-Fold Multi-Modèles (LightGBM, XGBoost, CatBoost)
- **Ce qu'on fait** :
  - Entraînement en 5 plis de 3 familles complémentaires d'arbres de décision avec arrêt anticipé (*early stopping*) sur 3 000 itérations :
    1. **LightGBM** (*leaf-wise*, `learning_rate=0.05`, `num_leaves=255`) : Arbres profonds et asymétriques très rapides.
    2. **XGBoost** (*depth-wise*, `learning_rate=0.05`, `max_depth=9`, `tree_method='hist'`) : Arbres équilibrés par niveau.
    3. **CatBoost** (*arbres symétriques*, `learning_rate=0.05`, `depth=8`) : Arbres régularisés résistants au bruit.
- **Pourquoi on le fait** :
  - *Convergence complète* : Avec un taux d'apprentissage fin (`lr=0.05`) et 3 000 rounds, les modèles apprennent toute la complexité des motifs urbains sans s'arrêter prématurément.
  - *Diversité structurelle* : Combiner 3 architectures d'arbres différentes permet de réduire la variance des erreurs résiduelles lors de l'assemblage.

---

### Étape 8 : Analyse d'Importance des Variables & Distribution des Résidus
- **Ce qu'on fait** :
  - Visualisation des 25 variables dominantes selon le gain total de décision (LightGBM gain).
  - Graphique de la distribution des résidus : $\text{résidu} = \log1\text{p}(y_{\text{réel}}) - \log1\text{p}(\hat{y}_{\text{prédit}})$.
  - Tracé du nuage de points *Actual vs Predicted* sur 15 000 échantillons.
- **Pourquoi on le fait** :
  - *Interprétabilité physique* : Confirme que les distances pivotées, les aéroports et le Target Encoding dominent la prise de décision.
  - *Contrôle du biais* : Valide que la distribution des erreurs est centrée sur 0 sans biais de sur-estimation ou de sous-estimation.

---

### Étape 9 : Optimisation du Blend Multi-Modèles
- **Ce qu'on fait** :
  - Recherche systématique par grille des poids optimaux $(w_1, w_2, w_3)$ dans l'espace logarithmique :
    $$\hat{y}_{\text{blend\_log}} = w_1 \cdot \hat{y}_{\text{lgb}} + w_2 \cdot \hat{y}_{\text{xgb}} + w_3 \cdot \hat{y}_{\text{cat}}$$
- **Pourquoi on le fait** :
  - Les poids optimaux identifiés sont **$55\%$ LightGBM + $45\%$ XGBoost**, produisant un score de **0.30636 RMSLE**, surpassant chaque modèle individuel grâce à la compensation des erreurs résiduelles.

---

### Étape 10 : Inférence & Vérification Stricte de la Soumission Kaggle
- **Ce qu'on fait** :
  - Moyennage des 5 plis pour chaque modèle sur `test.csv`.
  - Application du blend optimal et retour aux secondes réelles via $\hat{y} = \text{np.expm1}(\hat{y}_{\text{blend\_log}})$.
  - Post-traitement de sécurité : Borner les durées à $\ge 10.0$ secondes (`np.clip`).
  - Sauvegarde dans **`submission.csv`** et exécution des tests d'intégrité :
    1. Exactement 625 134 lignes.
    2. Colonnes exactes : `id,trip_duration`.
    3. Zéro valeur `NaN`.
    4. Toutes les durées $\ge 10$ secondes.
    5. Alignement parfait des IDs avec `sample_submission.csv`.

---

## 5. Tableau Récapitulatif des Performances

| Modèle | 5-Fold OOF RMSLE | Rôle dans l'Ensemble | Statut |
| :--- | :---: | :--- | :---: |
| **LightGBM Regressor** | **0.30788** | Modèle principal (croissance par feuille) | ✅ Validé (5-fold) |
| **XGBoost Regressor (Hist)** | **0.30845** | Modèle complémentaire (croissance par niveau) | ✅ Validé (5-fold) |
| **CatBoost Regressor** | **0.31878** | Régularisation structurelle (arbres symétriques) | ✅ Validé (5-fold) |
| **Ensemble Blend (0.55 LGB + 0.45 XGB)** | **0.30636** | **Combinaison Pondérée Optimale** | 🏆 **Meilleure Performance** |

> [!NOTE]
> Le score RMSLE Out-Of-Fold de **0.30636** est obtenu grâce à :
> 1. **Target Encoding Cross-Validé sans fuite** sur les paires de quartiers et créneaux horaires.
> 2. **Convergence complète des hyperparamètres** (`lr=0.05`, 3000 rounds, `num_leaves=255`, `max_depth=9`).
> 3. **Caractéristiques expertes** (interactions distance $\times$ heure, log-distances, indicateurs aéroports & Manhattan).
> 4. **Ensemble Blending pondéré** réduisant la variance des prédictions.

---

## 6. Comment Utiliser et Re-exécuter le Projet

### Consultation directe
Le notebook maître est déjà entièrement exécuté et prêt à être consulté dans VS Code ou Jupyter :
👉 **[`NYC_Taxi_Trip_Duration_Master.ipynb`](./NYC_Taxi_Trip_Duration_Master.ipynb)**

### Soumission directe sur Kaggle
Le fichier de soumission est généré, vérifié et prêt à l'emploi :
👉 **[`submission.csv`](./submission.csv)** (11.35 Mo, 625 134 lignes).

### Ré-exécution complète en ligne de commande (si souhaité)
Pour ré-exécuter l'ensemble des 26 cellules du notebook de zéro et régénérer un fichier `submission.csv` :
```powershell
$env:LOKY_MAX_CPU_COUNT = "$env:NUMBER_OF_PROCESSORS"
jupyter nbconvert --to notebook --execute --inplace NYC_Taxi_Trip_Duration_Master.ipynb
```
