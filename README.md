# TP Random Forest — Classification de chiffres manuscrits

## Description du projet

Ce projet est un travail pratique (TP) réalisé dans le cadre du Master 2 Data & Business Intelligence à l'ISTIC - Université de Rennes. Il porte sur l'utilisation des **forêts aléatoires (Random Forest)** pour la classification de chiffres manuscrits issus du dataset MNIST.

Le TP fait suite au TP sur les arbres de décision (voir **https://github.com/Oumou30/tp-decision-tree/**) et explore comment les méthodes d'ensemble améliorent les performances par rapport à un arbre unique. L'impact de la représentation des données est également étudié via les descripteurs HOG.

## Structure du notebook

### 1. Random Forest sur les pixels bruts

- **Chargement et séparation des données** : split Train / Validation / Test (70% / 15% / 15%) respectivement sur un échantillon de 1000 images MNIST (28×28 pixels, 10 classes)
- **Construction d'une forêt aléatoire** avec `RandomForestClassifier` de Scikit-learn
- **Optimisation du nombre d'arbres** : itération sur différentes valeurs de `n_estimators` (de 11 à 50) pour sélectionner la configuration optimale via le score de validation
- **Estimation de l'erreur de généralisation** sur l'ensemble de test

### 2. Random Forest sur les descripteurs HOG

- **Transformation HOG** des images : conversion des 784 pixels en vecteurs de 32 features (8 orientations, blocs de 14×14 pixels)
- **Optimisation et évaluation** : même procédure d'optimisation du nombre d'arbres sur les données HOG
- **Comparaison** des erreurs de généralisation entre pixels bruts et représentation HOG

## Technologies et bibliothèques

- **Python 3**
- **NumPy** / **Pandas** - manipulation de données
- **Matplotlib** - visualisation
- **Scikit-learn** - `RandomForestClassifier`, `train_test_split`
- **Scikit-image** - lecture d'images et extraction de descripteurs HOG (`skimage.feature.hog`)

## Concepts clés abordés

- Méthodes d'ensemble : bagging et forêts aléatoires
- Influence du nombre d'arbres sur les performances
- Représentation d'images par descripteurs HOG vs pixels bruts
- Sélection de modèle par validation et estimation de l'erreur de généralisation

## Fichiers nécessaires

| Fichier | Description |
|---|---|
| `TP3_Random_Forest_student.ipynb` | Notebook principal du TP |
| `cp_sample.csv` | Échantillon MNIST (1000 images) |
| `test_1.png` | Image de test 28×28 pour la prédiction personnalisée |

## Comment exécuter

1. Installer les dépendances :
   ```bash
   pip install numpy pandas matplotlib scikit-learn scikit-image
   ```
2. Placer les fichiers de données (`cp_sample.csv`, `test_1.png`) dans le même répertoire que le notebook.
3. Lancer le notebook :
   ```bash
   jupyter notebook TP3_Random_Forest_student.ipynb
   ```

## Résultats observés

- La forêt aléatoire surpasse l'arbre de décision unique grâce à l'agrégation de plusieurs arbres.
- La représentation HOG améliore les performances de classification par rapport aux pixels bruts en fournissant des features plus discriminantes.

## Auteure

**Oumou** - Master 2 Data & Business Intelligence, ISTIC - Université de Rennes
