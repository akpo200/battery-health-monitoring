#  Prédiction du SoH des Batteries avec LSTM Par Pascale Nancy Alia AKPO

Ce projet De Master 2 Deep learning implémente un modèle de Deep Learning (**LSTM**) pour prédire l’état de santé (**State of Health - SoH**) des batteries à partir de mesures électriques (tension, courant, température, SoC) collectées sur des cycles de décharge.

## 📁 Structure du Projet

- `data/` : Contient le dataset brut et les données prétraitées.
- `notebooks/` : Visualisations et historique d'entraînement.
- `models/` : Modèle LSTM sauvegardé (`.h5`) et métriques.
- `app/` : Application Web interactive (Streamlit).
- `preprocess.py` : Script de nettoyage et division en fenêtres glissantes.
- `train_lstm.py` : Script d'entraînement du réseau de neurones.
- `evaluate.py` : Script d'évaluation des performances (MAE, RMSE, R²).

## 🚀 Installation & Lancement

### 1. Prérequis
Installez les dépendances nécessaires :
```bash
pip install -r requirements.txt
```

### 2. Prétraitement des données
```bash
python preprocess.py
```

### 3. Entraînement du modèle
```bash
python train_lstm.py
```

### 4. Évaluation
```bash
python evaluate.py
```

### 5. Lancer l'application interactive
```bash
cd app
streamlit run app.py
```

##  Modèle LSTM
Le modèle utilise deux couches LSTM suivies de couches Dropout pour éviter le sur-apprentissage, et une couche Dense finale pour la régression du SoH.
Les données sont découpées en **fenêtres glissantes de 10 points** pour capturer la dynamique temporelle du signal.

##  Métriques (Exemple)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R² Score** (Corrélation entre réel et prédit)
