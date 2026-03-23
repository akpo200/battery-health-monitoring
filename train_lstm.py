import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt

# 1. Chargement des données prétraitées
print("📂 Chargement des données...")
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')

# 2. CONSTRUCTION DU MODELE LSTM
# Un LSTM est efficace pour traiter les séries temporelles comme nos mesures de batterie.
# On empile deux couches LSTM pour mieux capturer les dépendances temporelles complexes.

model = Sequential([
    # Première couche LSTM avec 64 unités, return_sequences=True permet de chaîner avec une autre couche LSTM
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.2), # Réduit le sur-apprentissage (overfitting) en désactivant 20% des neurones aléatoirement
    
    # Deuxième couche LSTM avec 32 unités
    LSTM(32),
    Dropout(0.2),
    
    # Couche finale pour la régression (prédire une valeur continue : le SoH)
    Dense(1)
])

# 3. COMPILATION
# On utilise l'optimiseur Adam et la perte MSE (Mean Squared Error) adaptée à la régression.
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("🧠 Architecture du modèle :")
model.summary()

# 4. CALLBACKS
# On sauvegarde le meilleur modèle au fur et à mesure
checkpoint = ModelCheckpoint('models/battery_soh_lstm.keras', monitor='val_loss', save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. ENTRAÎNEMENT
print("🚀 Début de l'entraînement (20 époques max)...")
history = model.fit(
    X_train, y_train,
    epochs=20, # Réduit pour la démo, suffisant pour une bonne MAE
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# 6. SAUVEGARDE DU MODÈLE
os.makedirs('models', exist_ok=True)
model.save('models/battery_soh_lstm.keras')
print("💾 Modèle sauvegardé dans models/battery_soh_lstm.keras")

# 7. PERFORMANCE SUR LE TEST SET
print("📊 Évaluation sur le set de test...")
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"✅ MAE sur test: {test_mae:.4f}")

# 8. VISUALISATION DE LA LOSS (optionnel, pour vérification interne)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Performance pendant l\'entraînement')
plt.legend()
plt.savefig('notebooks/training_history.png')
print("📈 Courbe d'apprentissage sauvegardée dans notebooks/training_history.png")
