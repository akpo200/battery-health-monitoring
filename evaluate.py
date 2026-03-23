import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import pickle
import json

# 1. Chargement des données, modèle et scalers
print("Chargement du modèle, des données et des scalers...")
model = tf.keras.models.load_model('models/battery_soh_lstm.keras')
X_test = np.load('data/processed/X_test.npy')
y_test_norm = np.load('data/processed/y_test.npy')

with open('models/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# 2. PRÉDICTION
print("Prediction sur le set de test...")
y_pred_norm = model.predict(X_test)

# 3. DÉ-NORMALISATION POUR RÉSULTATS RÉELS (%)
y_test = scaler_y.inverse_transform(y_test_norm.reshape(-1, 1))
y_pred = scaler_y.inverse_transform(y_pred_norm)

# 4. CALCUL DES MÉTRIQUES (Sur échelle réelle)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- PERFORMANCE DU MODÈLE (Echelle réelle %) ---")
print(f"MAE:   {mae:.4f} %")
print(f"RMSE:  {rmse:.4f} %")
print(f"R2:    {r2:.4f}")

# 5. VISUALISATION
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('SoH Réel (%)')
plt.ylabel('SoH Prédit (%)')
plt.title('Vérification : SoH Réel vs SoH Prédit')
plt.grid(True)
os.makedirs('notebooks', exist_ok=True)
plt.savefig('notebooks/actual_vs_predicted.png')
print("Graphique sauvegardé dans notebooks/actual_vs_predicted.png")

# 6. EXPORT DES RÉSULTATS
results = {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)}
with open('models/metrics_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("Metriques sauvegardées dans models/metrics_results.json")
