import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

# 1. Chargement des données
print("Chargement des données...")
df = pd.read_csv('data/battery_health_dataset.csv')

# 2. SELECTION DES VARIABLES
# X: Tension, Courant, Température, SoC, cycle_number
# y: SoH
features = ['Voltage_measured', 'Current_measured', 'Temperature_measured', 'SoC', 'cycle_number']
target = 'SoH'

# 3. NORMALISATION
# On normalise pour aider le modèle LSTM à converger
print("Normalisation des données...")
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

df[features] = scaler_x.fit_transform(df[features])
df[[target]] = scaler_y.fit_transform(df[[target]])

# 4. FENÊTRE GLISSANTE (Sliding Window)
# Chaque cycle doit être traité séparément pour ne pas mélanger les séquences entre batteries/cycles
def create_sequences(df, window_size=10):
    X, y = [], []
    # On groupe par batterie et cycle
    grouped = df.groupby(['battery_id', 'cycle_number'])
    
    for _, group in grouped:
        # On ne prend que les features
        data = group[features].values
        # Le SoH est constant pour un cycle donné
        soh_label = group[target].iloc[0]
        
        # Création des fenêtres dans le cycle
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i+window_size])
            y.append(soh_label)
            
    return np.array(X), np.array(y)

window_size = 10
print(f"Creation des sequences (fenetre = {window_size})...")
X, y = create_sequences(df, window_size)

print(f"Forme de X: {X.shape}") # (Samples, Time_steps, Features)
print(f"Forme de y: {y.shape}")

# 5. SÉPARATION TRAIN / TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. SAUVEGARDE ET EXPORT
import pickle
os.makedirs('models', exist_ok=True)
with open('models/scaler_x.pkl', 'wb') as f:
    pickle.dump(scaler_x, f)
with open('models/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)
print("Scalers sauvegardés dans models/scaler_x.pkl et models/scaler_y.pkl")

os.makedirs('data/processed', exist_ok=True)
np.save('data/processed/X_train.npy', X_train)
np.save('data/processed/X_test.npy', X_test)
np.save('data/processed/y_train.npy', y_train)
np.save('data/processed/y_test.npy', y_test)

print("💾 Données prétraitées et sauvegardées dans data/processed/")
