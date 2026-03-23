import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import json
import os
import pickle

# --- CONFIGURATION DES CHEMINS ---
# On utilise des chemins relatifs par rapport à la racine du projet pour Streamlit Cloud.
MODEL_PATH = 'models/battery_soh_lstm.keras'
SCALER_X_PATH = 'models/scaler_x.pkl'
SCALER_Y_PATH = 'models/scaler_y.pkl'
DATA_PATH = 'data/battery_health_dataset.csv'
METRICS_PATH = 'models/metrics_results.json'
TEST_DATA_X = 'data/processed/X_test.npy'
TEST_DATA_Y = 'data/processed/y_test.npy'

# --- CONFIGURATION DE LA PAGE STREAMLIT ---
# On configure ici l'apparence générale du tableau de bord.
# "page_title" donne un nom à l'onglet du navigateur web.
# "layout='wide'" permet d'utiliser toute la largeur de l'écran pour l'affichage.
st.set_page_config(
    page_title="Projet Nancy AKPO | Monitoring Batterie",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- DESIGN ÉPURÉ (White & Midnight Blue) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #FFFFFF;
        color: #121212;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0D1B2A; 
        color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    
    h1 {
        font-weight: 800;
        color: #000000 !important;
        margin-bottom: 2rem !important;
    }
    h2, h3 {
        color: #1B263B !important;
        font-weight: 600;
    }
    
    .stMetric {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #E9ECEF;
    }
    
    .signature-container {
        padding: 1rem;
        text-align: center;
        margin-top: 2rem;
    }
    .signature-name {
        font-weight: 400;
        color: #FFFFFF;
        font-size: 0.95rem;
    }
    
    .stButton>button {
        background-color: #1B263B !important;
        color: #FFFFFF !important;
        border: none !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100%;
    }
    
    /* FIXER LA SIDEBAR : Masquer les boutons de réduction/agrandissement */
    [data-testid="stSidebarNav"] + div {visibility: hidden;}
    button[aria-label="Open sidebar"], button[aria-label="Fermer la barre latérale"], button[aria-label="Close"] {
        display: none !important;
    }
    
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# --- CHARGEMENT DU MODÈLE ET MISE EN CACHE ---
# @st.cache_resource permet à Streamlit de garder le modèle AI (Tensorflow) en mémoire,
# ainsi il n'aura pas à recharger le modèle depuis le fichier .keras à chaque action de l'utilisateur.
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try: return tf.keras.models.load_model(MODEL_PATH)
        except: return None
    return None

# @st.cache_data fonctionne comme @st.cache_resource mais est spécifiquement conçu
# pour stocker des données comme notre DataFrame Pandas afin de charger plus vite.
@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH): return pd.read_csv(DATA_PATH)
    return None

model = load_model()
df = load_dataset()

# --- CHARGEMENT SCALERS ---
try:
    if os.path.exists(SCALER_X_PATH) and os.path.exists(SCALER_Y_PATH):
        with open(SCALER_X_PATH, 'rb') as f: scaler_x = pickle.load(f)
        with open(SCALER_Y_PATH, 'rb') as f: scaler_y = pickle.load(f)
except:
    scaler_x, scaler_y = None, None

# BARRE LATÉRALE - FIXE
st.sidebar.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h2 style='color: white !important; font-size: 1.2rem;'>MONITORING</h2>
    </div>
    """, unsafe_allow_html=True)

# --- BARRE DE NAVIGATION (SIDEBAR) ---
# On crée ici un menu avec trois options sous forme de boutons radio : Accueil, Performance, et Simulation.
# Ce menu est placé dans "st.sidebar" (à gauche de l'écran).
page = st.sidebar.radio("Navigation", ["Accueil", "Performance", "Simulation"])

st.sidebar.markdown(f"""
    <div class="signature-container">
        <span class="signature-name">Projet réalisé par Nancy AKPO</span>
    </div>
    """, unsafe_allow_html=True)

# --- PAGE ACCUEIL ---
if page == "Accueil":
    # st.title et st.markdown permettent d'afficher du texte (les titres et les descriptions).
    st.title("Interface Intelligente de Monitoring")
    
    st.markdown("""
    ### Bienvenue Nancy AKPO
    Ce tableau de bord permet de superviser l'état de santé (SoH) des batteries grâce à une intelligence artificielle avancée.
    
    **Vision Stratégique :**
    Le but est de transformer des séquences de mesures électriques (Tension, Courant, Température) en un diagnostic précis sur la longévité de l'équipement.
    """)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Séquences d'apprentissage : Les données sont traitées via une fenêtre glissante pour plus de précision.")
    with c2:
        st.success("Objectif Final : Prédire le vieillissement pour optimiser la maintenance préventive.")

# --- PAGE PERFORMANCE ---
# Si l'utilisateur clique sur "Performance" dans le menu de gauche, ce bloc de code s'exécute.
elif page == "Performance":
    st.title("Performance du Modèle")
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f: m = json.load(f)
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{m['mae']:.2f}%")
        col2.metric("RMSE", f"{m['rmse']:.2f}%")
        col3.metric("R2 Score", f"{m['r2']:.2f}")
    else:
        st.error("Métriques non trouvées.")

    nb_plot = 'notebooks/actual_vs_predicted.png'
    if os.path.exists(nb_plot):
        st.image(nb_plot, caption="Précision LSTM : Réel vs Prédit", use_container_width=True)

# --- PAGE SIMULATION ---
# Section dédiée à la prédiction en direct du SoH (State Of Health ou État de Santé).
elif page == "Simulation":
    st.title("Diagnostic Prédictif")
    
    if model is None or scaler_x is None:
        st.error("Services indisponibles.")
    else:
        st.subheader("Paramètres")
        c1, c2, c3 = st.columns(3)
        v = c1.slider("Tension (V)", 3.0, 4.5, 3.8)
        i = c2.slider("Courant (A)", -2.0, 2.0, 0.5)
        t = c3.slider("Température (C)", 20.0, 50.0, 30.0)
        soc = c1.slider("SoC (%)", 0.0, 100.0, 80.0)
        cycle = c2.number_input("Cycle", 1, 1000, 100)
        
        # Si l'utilisateur clique sur le bouton "LANCER LA PREDICTION"...
        if st.button("LANCER LA PREDICTION"):
            # 1. On liste les valeurs entrées par l'utilisateur
            raw = np.array([[v, i, t, soc, cycle]])
            # 2. On transforme ces données à l'aide de "scaler_x" pour qu'elles aient 
            #    la même échelle que les données avec lesquelles le modèle a appris
            norm = scaler_x.transform(raw)
            # 3. On crée une séquence (le LSTM a besoin de voir l'historique sur une fenêtre de 10)
            seq = np.repeat(norm, 10, axis=0).reshape(1, 10, 5)
            
            # 4. Le modèle AI génère une prédiction (valeur normalisée)
            p_norm = model.predict(seq, verbose=0)
            # 5. On repasse aux valeurs réelles en % avec "scaler_y" (inverse_transform)
            res = scaler_y.inverse_transform(p_norm)[0][0]
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"""
                <div style='background: #0D1B2A; padding: 30px; border-radius: 12px; color: white; text-align: center;'>
                    <span style='font-size: 0.9rem; opacity: 0.8;'>SOH PREDIT</span><br>
                    <span style='font-size: 2.5rem; font-weight: 800;'>{res:.2f}%</span>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = res,
                    gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "#0D1B2A"},
                             'steps': [{'range': [0, 70], 'color': "#FF8A8A"},
                                       {'range': [70, 85], 'color': "#FFD58A"},
                                       {'range': [85, 100], 'color': "#8AFFD5"}]}))
                fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Validation sur Historique")
        if os.path.exists(TEST_DATA_X):
            X, Y = np.load(TEST_DATA_X), np.load(TEST_DATA_Y)
            idx = st.number_input("Echantillon ID", 0, len(X)-1, 42)
            if st.button("ANALYSER L'ERREUR"):
                rv = scaler_y.inverse_transform(Y[idx:idx+1].reshape(-1, 1))[0][0]
                pv = scaler_y.inverse_transform(model.predict(X[idx:idx+1], verbose=0))[0][0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Valeur Reelle", f"{rv:.2f}%")
                c2.metric("Prediction AI", f"{pv:.2f}%")
                c3.metric("Ecart", f"{abs(rv-pv):.4f}%")
