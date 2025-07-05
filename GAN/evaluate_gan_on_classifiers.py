import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# === CONFIG ===
GAN_FILE = "gan_generated_sequences.csv"
CLASSIFIERS = {
    "random_forest": "models/random_forest.joblib",
    "xgboost": "models/xgboost.joblib",
    "svm": "models/svm.joblib",
    "knn": "models/knn.joblib"
}
HUMAN_SAMPLE_FILE = "sampled_balanced_dataset_labeled.csv"
N_HUMAN = 5000  # quante righe human per confronto

# === 1. CARICA GAN E HUMAN SAMPLES ===
print(" Caricamento dati GAN...")
gan_df = pd.read_csv(GAN_FILE)

# Costruisci DataFrame a righe
rows = []
for i, row in gan_df.iterrows():
    for j in range(10):
        hold = row[f"hold_{j+1}"]
        flight = row[f"flight_{j+1}"]
        jitter = np.random.uniform(10, 400)  # stima media jitter
        rows.append({
            "Hold": hold,
            "Flight": flight,
            "jitter": jitter,
            "label": 1  # machine
        })

gan_ready = pd.DataFrame(rows)

print(" Caricamento dati HUMAN...")
df_human = pd.read_csv(HUMAN_SAMPLE_FILE)
df_human = df_human[df_human['label'] == 0].sample(n=N_HUMAN, random_state=42)
df_human_ready = df_human[["Hold", "Flight", "jitter", "label"]]

# === 2. UNISCI I DUE DATASET ===
df_test = pd.concat([df_human_ready, gan_ready], ignore_index=True)
X = df_test[["Hold", "Flight", "jitter"]]
y = df_test["label"]

# === 3. STANDARDIZZAZIONE ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 4. TESTA I CLASSIFICATORI ===
for name, path in CLASSIFIERS.items():
    try:
        model = joblib.load(path)
        preds = model.predict(X_scaled)
        print(f"\n {name.upper()} Results:")
        print(classification_report(y, preds, digits=4))
    except Exception as e:
        print(f" Errore con {name}: {e}")
