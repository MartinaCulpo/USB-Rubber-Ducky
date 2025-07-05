import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

from xgboost import XGBClassifier

# === 1. Setup cartelle output ===
os.makedirs("reports", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# === 2. Caricamento dati ===
df = pd.read_csv("sampled_balanced_dataset_labeled.csv")

# === 3. Feature e target ===
features = ['Hold', 'Flight', 'jitter']
X = df[features]
y = df['label']

# === 4. Train/test split + scaling ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 5. Inizializza e allena XGBoost ===
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# === 6. Report testuale ===
report = classification_report(y_test, y_pred)
print("\n📊 Classification Report:\n", report)

# Salva report
with open("reports/xgboost_report.txt", "w") as f:
    f.write(report)

# === 7. Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("plots/xgboost_confusion_matrix.png")
plt.show()

# === 8. ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc_score:.2f})", color='darkgreen')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve - XGBoost")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/xgboost_roc_curve.png")
plt.show()
