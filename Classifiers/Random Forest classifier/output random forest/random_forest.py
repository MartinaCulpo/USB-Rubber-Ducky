import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 1. Carica il dataset bilanciato
df = pd.read_csv("sampled_balanced_dataset.csv")

# 2. Preprocessing
# Usa solo feature numeriche rilevanti
features = ["Hold", "Flight", "jitter"]
X = df[features]

# Crea label binaria: 0 = human, 1 = PRNG
df["label"] = df["source"].apply(lambda x: 0 if x == "human" else 1)
y = df["label"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Allena Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Valutazione
y_pred = clf.predict(X_test)
print("✅ Accuracy:", clf.score(X_test, y_test))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
