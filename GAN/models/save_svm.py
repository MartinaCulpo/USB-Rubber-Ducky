import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("sampled_balanced_dataset_small.csv")
X = df[["Hold", "Flight", "jitter"]]
y = df["label"]

X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = SVC(probability=True, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "models/svm.joblib")
print(" SVM salvato in models/svm.joblib")
