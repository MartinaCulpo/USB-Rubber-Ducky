import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("sampled_balanced_dataset_labeled.csv")
X = df[["Hold", "Flight", "jitter"]]
y = df["label"]

X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

joblib.dump(model, "models/knn.joblib")
print(" k-NN salvato in models/knn.joblib")
