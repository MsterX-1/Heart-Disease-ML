# scripts/train_pipeline.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load preprocessed dataset (after feature selection)
DATA_PATH = r"E:\Projects\Python Projects\ML\heart-disease-ml\notebooks\data\processed_cleaned.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Expected cleaned dataset at data/heart_disease_selected.csv. Run preprocessing notebook first.")

df = pd.read_csv(DATA_PATH)
X = df.drop(columns=['target']).values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

pipeline.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/final_model.pkl")
print("Model saved to models/final_model.pkl")
