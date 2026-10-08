import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BERT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"
MODEL_DIR = PROJECT_ROOT / "models" / "classical" / "logistic_regression"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

def train_consolidated_baseline():
    print("Training calibrated baseline classifier on consolidated dataset...")
    train_df = pd.read_csv(BERT_DATA_DIR / "train.csv")
    test_df = pd.read_csv(BERT_DATA_DIR / "test.csv")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["symptom_text"])
    y_train = train_df["label"].values
    
    X_test = vectorizer.transform(test_df["symptom_text"])
    y_test = test_df["label"].values
    
    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Baseline Test Accuracy: {acc * 100:.2f}%")
    
    # Save vectorizer and classifier
    joblib.dump(clf, MODEL_DIR / "classifier.pkl")
    joblib.dump(vectorizer, MODEL_DIR / "vectorizer.pkl")
    print(f"Saved calibrated baseline to {MODEL_DIR}")

if __name__ == "__main__":
    train_consolidated_baseline()
