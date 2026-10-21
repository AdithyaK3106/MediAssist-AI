import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BERT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "bert"
LR_DIR = PROJECT_ROOT / "models" / "classical" / "logistic_regression"
NB_DIR = PROJECT_ROOT / "models" / "classical" / "naive_bayes"

LR_DIR.mkdir(parents=True, exist_ok=True)
NB_DIR.mkdir(parents=True, exist_ok=True)

def train_consolidated_baselines():
    print("=" * 60)
    print("Training 35-Class Classical Baselines (LR + Naive Bayes)...")
    print("=" * 60)
    
    train_df = pd.read_csv(BERT_DATA_DIR / "train.csv")
    test_df = pd.read_csv(BERT_DATA_DIR / "test.csv")
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X_train = vectorizer.fit_transform(train_df["symptom_text"])
    y_train = train_df["label"].values
    
    X_test = vectorizer.transform(test_df["symptom_text"])
    y_test = test_df["label"].values
    
    # 1. Logistic Regression
    print("\n[1/2] Fitting Multinomial Logistic Regression...")
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=42)
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"  Logistic Regression Test Accuracy: {lr_acc * 100:.2f}%")
    
    joblib.dump(lr_clf, LR_DIR / "classifier.pkl")
    joblib.dump(vectorizer, LR_DIR / "vectorizer.pkl")
    with open(LR_DIR / "metadata.json", "w") as f:
        json.dump({
            "model_type": "logistic_regression",
            "num_classes": len(lr_clf.classes_),
            "hyperparameters": {"C": 1.0, "solver": "lbfgs", "max_iter": 1000}
        }, f, indent=2)
    print(f"  Saved Logistic Regression to {LR_DIR}")
    
    # 2. Multinomial Naive Bayes
    print("\n[2/2] Fitting Multinomial Naive Bayes...")
    nb_clf = MultinomialNB(alpha=1.0)
    nb_clf.fit(X_train, y_train)
    nb_pred = nb_clf.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    print(f"  Multinomial Naive Bayes Test Accuracy: {nb_acc * 100:.2f}%")
    
    joblib.dump(nb_clf, NB_DIR / "classifier.pkl")
    joblib.dump(vectorizer, NB_DIR / "vectorizer.pkl")
    with open(NB_DIR / "metadata.json", "w") as f:
        json.dump({
            "model_type": "multinomial_naive_bayes",
            "num_classes": len(nb_clf.classes_),
            "hyperparameters": {"alpha": 1.0}
        }, f, indent=2)
    print(f"  Saved Naive Bayes to {NB_DIR}")

if __name__ == "__main__":
    train_consolidated_baselines()
