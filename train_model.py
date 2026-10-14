from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

DATA_PATH = Path("data/spam_dataset.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "spam_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"

if not DATA_PATH.exists():
    raise SystemExit("Dataset not found. Run: python download_dataset.py")

df = pd.read_csv(DATA_PATH)
required = {"label", "message"}
if not required.issubset(df.columns):
    raise SystemExit("Dataset must contain columns named 'label' and 'message'.")

df = df.dropna(subset=["label", "message"]).copy()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df["message"] = df["message"].astype(str).str.strip()
df = df[df["label"].isin(["ham", "spam"]) & df["message"].ne("")]
df = df.drop_duplicates(subset=["label", "message"])
if df["label"].nunique() != 2 or df["label"].value_counts().min() < 2:
    raise SystemExit("Need examples of both ham and spam labels to train.")

X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Keep the original text. Removing punctuation/numbers can discard URL and scam clues.
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                               strip_accents="unicode", sublinear_tf=True,
                               max_features=100000)),
    ("classifier", MultinomialNB(alpha=0.5)),
])
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

metrics = {
    "dataset_rows": int(len(df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
    "accuracy": float(accuracy_score(y_test, pred)),
    "spam_precision": float(precision_score(y_test, pred, pos_label=1, zero_division=0)),
    "spam_recall": float(recall_score(y_test, pred, pos_label=1, zero_division=0)),
    "spam_f1": float(f1_score(y_test, pred, pos_label=1, zero_division=0)),
    "confusion_matrix_labels": ["ham", "spam"],
    "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
}
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)
METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

print(f"Dataset rows: {len(df):,}")
print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
print("\nClassification report:")
print(classification_report(y_test, pred, labels=[0, 1],
                            target_names=["Legitimate (ham)", "Spam"], zero_division=0))
print("Confusion matrix (rows=true, columns=predicted; order ham, spam):")
print(confusion_matrix(y_test, pred, labels=[0, 1]))
print(f"\nSaved model: {MODEL_PATH}")
print(f"Saved metrics: {METRICS_PATH}")
