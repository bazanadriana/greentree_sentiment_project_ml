import os, json
import pandas as pd
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from .sentiment_pipeline import label_sentiment
from .utils import read_dataset

MODEL_PATH = "outputs/ml_sentiment.joblib"
REPORT_PATH = "outputs/ml_sentiment_report.json"

def _ensure_labels(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    return label_sentiment(df, text_col)

def train(text_col: str = None, test_size: float = 0.2, random_state: int = 42):
    df, cols = read_dataset()
    text_col = text_col or cols.text
    labeled = _ensure_labels(df, text_col)
    X = labeled[text_col].fillna("").astype(str)
    y = labeled["sentiment_label"].astype(str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    os.makedirs("outputs", exist_ok=True)
    dump(pipe, MODEL_PATH)
    with open(REPORT_PATH, "w") as f: json.dump({"accuracy": acc, "report": report}, f, indent=2)
    return {"accuracy": acc, "report": report, "model_path": MODEL_PATH, "labels": sorted(y.unique())}

def predict(save_path: str = "outputs/sentiment_scored_ml.csv"):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first (use --ml-sentiment)")
    model = load(MODEL_PATH)
    df, cols = read_dataset()
    preds = model.predict(df[cols.text].fillna("").astype(str))
    out = df.copy()
    out["ml_sentiment_label"] = preds
    out.to_csv(save_path, index=False)
    return out, cols
