import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from .utils import read_dataset
from .config import POS_THRESH, NEG_THRESH

def label_sentiment(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    analyzer = SentimentIntensityAnalyzer()
    scores = df[text_col].fillna("").astype(str).map(lambda t: analyzer.polarity_scores(t)["compound"])
    label = scores.map(lambda c: "Positive" if c >= POS_THRESH else ("Negative" if c <= NEG_THRESH else "Neutral"))
    out = df.copy()
    out["compound"] = scores
    out["sentiment_label"] = label
    return out

def run_and_save(path="outputs/sentiment_scored.csv"):
    df, cols = read_dataset()
    out = label_sentiment(df, cols.text)
    # Write CSV to avoid parquet engine problems across environments
    out.to_csv(path, index=False)
    return out, cols
