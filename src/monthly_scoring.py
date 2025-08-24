import pandas as pd
from .utils import year_month
from .sentiment_pipeline import run_and_save as run_sentiment

def monthly_scores(sent_df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    g = sent_df.groupby(year_month(sent_df[date_col]))
    out = g.agg(
        avg_compound=("compound", "mean"),
        n=("compound", "size"),
        pos=("sentiment_label", lambda s: (s=="Positive").sum()),
        neu=("sentiment_label", lambda s: (s=="Neutral").sum()),
        neg=("sentiment_label", lambda s: (s=="Negative").sum()),
    ).reset_index(names="year_month")
    out["pos_ratio"] = out["pos"]/out["n"]
    out["neg_ratio"] = out["neg"]/out["n"]
    out["neu_ratio"] = out["neu"]/out["n"]
    return out

def run_and_save(path="outputs/monthly_scores.csv"):
    sent_df, cols = run_sentiment()
    out = monthly_scores(sent_df, cols.date)
    out.to_csv(path, index=False)
    return out, cols
