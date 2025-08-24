import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .utils import year_month
from .sentiment_pipeline import run_and_save as run_sentiment

def fit_trend(sent_df: pd.DataFrame, date_col: str, out_json="outputs/trend_regression.json") -> dict:
    monthly = (sent_df.groupby(year_month(sent_df[date_col]))["compound"].mean()
               .reset_index(name="avg_compound").rename(columns={date_col:"year_month"}))
    monthly["t"] = np.arange(1, len(monthly)+1).reshape(-1,)
    X, y = monthly[["t"]].values, monthly["avg_compound"].values
    model = LinearRegression().fit(X, y)
    payload = dict(r2=float(model.score(X, y)), coef=float(model.coef_[0]), intercept=float(model.intercept_), n=len(monthly))
    with open(out_json, "w") as f: json.dump(payload, f, indent=2)
    return payload, monthly

def run_and_save():
    sent_df, cols = run_sentiment()
    return fit_trend(sent_df, cols.date)
