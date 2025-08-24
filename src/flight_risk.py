import numpy as np
import pandas as pd
from .utils import year_month
from .sentiment_pipeline import run_and_save as run_sentiment
from .config import NEG_RATIO_W, TREND_DROP_W, LOW_AVG_W, HIGH_RISK_THRESHOLD, LOW_AVG_PIVOT

def _recent_trend(monthly: pd.DataFrame) -> float:
    if len(monthly) < 2:
        return 0.0
    x = np.arange(len(monthly))
    y = monthly["avg_compound"].values
    return float((y[-1] - y[-2]) / (x[-1] - x[-2]))

def compute_flight_risk(sent_df: pd.DataFrame, date_col: str, emp_col: str) -> pd.DataFrame:
    sent_df = sent_df.copy()
    sent_df["ym"] = year_month(sent_df[date_col])
    rows = []
    for emp, sub in sent_df.groupby(emp_col):
        monthly = (sub.groupby("ym")
                   .agg(avg_compound=("compound","mean"),
                        neg_ratio=("sentiment_label", lambda s: (s=="Negative").sum()/len(s)))
                   .reset_index()
                   .sort_values("ym"))
        neg_ratio = monthly["neg_ratio"].iloc[-1] if len(monthly) else 0.0
        recent_avg = monthly["avg_compound"].iloc[-1] if len(monthly) else 0.0
        trend = _recent_trend(monthly)
        score = 100.0*(NEG_RATIO_W*neg_ratio + TREND_DROP_W*max(0.0, -trend) + LOW_AVG_W*max(0.0, LOW_AVG_PIVOT - recent_avg))
        rows.append(dict(employee_id=emp, neg_ratio=float(neg_ratio), recent_avg=float(recent_avg),
                         recent_trend=float(trend), risk_score=float(score), high_risk=bool(score >= HIGH_RISK_THRESHOLD)))
    return pd.DataFrame(rows).sort_values(["high_risk","risk_score"], ascending=[False, False]).reset_index(drop=True)

def run_and_save(path="outputs/flight_risk.csv"):
    sent_df, cols = run_sentiment()
    out = compute_flight_risk(sent_df, cols.date, cols.employee_id)
    out.to_csv(path, index=False)
    return out, cols
