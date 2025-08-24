import pandas as pd
from .utils import year_month
from .sentiment_pipeline import run_and_save as run_sentiment
from .config import RANK_MONTHS

def employee_ranking(sent_df: pd.DataFrame, date_col: str, emp_col: str) -> pd.DataFrame:
    sent_df = sent_df.copy()
    sent_df["ym"] = year_month(sent_df[date_col])
    months = sorted(sent_df["ym"].dropna().unique())
    use_months = months[-RANK_MONTHS:] if len(months) >= RANK_MONTHS else months
    window = sent_df[sent_df["ym"].isin(use_months)]
    agg = window.groupby(emp_col).agg(avg_compound=("compound","mean"), n=("compound","size")).reset_index()
    latest = use_months[-1] if use_months else None
    latest_avg = window[window["ym"]==latest].groupby(emp_col)["compound"].mean().rename("latest_month_avg")
    out = agg.merge(latest_avg, on=emp_col, how="left").fillna({"latest_month_avg": agg["avg_compound"].mean()})
    out = out.sort_values(["avg_compound","latest_month_avg","n"], ascending=[False, False, False]).reset_index(drop=True)
    out["rank"] = range(1, len(out)+1)
    return out

def run_and_save(path="outputs/employee_ranking.csv"):
    sent_df, cols = run_sentiment()
    out = employee_ranking(sent_df, cols.date, cols.employee_id)
    out.to_csv(path, index=False)
    return out, cols
