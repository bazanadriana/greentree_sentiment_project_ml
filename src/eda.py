import os
import pandas as pd
import matplotlib.pyplot as plt
from .sentiment_pipeline import run_and_save as run_sentiment
from .utils import year_month

def eda_plots(sent_df: pd.DataFrame, date_col: str):
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)

    sent_df["sentiment_label"].value_counts().sort_index().plot(kind="bar", title="Sentiment Label Distribution")
    plt.tight_layout(); plt.savefig("outputs/fig_label_distribution.png"); plt.savefig("visualizations/fig_label_distribution.png"); plt.clf()

    sent_df["compound"].plot(kind="hist", bins=30, title="Compound Sentiment Histogram")
    plt.tight_layout(); plt.savefig("outputs/fig_compound_hist.png"); plt.savefig("visualizations/fig_compound_hist.png"); plt.clf()

    monthly = sent_df.groupby(year_month(sent_df[date_col]))["compound"].mean()
    monthly.plot(kind="line", marker="o", title="Monthly Avg Compound")
    plt.tight_layout(); plt.savefig("outputs/fig_monthly_avg_compound.png"); plt.savefig("visualizations/fig_monthly_avg_compound.png"); plt.clf()

def run_and_save():
    sent_df, cols = run_sentiment()
    eda_plots(sent_df, cols.date)
    return sent_df, cols
