import argparse
from pathlib import Path
from . import sentiment_pipeline, eda, monthly_scoring, ranking, flight_risk, trend_regression, ml_sentiment

def main():
    p = argparse.ArgumentParser(description="Greentree Assessment Pipeline")
    p.add_argument("--sentiment", action="store_true", help="Run sentiment labeling")
    p.add_argument("--eda", action="store_true", help="Run EDA and plots")
    p.add_argument("--monthly", action="store_true", help="Monthly sentiment scoring")
    p.add_argument("--rank", action="store_true", help="Employee ranking")
    p.add_argument("--risk", action="store_true", help="Flight risk identification")
    p.add_argument("--trend", action="store_true", help="Linear regression for trends")
    p.add_argument("--ml-sentiment", action="store_true", help="Train and apply sklearn sentiment model")
    p.add_argument("--all", action="store_true", help="Run everything")
    args = p.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.all or args.sentiment:
        sentiment_pipeline.run_and_save()
    if args.all or args.eda:
        eda.run_and_save()
    if args.all or args.monthly:
        monthly_scoring.run_and_save()
    if args.all or args.rank:
        ranking.run_and_save()
    if args.all or args.risk:
        flight_risk.run_and_save()
    if args.all or args.trend:
        trend_regression.run_and_save()
    if args.all or args.ml_sentiment:
        ml_sentiment.train()
        ml_sentiment.predict()

if __name__ == "__main__":
    main()
