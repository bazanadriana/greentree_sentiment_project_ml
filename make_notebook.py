import nbformat as nbf
from pathlib import Path
import shutil

# --- Build the notebook content ---
nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("# Greentree LLM Assessment — End-to-End Notebook\n"
"Loads `.env`, reads dataset (Excel or CSV), runs pipeline steps, and shows quick views."))

cells.append(nbf.v4.new_code_cell("%pip -q install -r requirements.txt ipywidgets python-dotenv"))

cells.append(nbf.v4.new_markdown_cell("## 1) Load data via `.env`"))
cells.append(nbf.v4.new_code_cell("""\
import os, pandas as pd
from dotenv import load_dotenv
load_dotenv(override=True)

data_path = os.getenv("DATA_PATH")
print("DATA_PATH =", data_path)
if not data_path:
    raise ValueError("DATA_PATH not set. Edit your .env.")

if data_path.lower().endswith(".xlsx"):
    df = pd.read_excel(data_path)
else:
    for enc in ["utf-8","utf-8-sig","cp1252","latin-1"]:
        try:
            df = pd.read_csv(data_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
print("Rows x Cols:", df.shape)
df.head()"""))

cells.append(nbf.v4.new_markdown_cell("## 2) Import project modules"))
cells.append(nbf.v4.new_code_cell("""\
from pathlib import Path
import sys
sys.path.append(str(Path("src").resolve()))
sys.path.append(str(Path("..").resolve()))

from src.sentiment_pipeline import run_and_save as run_sentiment
from src.eda import run_and_save as run_eda
from src.monthly_scoring import run_and_save as run_monthly
from src.ranking import run_and_save as run_rank
from src.flight_risk import run_and_save as run_risk
from src.trend_regression import run_and_save as run_trend
try:
    from src.ml_sentiment import train as train_ml, predict as predict_ml
    HAS_ML = True
except Exception as e:
    print("ML module not available:", e)
    HAS_ML = False
"""))

cells.append(nbf.v4.new_markdown_cell("## 3) Run pipeline steps"))
cells.append(nbf.v4.new_code_cell("""\
sent_df, cols = run_sentiment()
sent_df.head()"""))
cells.append(nbf.v4.new_code_cell("""\
_ = run_eda()
print("EDA charts saved to outputs/ and visualizations/")"""))
cells.append(nbf.v4.new_code_cell("""\
monthly, _ = run_monthly()
monthly.head()"""))
cells.append(nbf.v4.new_code_cell("""\
ranking, _ = run_rank()
ranking.head()"""))
cells.append(nbf.v4.new_code_cell("""\
risk, _ = run_risk()
risk.head()"""))
cells.append(nbf.v4.new_code_cell("""\
trend, monthly_for_trend = run_trend()
trend"""))

cells.append(nbf.v4.new_markdown_cell("## 4) Optional ML sentiment (sklearn)"))
cells.append(nbf.v4.new_code_cell("""\
if HAS_ML:
    ml_report = train_ml()
    ml_report
else:
    print("Skip — ml_sentiment module not found.")"""))
cells.append(nbf.v4.new_code_cell("""\
if HAS_ML:
    ml_out, _ = predict_ml()
    ml_out.head()"""))

cells.append(nbf.v4.new_markdown_cell("## 5) Quick interactive exploration"))
cells.append(nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display

df = sent_df.copy()
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year_month"] = df["date"].dt.to_period("M").astype(str)

btns = W.ToggleButtons(options=["Label Distribution","Compound Histogram","Monthly Avg"], value="Label Distribution")
out = W.Output()

def draw(view):
    out.clear_output()
    with out:
        if view == "Label Distribution":
            df["sentiment_label"].value_counts().sort_index().plot(kind="bar", title=view)
            plt.tight_layout(); plt.show()
        elif view == "Compound Histogram":
            df["compound"].plot(kind="hist", bins=30, title=view)
            plt.tight_layout(); plt.show()
        else:
            if "year_month" in df.columns:
                df.groupby("year_month")["compound"].mean().plot(marker="o", title=view)
                plt.tight_layout(); plt.show()
            else:
                print("No date column available.")

draw(btns.value)
btns.observe(lambda ch: draw(ch["new"]), names="value")
display(btns, out)"""))

nb['cells'] = cells

# --- Write notebook to project root ---
root_nb = Path("LLM_Assessment_template.ipynb")
root_nb.write_text(nbf.writes(nb), encoding="utf-8")

# --- Also save into notebooks/ subfolder ---
notebooks_dir = Path("notebooks")
notebooks_dir.mkdir(exist_ok=True)
(shutil.copy2(root_nb, notebooks_dir / root_nb.name))

print(f"Created: {root_nb.resolve()}")
print(f"Copied to: {(notebooks_dir / root_nb.name).resolve()}")
