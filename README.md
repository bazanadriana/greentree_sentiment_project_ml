<<<<<<< HEAD
# Greentree Sentiment Project (LLM Practical)

This repository contains the implementation for the Employee Sentiment Analysis practical assessment. It performs:

- **Sentiment Labeling** (Positive, Negative, Neutral)
- **Exploratory Data Analysis (EDA)** and visualizations
- **Monthly sentiment scoring**
- **Employee ranking**
- **Flight risk identification**
- **Predictive modeling (linear regression)**

---

## Folder Structure

greentree_sentiment_project_ml/
│
├── data/ # Place your dataset here (CSV or XLSX)
│ └── surveys.csv # Example dataset
├── notebooks/ # Interactive notebooks
│ └── LLM_Assessment.ipynb
├── outputs/ # Processed outputs (parquet, charts, reports)
├── src/ # Source code
│ ├── main.py # Entry point for running the full pipeline
│ ├── sentiment_pipeline.py
│ ├── eda.py
│ ├── monthly_scoring.py
│ ├── ranking.py
│ ├── flight_risk.py
│ ├── trend_regression.py
│ └── utils.py
├── visualizations/ # Charts and plots
├── .env # Environment variables (update DATA_PATH, etc.)
├── README.md # Setup and instructions
├── REPORT.md # Final report with insights
└── requirements.txt # Dependencies


---

## Requirements

- Python 3.9+ (tested with 3.13)
- Virtual environment recommended

**Key Libraries:**
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `pyarrow` or `fastparquet`
- `jupyterlab`, `ipywidgets`
- `python-dotenv`

Install all with:
```bash
pip install -r requirements.txt


Setup

Clone the repo:

git clone <your_repo_url>
cd greentree_sentiment_project_ml

Create and activate virtual environment:

python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

Install dependencies:

pip install --upgrade pip
pip install -r requirements.txt

Usage

Run the full pipeline:

python3 -m src.main --all

Open the interactive notebook:

jupyter notebook # Then open notebooks/LLM_Assessment.ipynb

View Outputs

Processed data in outputs/

Charts in visualizations/
=======
# greentree_sentiment_project_ml
>>>>>>> 4b8129fb885f3acc927553e82ec8b888d83168a4
