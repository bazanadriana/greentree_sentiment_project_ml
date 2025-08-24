# Employee Sentiment Analysis - Greentree LLM Practical

**Author:** Adriana Bazan  
**Date:** 08/23/2025  

---

## 1. Project Overview

This project analyzes an unlabeled dataset of employee messages to assess sentiment and engagement. Using Python and data science techniques, we:

- Labeled each message as Positive, Negative, or Neutral.
- Explored data trends (EDA).
- Calculated monthly sentiment scores.
- Ranked employees by sentiment.
- Identified flight risk employees.
- Modeled sentiment trends using linear regression.

**Tech Stack:** Python, Pandas, Matplotlib/Seaborn, scikit-learn (optional ML), Jupyter Notebook.

**Data Path:** `test.xlsx`  
**Key Columns:** `body`, `date`, `from`

---

## 2. Data & Preprocessing

- **Source:** Provided Greentree dataset.
- **Columns used:** 
  - `body` – message content
  - `date` – message timestamp
  - `from` – employee identifier
- **Cleaning steps:** 
  - Removed empty or corrupted rows.
  - Parsed dates to `datetime`.
  - Handled missing values (dropped or filled).
- **Encoding issues:** The Excel version (`test.xlsx`) caused UTF-8 errors; CSV was used instead.

---

## 3. Sentiment Labeling

**Approach:**  
Used <rule-based keywords / pre-trained model / ML pipeline> to classify each message.

**Classes:**
- Positive (+1)
- Negative (-1)
- Neutral (0)

**Sample Distribution:**
- Positive: <count>
- Negative: <count>
- Neutral: <count>


---

## 4. Exploratory Data Analysis (EDA)

- Total messages: `< >`
- Date range: `<start>` to `<end>`
- Most active employees: `< >`

**Key Visualizations (saved in `visualizations/`):**
- Sentiment distribution chart.
- Messages per month.
- Top senders.

Example:  
![Sentiment Trend](visualizations/monthly_sentiment.png)

---

## 5. Monthly Sentiment Scoring

**Method:**  
Scores computed by summing message sentiments per employee per month.  
Formula: **Positive=+1, Neutral=0, Negative=-1**

**Example Output:**

| Employee       | Month   | Score |
|----------------|---------|-------|
| sally.beck     | 2020-05 | +3    |
| eric.bass      | 2020-07 | -2    |

---

## 6. Employee Ranking

**Top 3 Positive Employees (per month):**
1. `<name>` - score `< >`
2. `<name>` - score `< >`
3. `<name>` - score `< >`

**Top 3 Negative Employees (per month):**
1. `<name>` - score `< >`
2. `<name>` - score `< >`
3. `<name>` - score `< >`

---

## 7. Flight Risk Identification

**Criteria:** Any employee with **≥4 negative messages within 30 days** flagged.

**Flagged Employees:**
- `<employee1>`
- `<employee2>`

---

## 8. Predictive Modeling

**Goal:** Analyze sentiment trends over time.

- Built a linear regression model to predict monthly average sentiment.
- Independent variables: message frequency, sentiment counts, word counts.
- Output: `<brief findings, e.g., slight downward trend in mid-2010>`.

Example Chart:  
![Trend Regression](visualizations/sentiment_trend.png)

---

## 9. Insights & Recommendations

- Periods of increased negativity may align with specific events or departments.
- Few employees dominate negative messages; targeted follow-up may reduce risk.
- Consistent positive contributors could be leveraged for engagement programs.

---

## 10. How to Run

```bash
git clone <repo>
cd greentree_sentiment_project_ml
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m src.main --all
jupyter notebook notebooks/LLM_Assessment.ipynb

