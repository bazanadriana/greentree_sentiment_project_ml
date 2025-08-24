# Greentree Sentiment Project – Final Report

## 1. Project Overview
This project analyzes employee messages to assess sentiment and engagement using natural language processing (NLP) and basic statistical techniques. It includes:
- Sentiment labeling (Positive, Negative, Neutral)
- Exploratory data analysis (EDA)
- Monthly sentiment scoring
- Employee ranking and flight risk detection
- Predictive modeling for sentiment trends

## 2. Approach and Rationale

### Sentiment Labeling
- **Models used:** We started with TextBlob and later cross-checked results with VADER to reduce bias from a single tool.
- **Thresholds:** We avoided fixed arbitrary thresholds. Instead, we analyzed score distributions and chose cutoffs (e.g., ±0.2) after reviewing sample messages.
- **Validation:** Sampled 20+ messages to confirm labeling accuracy.

### Exploratory Data Analysis (EDA)
- Checked for missing values, data types, and unusual patterns.
- Visualized sentiment counts, trends over time, and employee activity.
- **Interpretation focus:** Each chart is explained in the notebook, highlighting why the pattern matters (e.g., spikes in negative messages could indicate organizational issues).

### Monthly Sentiment Scoring
- Assigned numeric values: Positive = +1, Negative = –1, Neutral = 0.
- Grouped messages by employee and month.
- Reset scores monthly and documented the logic.

### Employee Ranking
- Produced monthly rankings for:
    - Top 3 most positive employees
    - Top 3 most negative employees
- Explained ranking context (e.g., high negative scores could reflect workload stress).

### Flight Risk Identification
- Defined a flight risk as 4+ negative messages in 30 days, based on FAQ guidance.
- Discussed possible false positives (e.g., sarcastic but engaged employees).

### Predictive Modeling
- Implemented a simple linear regression to analyze sentiment trends.
- **Feature selection:** Chose features logically related to messaging behavior (e.g., message count, average length). Avoided irrelevant columns.
- **Metrics:** Used R² and MSE but explained context. A good R² may still hide high errors; interpretation included.

## 3. AI and Analytical Thinking
- **AI as a helper, not a decider:** Outputs were checked manually. 
- Guided the analysis with clear questions (e.g., “Are negative spikes aligned with company events?”).
- Documented limitations (AI models may misread tone, sarcasm).

## 4. Validation and Cross-Checking
- Verified sentiment distributions against sample messages.
- Compared TextBlob vs. VADER scores for consistency.
- Re-ran EDA with filtered datasets to ensure stable findings.

## 5. Key Insights
- Neutral messages dominate, but spikes of negative sentiment occur in some months.
- A few employees repeatedly appear in top negative ranks, potential HR follow-up.
- Sentiment trends show improvements after certain dates (possible organizational changes).

## 6. Deliverables
- **Code:** See `/src` for scripts and `notebooks/LLM_Assessment.ipynb` for step-by-step analysis.
- **Outputs:** Charts saved in `/visualizations`, results in `/outputs`.
- **Environment:** Requirements listed in `requirements.txt`.
- **Documentation:** This `REPORT.md` provides the summary.

---

**Author:** Adriana Bazan  
**Date:** August 2025  

