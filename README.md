
---

# Lead Conversion Predictor

An end-to-end machine learning project that predicts whether a sales lead will convert into a paying customer. Built to help sales teams prioritize their outreach by scoring leads based on behavioral and demographic signals.

---

## The Problem

Sales organizations often deal with a high volume of inbound leads coming from multiple channels—landing pages, digital advertisements, and direct interactions. However, not every lead has the same intent or likelihood of purchasing. Without a data-driven way to rank these prospects, sales representatives often spend disproportionate time on cold leads while high-value opportunities may go cold.

This project builds a binary classifier that assigns a conversion probability to each lead, using features like engagement time, lead source, last recorded activity, and demographic profiles to optimize the sales funnel.

---

## Dataset

The dataset contains **9,240 leads** with 37 raw features, reduced to 9 meaningful predictors after cleaning. The target variable is `Converted` (1 = converted, 0 = not converted), with an approximate 39/61 class split.

---

## Data Cleaning

The raw data had several issues that needed careful handling:

- **Zero-variance and near-constant columns** (13 features) were dropped entirely — they carried no signal.
- **High-null columns** like `Lead Quality`, `Lead Profile`, and `Asymmetrique Index` were removed rather than imputed, since they were either redundant with other features or missing in over 45% of rows.
- **Placeholder values** (`"Select"` from dropdown defaults) were converted to NaN and handled as missing.
- **Numerical outliers**: `TotalVisits` was capped at 21 ("heavy visitor" flag) and `Page Views Per Visit` was rounded and capped at 11 to reduce noise.
- **Sparse categories** across Lead Source, Last Activity, Tags, and Occupation were consolidated into `"Other"` to prevent the model from fitting on noise.
- **Country** was collapsed into a binary `is_india` flag since 96% of known values were India anyway.

---

## Exploratory Data Analysis

A few patterns stood out clearly:

**Time on site is the strongest numeric signal.** Converted leads spent an average of 738 seconds on the website versus 330 seconds for non-converted leads — more than double. Total visits and page views showed weaker separation.

**Lead origin matters a lot.** Leads from the "Lead Add Form" converted at ~92%, compared to ~36% from Landing Page Submissions and ~31% from the API. This suggests form-submitted leads have much higher intent.

**Last activity is highly predictive.** Leads whose last recorded activity was "SMS Sent" converted at a high rate, while those whose last activity was "Olark Chat Conversation" or "Email Bounced" converted much less frequently.

**Working Professionals convert at the highest rate** (~91%) among known occupations, while "Unknown" occupation leads converted at only ~14% — indicating that missing occupation data itself is a negative signal.

**Cramér's V association analysis** confirmed that `Lead Source`, `Lead Origin`, `Last Activity`, and `Tags` have the highest categorical associations with the target.

---

## Models

Four models were trained and evaluated using 5-fold cross-validation on the training set (80/20 stratified split):

**Logistic Regression** served as the baseline. Features were log-transformed and scaled, with `class_weight='balanced'` to handle the class imbalance. Straightforward and interpretable, but limited in capturing non-linear relationships.

**Random Forest** (100 estimators, balanced class weights) improved over logistic regression on all metrics and produced well-separated probability distributions between converted and non-converted leads.

**XGBoost** (100 estimators, logloss objective) was the strongest individual model across every metric.

**Voting Ensemble** combined all three using soft voting. Despite the intuition that ensembling should help, the ensemble's test performance came in slightly below XGBoost, suggesting XGBoost's predictions were already well-calibrated enough that averaging in the weaker models diluted the signal.

---

## Results (Test Set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 78.84% | 70.14% | 78.51% | 74.09% | 0.904 |
| Random Forest | 80.57% | 74.01% | 76.40% | 75.19% | 0.904 |
| **XGBoost** | **82.79%** | **77.21%** | **78.51%** | **77.86%** | **0.904** |
| Voting Ensemble | 81.22% | 74.63% | 77.67% | 76.12% | 0.904 |

**XGBoost is the best model.** It leads on accuracy, precision, and F1 while matching the others on AUC. The probability gap between converted (mean: 0.73) and non-converted (mean: 0.22) leads is clear and usable for threshold-based prioritization.

---

## How to Run Locally

**Prerequisites:** Python 3.9+

```bash
# 1. Clone the repo
git clone https://github.com/your-username/lead-conversion-predictor.git
cd lead-conversion-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run notebooks in order
jupyter notebook
```

Open and run the notebooks sequentially:

```
01_data_cleaning.ipynb   → cleans raw data, saves clean_df.csv
02_EDA.ipynb             → exploratory analysis and visualizations
03_Base_model.ipynb      → logistic regression baseline
04_rf_model.ipynb        → random forest
05_XGBoost_model.ipynb   → XGBoost (best model)
06_Ensemble_model.ipynb  → voting ensemble
Test_.ipynb              → final evaluation on held-out test set
```

---

## Tech Stack

- **Python**
- **pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing pipelines, Logistic Regression, Random Forest, cross-validation, metrics
- **XGBoost** — gradient boosting classifier
- **matplotlib / seaborn** — visualization
- **scipy** — Cramér's V association analysis
- **joblib** — model serialization

---

## Project Structure

```
lead-conversion-predictor/
├── Data/
│   ├── data.csv                # raw dataset
│   ├── clean_df.csv            # after cleaning
│   ├── finalized_feas.csv      # final feature set used for modeling
│   ├── x_test.csv
│   └── y_test.csv
├── Models/
│   ├── lr_model.pkl
│   ├── rf_model.pkl
│   ├── xgb_model.pkl
│   └── ensemble_model.pkl
├── 01_data_cleaning.ipynb
├── 02_EDA.ipynb
├── 03_Base_model.ipynb
├── 04_rf_model.ipynb
├── 05_XGBoost_model.ipynb
├── 06_Ensemble_model.ipynb
├── Test_.ipynb
└── README.md
```
