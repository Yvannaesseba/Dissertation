# Fair Credit Assessment

## Evaluating Synthetic Data for Fair and Transparent Credit Scoring

**MSc Big Data Analytics Dissertation | Sheffield Hallam University | 2026**

This project investigates whether synthetic credit data can substitute for real training data without significantly changing model performance, fairness or explainability.

Using two public credit-risk datasets, I generated synthetic training data with **CTGAN** and compared models trained on real and synthetic data. Both were evaluated on the same held-out real data using a **Train-Synthetic-Test-Real (TSTR)** framework.

The evaluation focuses on:

- **Predictive performance** : how well the models identify credit risk.
- **Fairness** : how outcomes and error rates differ across age groups.
- **Explainability** : whether real and synthetic-trained models rely on similar features.

---

## The Problem

Synthetic data can provide an alternative where access to real data is restricted by privacy or data-sharing constraints. However, similar predictive performance does not necessarily mean synthetic data is a reliable substitute.

A model trained on synthetic data may perform reasonably well while behaving differently across demographic groups or relying on different features when making predictions.

This project therefore evaluates synthetic data across **performance, fairness and explainability**, rather than predictive performance alone.

---

## Research Question

> **Under what conditions, if any, can CTGAN-generated synthetic tabular data credibly substitute for real credit data in the development of credit scoring models while preserving performance, fairness and explainability?**

---

## Approach

The project uses a **Train-Synthetic-Test-Real (TSTR)** framework.

### TSTR Workflow

**Real training data → Real-trained models → Held-out real test data**

**Real training data → CTGAN → Synthetic training data → Synthetic-trained models → Held-out real test data**

Both model groups are then compared across predictive performance, fairness and explainability.

The experimental process was:

1. Split each real dataset into training, validation and test partitions.
2. Train CTGAN on the real training partition.
3. Generate synthetic training data.
4. Train equivalent models on real and synthetic training data.
5. Evaluate both model groups on the same held-out real test data.

This helps isolate the effect of replacing real training data with synthetic data.

---

## Datasets

Two public credit-risk datasets were selected to represent different levels of complexity.

| Dataset | Initial Size | Final Modelling Size | Characteristics |
|---|---:|---:|---|
| Give Me Some Credit (GMSC) | 150,000 | 150,000 | Primarily numerical, lower complexity |
| Home Credit Default Risk | 307,511 | 184,506 | Mixed data types, higher complexity |

GMSC provided the lower-complexity case, while Home Credit included more complex applicant and bureau information. This allowed the study to examine how dataset complexity affected synthetic-data quality and downstream model behaviour.

> **Data availability:** The original datasets are not included in this repository and can be obtained from their respective Kaggle sources.

---

## Key Findings

### 1. Predictive Performance

Synthetic-trained models retained more predictive utility on GMSC than on the more complex Home Credit dataset.

| Model | GMSC Real AUC | GMSC Synthetic AUC | HC Real AUC | HC Synthetic AUC |
|---|---:|---:|---:|---:|
| XGBoost | 0.862 | 0.835 | 0.761 | 0.696 |
| Logistic Regression | 0.861 | 0.850 | 0.750 | 0.710 |
| EBM | 0.869 | 0.838 | 0.768 | 0.690 |

AUC retention ranged from **96.5–98.7% on GMSC**, compared with **89.8–94.7% on Home Credit**.

### 2. Fairness

Fairness was evaluated across three age groups using demographic parity and equal opportunity.

| Dataset | DP Gap: Real → Synthetic | TPR Gap: Real → Synthetic |
|---|---:|---:|
| GMSC | 0.292 → 0.341 | 0.231 → 0.275 |
| Home Credit | 0.301 → 0.334 | 0.377 → 0.409 |

Both gaps increased under synthetic training, showing that relatively strong predictive utility did not guarantee preservation of fairness behaviour.

### 3. Explainability

SHAP was used to compare global feature-importance rankings between real- and synthetic-trained XGBoost models.

| Dataset | SHAP Rank Correlation (ρ) |
|---|---:|
| GMSC | 0.806 |
| Home Credit | 0.299 |

GMSC showed relatively strong alignment, while Home Credit showed much weaker agreement. This indicates that models can retain some predictive performance while changing which features influence their predictions.

### Overall Finding

Synthetic data was **not an equally reliable substitute across datasets or evaluation dimensions**.

CTGAN preserved predictive utility more successfully on the simpler GMSC dataset, while the more complex Home Credit dataset showed greater degradation. Performance, fairness and explainability did not always change in the same way after synthetic substitution.

---

## Methods

### Synthetic Data Generation
- CTGAN

### Models
- Logistic Regression
- XGBoost
- Explainable Boosting Machine (EBM)

### Evaluation

**Predictive performance**
- AUC-ROC
- Precision
- Recall
- F1-score
- Average Precision

**Fairness**
- Demographic parity
- Equal opportunity
- Age groups: 18–35, 36–55 and 56+

**Explainability**
- SHAP
- Spearman rank correlation for global feature-importance alignment

---

## Limitations

- The study used two public credit datasets, which may not represent the full complexity of proprietary financial data.
- Available protected characteristics limited the scope of the fairness analysis, so age groups were used for the main comparison.
- CTGAN was the only synthetic-data generator evaluated, so the findings should not be generalised to all synthetic-data methods.
- The study evaluates synthetic data for model development and comparison; it does not establish that the resulting models are suitable for real-world lending decisions.

---

## Repository Structure

```text
fair-credit-assessment/
├── notebooks/
│   ├── 01_eda_gmsc.py
│   ├── 02_ctgan_gmsc.py
│   ├── 03_models_gmsc.py
│   ├── 04_validation_gmsc.py
│   ├── 05_eda_hc.py
│   ├── 06_feature_selection_hc.py
│   ├── 07_ctgan_hc.py
│   ├── 08_models_hc.py
│   ├── 09_tstr_hc.py
│   ├── 10_fairness_hc.py
│   └── 11_explainability_hc.py
├── src/
│   ├── fairness_metrics_gmsc.py
│   ├── shap_utils.py
│   ├── statistical_validation_gmsc.py
│   └── statistical_validation_hc.py
├── outputs/
│   ├── roc_curves/
│   ├── shap/
│   ├── fairness/
│   └── metrics/
├── data/
│   └── README.md
├── requirements.txt
└── README.md
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/Yvannaesseba/fair-credit-assessment.git
cd fair-credit-assessment
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The original datasets are not stored in the repository. See `data/README.md` for data setup information.

---

## Author

**Emmanuelle Yvanna Esseba Ayangma**  
MSc Big Data Analytics (Distinction), Sheffield Hallam University