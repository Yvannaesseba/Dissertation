# Fair Credit Assessment

### Evaluating Synthetic Data for Fair and Transparent Credit Scoring

**MSc Big Data Analytics Dissertation | Sheffield Hallam University | 2026**

This project investigates whether synthetic credit data can be used as a substitute for real training data without significantly changing how credit scoring models perform, how fairly they behave across demographic groups, or how their predictions are explained.

Using two public credit datasets, I generated synthetic training data with CTGAN and compared models trained on real and synthetic data. Both sets of models were evaluated on held-out real data using a Train-Synthetic-Test-Real (TSTR) framework.

The evaluation focuses on three areas:

- **Predictive performance** — how well the models identify credit risk.
- **Fairness** — whether model outcomes remain consistent across age groups.
- **Explainability** — whether models trained on real and synthetic data rely on similar features when making predictions.

## The Problem

Synthetic data can provide an alternative to using real data in situations where privacy, access or data-sharing constraints are important. However, a synthetic dataset producing similar model performance does not necessarily mean that it is a reliable substitute for the original data.

A model trained on synthetic data may still behave differently across demographic groups or rely on different features when making predictions. This project therefore evaluates synthetic data beyond predictive performance alone by examining fairness and explainability alongside model utility.

## Research Question

> **Under what conditions, if any, can CTGAN-generated synthetic tabular data credibly substitute for real credit data in the development of credit scoring models while preserving performance, fairness and explainability?**