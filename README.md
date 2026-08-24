# Used Car Price Prediction

Competition project for predicting used-car prices from structured attributes and vehicle-title text. The solution combines tabular boosting, linear models, text features, target statistics and comparable-vehicle retrieval.

## Results

| Evaluation | MAPE |
|---|---:|
| Final leaderboard submission | **12.66%** |
| Nested out-of-fold ensemble after conditional calibration | **11.47%** |
| Best single CatBoost holdout model | **12.13%** |
| Ridge baseline holdout | **13.82%** |

The final ensemble improved on every standalone baseline and used out-of-fold predictions for model selection and calibration.

## Key decisions

- Removed the feature `Предложение`: it was present in every training row, absent from every test row and had a 0.9938 correlation with price.
- Used a log-transformed target and MAPE-oriented evaluation.
- Validated models with fold-safe preprocessing and out-of-fold predictions.
- Combined complementary signals instead of relying on one high-capacity model.
- Audited train/test distribution shift before final submission.

## Final ensemble

The final recipe combines:

- CatBoost models with title and hierarchy features;
- Ridge with one-hot encoded tabular features;
- fold-safe target statistics;
- TF-IDF text regression;
- nearest-comparable vehicle retrieval;
- conditional calibration based on model disagreement.

## Repository guide

- [`shift_ml/notebooks/`](shift_ml/notebooks) — EDA, validation, ablation and ensemble experiments;
- [`shift_ml/reports/`](shift_ml/reports) — experiment tables and the final ensemble recipe;
- [`shift_ml/src/`](shift_ml/src) — reusable project code;
- [`shift_ml/submission/`](shift_ml/submission) — generated competition submissions;
- [`shift_ml/Как мы пришли к submission v10.pdf`](shift_ml/%D0%9A%D0%B0%D0%BA%20%D0%BC%D1%8B%20%D0%BF%D1%80%D0%B8%D1%88%D0%BB%D0%B8%20%D0%BA%20submission%20v10.pdf) — detailed experiment narrative in Russian.

## Technology

Python · pandas · scikit-learn · CatBoost · TF-IDF · model ensembling · out-of-fold validation

## What this project demonstrates

This repository shows a full competition workflow: leakage detection, systematic experimentation, reproducible validation, feature ablation, model diversity, ensemble construction and final submission engineering.