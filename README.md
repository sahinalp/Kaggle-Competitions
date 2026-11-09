# 🏆 Kaggle Competition Portfolio

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/sahinalpakosman)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.2+-yellow.svg)](https://catboost.ai/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competitions-20BEFF.svg)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A collection of my solutions to Kaggle Playground Series competitions. Each folder contains a full end-to-end pipeline: exploratory data analysis, feature engineering, model training, and ensembling.

---

## 📂 Competitions

| Season / Competition | Description | Task Type | Evaluation Metric | Best CV | Public LB | Private LB | Key Models & Highlights | Code Link |
|---|---|---|---|---|---|---|---|---|
| **[Playground S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10)** | <details><summary><b>View Description</b></summary><br>Predict whether an airline passenger is satisfied or neutral/dissatisfied based on flight details, demographic data, delay metrics, and 14 service quality ratings.</details> | Binary Classification | **ROC AUC** | **0.96111** | **0.96048** | **0.96155** | <details><summary><b>View Highlights</b></summary><br>Online Boarding interactions, grouped service rating aggregations, fold-safe target encoding, TabM, RealMLP-TD, Nelder-Mead rank blend</details> | [View Folder](./Playground_S6E10_Airline_Passenger_Satisfaction) |
| **[Playground S6E9: Predicting Electric Vehicle Purchase](https://www.kaggle.com/competitions/playground-series-s6e9)** | <details><summary><b>View Description</b></summary><br>Predict whether a customer will purchase an Electric Vehicle based on demographic, behavioral, and infrastructure features.</details> | Binary Classification | **ROC AUC** | **0.94608** | **0.94615** | **0.94514** | <details><summary><b>View Highlights</b></summary><br>"Millionaire Cliff" thresholding, Subsidy x Concern interactions, InfoNCE contrastive embeddings, RealMLP-TD, TabM, Rank blend</details> | [View Folder](./Playground_S6E9_EV_Purchase) |

---

## 🛠️ Common Stack
- **Languages:** Python
- **ML Libraries:** LightGBM, XGBoost, CatBoost, scikit-learn
- **Deep Learning:** PyTorch (custom architectures)
- **Neural Tabular Models:** TabM, RealMLP-TD (custom implementation)
- **Workflow:** Stratified K-Fold CV · Seed Averaging · Scipy Rank Blending
