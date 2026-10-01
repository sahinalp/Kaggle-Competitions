# 🏆 Kaggle Competition Portfolio

[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/sahinalpakosman)

A collection of my solutions to Kaggle Playground Series competitions. Each folder contains a full end-to-end pipeline: exploratory data analysis, feature engineering, model training, and ensembling.

---

## 📂 Competitions

### [S6E9 — Predicting Electric Vehicle Purchase](https://www.kaggle.com/competitions/playground-series-s6e9/)
> **Binary Classification** | Tabular Data

Predict whether a customer will purchase an Electric Vehicle based on demographic, behavioral, and infrastructure features.

| Metric | Score |
|--------|-------|
| CV AUC | 0.94608 |
| Public LB | 0.94615 |
| Private LB | 0.94514 |

**Key Techniques:** Fold-safe Triple Target Encoding · Contrastive Learning Embeddings (InfoNCE) · Custom RealMLP-TD from scratch · XGBoost + TabM Rank Blend

---

## 🛠️ Common Stack
- **Languages:** Python
- **ML Libraries:** LightGBM, XGBoost, CatBoost, scikit-learn
- **Deep Learning:** PyTorch (custom architectures)
- **Neural Tabular Models:** TabM, RealMLP-TD (custom implementation)
- **Workflow:** Stratified K-Fold CV · Seed Averaging · Scipy Rank Blending
