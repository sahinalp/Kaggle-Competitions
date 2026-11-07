# ✈️ Predicting Airline Passenger Satisfaction
<img width="560" height="280" alt="image" src="https://github.com/user-attachments/assets/cb12fe5f-d6c4-47d6-b1df-43490059f0af" />

This repository contains my end-to-end solution for the [Kaggle Playground Series S6E10: Predicting Airline Satisfaction](https://www.kaggle.com/competitions/playground-series-s6e10).

The goal of this competition is to predict whether an airline passenger is satisfied or neutral/dissatisfied based on passenger demographics, flight details, delay metrics, and 14 distinct service quality ratings.

## 📁 Repository Structure

* `01_EDA_and_Feature_Engineering.ipynb`: Comprehensive Exploratory Data Analysis. Covers distribution analysis, categorical target rates, feature interaction heatmaps, class separation analysis, and train vs. test vs. original dataset consistency checks.
* `02_Model_Training_and_Ensembling.ipynb`: The main modeling pipeline. Implements domain-driven feature engineering, group rating aggregations, fold-safe target encoding, 5-fold Stratified CV across diverse models with multi-seed averaging, and rank blending.
* `utils.py`: Modularized Python script containing cross-validation loops, PyTorch Tabular Encoder with InfoNCE contrastive learning, model definitions (LightGBM, XGBoost, CatBoost, TabM, RealMLP-TD, Logistic Regression), and Scipy rank blending optimization.
* `realmlp.py`: From-scratch PyTorch implementation of the RealMLP-TD architecture. Implements an ensemble MLP with Periodic Basis Linear Decomposition (PBLD) numerical embeddings, Normalized-Then-Projected (NTP) linear layers with per-parameter-group learning rates, Exponential Moving Average (EMA) weight averaging, and smooth cross-entropy loss with label smoothing and focal loss support.

## 🚀 Key Strategies & Highlights

### 1. Feature Engineering
Through thorough EDA and class separation analysis, I engineered high-impact features specifically targeting non-linear satisfaction triggers:
- **Online Boarding as Primary Satisfaction Anchor**: EDA revealed Online Boarding as the single most predictive feature in the raw dataset (+0.56 Pearson correlation, +53.4% Δmean). Passengers with ratings 1–3 had low satisfaction rates (8.5%–12.8%), while rating 4 jumped to 64.2% and rating 5 to 87.5%. Created `boarding_high` ($\ge 4$) and `boarding_unrated` ($= 0$, as 0 represents unrated rather than low satisfaction).
- **Service Quality Domain Aggregations**: The 14 survey ratings were grouped into functional service dimensions to extract clearer composite signals:
  - `digital_rating_mean`: Wifi, Ease of Online booking, Online boarding
  - `comfort_rating_mean`: Food & drink, Seat comfort, Inflight entertainment, Leg room service, Cleanliness
  - `onboard_rating_mean`: On-board service, Baggage handling, Checkin service
  - `airport_rating_mean`: Departure/Arrival time convenience, Gate location
  - Global rating statistics: `rating_mean`, `rating_count`, `rating_unrated_count`, `rating_high_count` ($\ge 4$), and `rating_low_count` ($\le 2$).
- **High-Impact Domain Interactions**:
  - `Class` and `Type of Travel` are decisive factors (Business class = 72.5% satisfaction vs Eco = 16.8%; Business travel = 58.4% vs Personal = 9.7%).
  - Engineered interaction terms: `boarding_high_business_travel` (proved to be the **#1 feature in model importance**, scoring >26M), `boarding_high_business_class` (#3 in importance), `business_class_business_travel`, `personal_travel_economy_class`, and `boarding_high_loyal_customer`.
  - `log_distance_x_boarding_high`: Captures interaction showing passenger tolerance shifts on long flights depending on onboarding ease.
- **Delay Dynamics & Imputation**:
  - `Departure Delay in Minutes` and `Arrival Delay in Minutes` exhibit heavy right-skew with extreme outliers.
  - Applied consistent median imputation for missing arrival delays across train, test, and supplemental datasets.
  - Engineered `total_delay_minutes`, `log_total_delay`, `delay_difference` (arrival delay minus departure delay to capture flight recovery or worsening in-air), and binary `any_delay`.

### 2. Fold-Safe Target Encoding
Categorical features (`Customer Type`, `Class`, `Type of Travel`) were target-encoded inside the 5-fold cross-validation loop to prevent data leakage:
- Evaluated multiple smoothing parameters (`auto`, `10`, `100`) to provide models with multi-scale views of category target distributions.
- Feature importance analysis confirmed `Customer Type_TE` among the top predictors.

### 3. Diverse Model Zoo (Tree-Based + Deep Tabular Architectures)
Rather than relying solely on a single model family, I trained a diverse ensemble across fundamentally different inductive biases:
- **Gradient Boosted Decision Trees**:
  - **LightGBM**: Tuned with deep leaf-wise splits (`num_leaves=68`, `max_depth=9`, `learning_rate=0.005`).
  - **XGBoost**: Histogram-based boosting capturing complementary split boundaries.
  - **CatBoost**: Symmetric oblivious trees mitigating overfitting on discrete rating tiers.
- **Deep Tabular Architectures**:
  - **TabM**: Tabular Model with parameter-efficient multi-head / ensemble representations.
  - **RealMLP-TD**: Custom PyTorch neural network specifically optimized for tabular data with PBLD numerical embeddings, NTP ensemble layers, and EMA weight averaging.
  - *(Exploratory)* **TabPFN**: Zero-shot tabular transformer producing highly complementary probabilistic rankings.

### 4. Multi-Seed Variance Reduction
Applied seed averaging across seeds `42`, `101`, and `2026` for each model architecture. This reduced prediction variance, smoothed decision boundaries, and improved validation stability.

### 5. Multi-Model Rank Blending
Instead of probability averaging (which suffers from calibration mismatches across GBDTs and neural networks), predictions were converted to percentile ranks (`scipy.stats.rankdata`):
- Optimization: Used `scipy.optimize.minimize` (Nelder-Mead) on OOF predictions to find the optimal ensemble weights that directly maximize ROC AUC.
- Result: Produced a well-calibrated, robust final submission.

### 6. Custom RealMLP-TD Implementation (`realmlp.py`)
Built a from-scratch PyTorch implementation of the RealMLP-TD architecture:
- **PBLD Embeddings**: Periodic Basis Linear Decomposition for richer numerical feature representations using cosine projections.
- **NTP Linear Layers**: Normalized-then-projected linear layers (`einsum` based) for a stable ensemble of `n_ens` models trained in parallel.
- **Per-group LR & WD**: Separate learning rate and weight decay schedules for scale layers, PBLD parameters, first layer, other weights, and biases.
- **EMA Averaging**: Exponential Moving Average of model weights for better generalization.
- **Flexible Losses**: Smooth cross-entropy with label smoothing, focal loss weighting, prior logit bias, and class/sample weighting.

## 📈 Results

| Metric | Score |
|---|---|
| **CV AUC (5-Model Rank Blend)** (5-Fold StratifiedKFold) | **0.95937** |
| **CV AUC (With TabPFN Integration)** | **0.96111** |
| **Public Leaderboard AUC** | *0.96048* |
| **Private Leaderboard AUC** | *0.96155* |

The StratifiedKFold cross-validation strategy aligned closely with test distribution characteristics verified via KS-test in the EDA notebook.

### Individual Model Performance (5-Fold CV OOF AUC)

| Model | OOF AUC | Configuration / Notes |
|---|---|---|
| **LightGBM** | 0.95898 | 3-Seed Averaged (42, 101, 2026) |
| **TabM** | 0.95898 | Tabular Multi-Head Embedding Model |
| **XGBoost** | 0.95881 | 3-Seed Averaged (42, 101, 2026) |
| **RealMLP-TD** | 0.95878 | Custom PyTorch PBLD + NTP Ensemble |
| **CatBoost** | 0.95858 | 3-Seed Averaged (42, 101, 2026) |

### Optimal Rank-Blend Weights (Nelder-Mead Optimization)

**5-Model GBDT + Tabular NN Blend:**

| Model | AUC Score | Optimal Weight |
|---|---|---|
| **TabM** | 0.95898 | 35.56% |
| **LightGBM** | 0.95898 | 25.65% |
| **XGBoost** | 0.95881 | 25.42% |
| **RealMLP-TD** | 0.95878 | 13.37% |
| **CatBoost** | 0.95858 | 0.00% |

> **Ensemble Insight**: The optimizer distributed weights evenly across TabM, LightGBM, and XGBoost, with meaningful contribution from RealMLP-TD. The diversity between neural representations (TabM, RealMLP) and gradient boosted trees (LightGBM, XGBoost) proved instrumental in pushing the CV score from ~0.9589 up to 0.95937. (When experimenting with zero-shot TabPFN predictions, the blend achieved **0.96111** OOF AUC).
