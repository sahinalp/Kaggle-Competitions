# ⚡ Predicting Electric Vehicle Purchase
<img width="560" height="280" alt="image" src="https://github.com/user-attachments/assets/24a1de55-05b7-4ccb-8bc1-8c4fdec0e84b" />

This repository contains my end-to-end solution for the [Kaggle Playground Series S6E9: Predicting Electric Vehicle Purchase](https://www.kaggle.com/competitions/playground-series-s6e9).

The goal of this competition is to predict whether a customer will purchase an Electric Vehicle based on demographic data, commute distance, charging infrastructure, and psychological factors.

## 📁 Repository Structure

* `01_EDA_and_Feature_Engineering.ipynb`: Comprehensive Exploratory Data Analysis. Covers distribution analysis, categorical target rates, feature interaction heatmaps, and initial feature engineering strategy based on class separation.

* `02_Model_Training_and_Ensembling.ipynb`: The main modeling pipeline. Implements advanced feature engineering, target encoding, model training, and ensembling.
* `utils.py`: Modularized Python script containing the cross-validation loops, PyTorch Tabular Encoder, model definitions (LightGBM, XGBoost, CatBoost, TabM, RealMLP, Logistic Regression), and Scipy rank blending logic.
* `realmlp.py`: A from-scratch PyTorch implementation of the RealMLP-TD architecture. Implements a custom ensemble MLP with Periodic Basis Linear Decomposition (PBLD) numerical embeddings, NTP linear layers with per-parameter-group learning rates, EMA weight averaging, and smooth cross-entropy loss with label smoothing and focal loss support.

## 🚀 Key Strategies & Highlights

### 1. Feature Engineering
Through the EDA, I identified critical non-linear signals and interactions:
- **The "Millionaire Cliff" & Dead Zones**: Identified sharp deterministic boundaries in the `Annual_Income_USD` feature.
- **Interaction Effects**: Engineered high-impact features such as `Subsidy_Available * Environmental_Concern_Level`. The data showed that without a subsidy, purchase rates are near-zero, but with a subsidy and high environmental concern, they jump to ~69%.
- **Ratio & Polynomial Features**: Created features like `Income_per_Car`, `Charging_Ratio`, and composite "Readiness" scores to help tree-based models split more effectively.

### 2. Fold-Safe Target Encoding
Tree-based models often struggle with high-cardinality categoricals. To address this without introducing data leakage, I implemented **Fold-Safe Target Encoding** inside the cross-validation loop. 
- Applied multiple smoothing parameters (`auto`, `10`, `100`) to give the models a multi-scale view of category distributions.

### 3. Contrastive Learning Embeddings (InfoNCE)
To capture deeper representations of the tabular data, I implemented a custom PyTorch `TabularEncoder`. 
- Trained using an **InfoNCE contrastive loss function** on masked views of the data. 
- The resulting L2-normalized embeddings were attached as additional numeric features for the gradient boosting models.

### 4. Multi-Model Rank Blending
Instead of relying on a single model, I trained a diverse ensemble:
- **Models**: LightGBM, XGBoost, CatBoost, TabM, and RealMLP-TD.
- **Variance Reduction**: Applied **Seed Averaging** (seeds 42, 101, 2026) for each model to smooth out predictions.
- **Ensembling**: Used `scipy.optimize.minimize` (Nelder-Mead) to find the optimal weights for a **Rank-Based Blend**. Rank blending proved much more robust to probability calibration differences between algorithms than simple averaging.

### 5. Custom RealMLP-TD Implementation (`realmlp.py`)
Built a from-scratch PyTorch implementation of the RealMLP-TD architecture:
- **PBLD Embeddings**: Periodic Basis Linear Decomposition for richer numerical feature representations using cosine projections.
- **NTP Linear Layers**: Normalized-then-projected linear layers (`einsum` based) for a stable ensemble of `n_ens` models trained in parallel.
- **Per-group LR & WD**: Separate learning rate and weight decay schedules for scale layers, PBLD parameters, first layer, other weights, and biases.
- **EMA Averaging**: Exponential Moving Average of model weights for better generalization.
- **Flexible Losses**: Smooth cross-entropy with label smoothing, focal loss weighting, prior logit bias, and class/sample weighting.

## 📈 Results

| Metric | Score |
|--------|-------|
| **CV AUC** (5-Fold StratifiedKFold) | **0.94608** |
| **Public Leaderboard AUC** | **0.94615** |
| **Private Leaderboard AUC** | **0.94514** |

The CV score closely tracked the public LB score, confirming that train/test distributions are well-aligned (verified via KS-test in the EDA notebook) and that StratifiedKFold is a reliable validation strategy for this dataset.

**Optimal Rank-Blend Weights** (found via Nelder-Mead optimization):

| Model | AUC Score | Weight |
|-------|--------|--------|
| XGBoost | 0.945987 | 49.96% |
| TabM | 0.945974 | 47.35% |
| CatBoost | 0.945725 | 2.60% |
| LightGBM | 0.945903 | 0.08% |

The optimizer heavily favored XGBoost and TabM, suggesting these two models learned complementary signal. The near-zero weight on LightGBM indicates that, in this configuration, XGBoost captures similar gradient boosting signal but with less noise.
