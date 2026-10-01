import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
from scipy.stats import rankdata
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pytabkit import TabM_D_Classifier
from realmlp import RealMLP_TD_Classifier, seed_everything
from sklearn.preprocessing import TargetEncoder
import gc
from tqdm import tqdm

class TabularEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1) # L2 normalized embeddings

def info_nce_loss(features, temperature=0.1):
    """
    Features: (2 * batch_size, embedding_dim) 
    where views 1 and 2 of the same row are adjacent.
    """
    batch_size = features.shape[0] // 2
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # Mask out self-contrast
    mask = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    similarity_matrix.masked_fill_(mask, -9e15)
    
    # Targets: view 1 matches view 2
    targets = torch.arange(features.shape[0], device=features.device)
    targets = targets + 1 - 2 * (targets % 2) 
    
    return F.cross_entropy(similarity_matrix, targets)

def train_cv_lgbm(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42, do_TE=True, use_embeddings=False):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()

        if do_TE:
            te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
            te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
            te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)
    
            X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
            X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])
    
            X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
            X_test_enc_10  = te_10.transform(X_tst[TE_COLS])
    
            X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
            X_test_enc_100  = te_100.transform(X_tst[TE_COLS])
    
            for i, col in enumerate(TE_COLS):
                X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
                X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
                X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
    
                X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
                X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
                X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')
    
                X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
                X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
                X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')
    
                # Drop raw high-cardinality string columns
                X_tr.drop(columns=[col], inplace=True)
                X_val.drop(columns=[col], inplace=True)
                X_tst.drop(columns=[col], inplace=True)
        else:
            X_tr.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_val.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_tst.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
        if use_embeddings:
            X_tr_np = X_tr.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)
            X_tst_np = X_tst.values.astype(np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = TabularEncoder(input_dim=X_tr_np.shape[1], embedding_dim=32).to(device)
            optimizer = optim.Adam(encoder.parameters(), lr=0.003)

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr_np, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)

            encoder.train()
            for epoch in tqdm(range(10)):
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    
                    mask = (torch.rand_like(batch_x) > 0.15).float()
                    x_view1 = batch_x
                    x_view2 = batch_x * mask

                    combined = torch.empty((batch_x.shape[0] * 2, batch_x.shape[1]), dtype=torch.float32, device=device)
                    combined[0::2] = x_view1
                    combined[1::2] = x_view2

                    embeddings = encoder(combined)
                    loss = info_nce_loss(embeddings, temperature=0.1)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            # Extraction
            encoder.eval()
            def extract_embeddings(data_np, batch_size=2048):
                emb_list = []
                with torch.no_grad():
                    for i in range(0, len(data_np), batch_size):
                        batch = torch.tensor(data_np[i:i + batch_size], dtype=torch.float32).to(device)
                        emb_list.append(encoder(batch).cpu().numpy())
                return np.vstack(emb_list)

            X_tr_emb = extract_embeddings(X_tr_np)
            X_val_emb = extract_embeddings(X_val_np)
            X_tst_emb = extract_embeddings(X_tst_np)

            # Cleanup
            del encoder, optimizer, loader, dataset
            torch.cuda.empty_cache()
            gc.collect()

            # Attach embeddings
            emb_cols = [f"emb_{k}" for k in range(X_tr_emb.shape[1])]
            df_tr_emb = pd.DataFrame(X_tr_emb, columns=emb_cols, index=X_tr.index)
            df_val_emb = pd.DataFrame(X_val_emb, columns=emb_cols, index=X_val.index)
            df_tst_emb = pd.DataFrame(X_tst_emb, columns=emb_cols, index=X_tst.index)

            X_tr = pd.concat([X_tr, df_tr_emb], axis=1)
            X_val = pd.concat([X_val, df_val_emb], axis=1)
            X_tst = pd.concat([X_tst, df_tst_emb], axis=1)

        model = lgb.LGBMClassifier(**params, random_state=seed, n_jobs=-1)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(300, first_metric_only=True, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_tst)[:, 1] / n_splits

    return oof, test_pred, roc_auc_score(y, oof)

def train_cv_xgb(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42, do_TE=True, use_embeddings=False):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()
        
        if do_TE:
            te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
            te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
            te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)
    
            X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
            X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])
    
            X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
            X_test_enc_10  = te_10.transform(X_tst[TE_COLS])
    
            X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
            X_test_enc_100  = te_100.transform(X_tst[TE_COLS])
    
            for i, col in enumerate(TE_COLS):
                X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
                X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
                X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
    
                X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
                X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
                X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')
    
                X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
                X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
                X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')
    
                # Drop raw high-cardinality string columns
                X_tr.drop(columns=[col], inplace=True)
                X_val.drop(columns=[col], inplace=True)
                X_tst.drop(columns=[col], inplace=True)
        else:
            X_tr.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_val.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_tst.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
        if use_embeddings:
            X_tr_np = X_tr.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)
            X_tst_np = X_tst.values.astype(np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = TabularEncoder(input_dim=X_tr_np.shape[1], embedding_dim=32).to(device)
            optimizer = optim.Adam(encoder.parameters(), lr=0.003)

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr_np, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)

            encoder.train()
            for epoch in tqdm(range(10)):
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    
                    mask = (torch.rand_like(batch_x) > 0.15).float()
                    x_view1 = batch_x
                    x_view2 = batch_x * mask

                    combined = torch.empty((batch_x.shape[0] * 2, batch_x.shape[1]), dtype=torch.float32, device=device)
                    combined[0::2] = x_view1
                    combined[1::2] = x_view2

                    embeddings = encoder(combined)
                    loss = info_nce_loss(embeddings, temperature=0.1)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            # Extraction
            encoder.eval()
            def extract_embeddings(data_np, batch_size=2048):
                emb_list = []
                with torch.no_grad():
                    for i in range(0, len(data_np), batch_size):
                        batch = torch.tensor(data_np[i:i + batch_size], dtype=torch.float32).to(device)
                        emb_list.append(encoder(batch).cpu().numpy())
                return np.vstack(emb_list)

            X_tr_emb = extract_embeddings(X_tr_np)
            X_val_emb = extract_embeddings(X_val_np)
            X_tst_emb = extract_embeddings(X_tst_np)

            # Cleanup
            del encoder, optimizer, loader, dataset
            torch.cuda.empty_cache()
            gc.collect()

            # Attach embeddings
            emb_cols = [f"emb_{k}" for k in range(X_tr_emb.shape[1])]
            df_tr_emb = pd.DataFrame(X_tr_emb, columns=emb_cols, index=X_tr.index)
            df_val_emb = pd.DataFrame(X_val_emb, columns=emb_cols, index=X_val.index)
            df_tst_emb = pd.DataFrame(X_tst_emb, columns=emb_cols, index=X_tst.index)

            X_tr = pd.concat([X_tr, df_tr_emb], axis=1)
            X_val = pd.concat([X_val, df_val_emb], axis=1)
            X_tst = pd.concat([X_tst, df_tst_emb], axis=1)

        # dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_cols, enable_categorical=True)
        # dvalid = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols, enable_categorical=True)
        # dtest = xgb.DMatrix(X_tst, feature_names=feature_cols, enable_categorical=True)
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        dtest = xgb.DMatrix(X_tst, enable_categorical=True)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=[(dvalid, 'valid')],
            early_stopping_rounds=300,
            verbose_eval=False,
        )
        
        oof[val_idx] = model.predict(dvalid)
        test_pred += model.predict(dtest) / n_splits
        gc.collect()
    
    if params.get('objective') == 'rank:pairwise':
        oof = scale_group(oof)
        oof = np.clip(oof, 0.01,1.00)
        test_pred = scale_group(test_pred)
        test_pred = np.clip(test_pred, 0.01, 1.00)

    return oof, test_pred, roc_auc_score(y, oof)

def train_cv_catboost(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42, do_TE=True, use_embeddings=False):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cat_cols = [c for c in feature_cols if str(X[c].dtype) == 'category']

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()

        if do_TE:
            te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
            te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
            te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)
    
            X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
            X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])
    
            X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
            X_test_enc_10  = te_10.transform(X_tst[TE_COLS])
    
            X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
            X_test_enc_100  = te_100.transform(X_tst[TE_COLS])
    
            for i, col in enumerate(TE_COLS):
                X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
                X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
                X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
    
                X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
                X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
                X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')
    
                X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
                X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
                X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')
    
                # Drop raw high-cardinality string columns
                X_tr.drop(columns=[col], inplace=True)
                X_val.drop(columns=[col], inplace=True)
                X_tst.drop(columns=[col], inplace=True)
        else:
            X_tr.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_val.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_tst.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
        if use_embeddings:
            X_tr_np = X_tr.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)
            X_tst_np = X_tst.values.astype(np.float32)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoder = TabularEncoder(input_dim=X_tr_np.shape[1], embedding_dim=32).to(device)
            optimizer = optim.Adam(encoder.parameters(), lr=0.003)

            dataset = torch.utils.data.TensorDataset(torch.tensor(X_tr_np, dtype=torch.float32))
            loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, drop_last=True, pin_memory=True)

            encoder.train()
            for epoch in tqdm(range(10)):
                for (batch_x,) in loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    
                    mask = (torch.rand_like(batch_x) > 0.15).float()
                    x_view1 = batch_x
                    x_view2 = batch_x * mask

                    combined = torch.empty((batch_x.shape[0] * 2, batch_x.shape[1]), dtype=torch.float32, device=device)
                    combined[0::2] = x_view1
                    combined[1::2] = x_view2

                    embeddings = encoder(combined)
                    loss = info_nce_loss(embeddings, temperature=0.1)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            # Extraction
            encoder.eval()
            def extract_embeddings(data_np, batch_size=2048):
                emb_list = []
                with torch.no_grad():
                    for i in range(0, len(data_np), batch_size):
                        batch = torch.tensor(data_np[i:i + batch_size], dtype=torch.float32).to(device)
                        emb_list.append(encoder(batch).cpu().numpy())
                return np.vstack(emb_list)

            X_tr_emb = extract_embeddings(X_tr_np)
            X_val_emb = extract_embeddings(X_val_np)
            X_tst_emb = extract_embeddings(X_tst_np)

            # Cleanup
            del encoder, optimizer, loader, dataset
            torch.cuda.empty_cache()
            gc.collect()

            # Attach embeddings
            emb_cols = [f"emb_{k}" for k in range(X_tr_emb.shape[1])]
            df_tr_emb = pd.DataFrame(X_tr_emb, columns=emb_cols, index=X_tr.index)
            df_val_emb = pd.DataFrame(X_val_emb, columns=emb_cols, index=X_val.index)
            df_tst_emb = pd.DataFrame(X_tst_emb, columns=emb_cols, index=X_tst.index)

            X_tr = pd.concat([X_tr, df_tr_emb], axis=1)
            X_val = pd.concat([X_val, df_val_emb], axis=1)
            X_tst = pd.concat([X_tst, df_tst_emb], axis=1)

        cat_feats = [c for c in X_tr.columns if str(X_tr[c].dtype) == 'category']

        model = CatBoostClassifier(**{**params, 'random_seed': seed, 'verbose': False})
        model.fit(
            X_tr, y_tr,
            eval_set=(X_val, y_val),
            cat_features=cat_feats,
            early_stopping_rounds=300,
            verbose=False,
        )

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_tst)[:, 1] / n_splits

    return oof, test_pred, roc_auc_score(y, oof)

def train_cv_tabm(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cat_cols = [c for c in feature_cols if str(X[c].dtype) == 'category']

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()
        
        te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
        te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
        te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)

        X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
        X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])

        X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
        X_test_enc_10  = te_10.transform(X_tst[TE_COLS])

        X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
        X_test_enc_100  = te_100.transform(X_tst[TE_COLS])

        for i, col in enumerate(TE_COLS):
            X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
            X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
            X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')

            X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
            X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
            X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')

            X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
            X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
            X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')

            # Drop raw high-cardinality string columns
            X_tr.drop(columns=[col], inplace=True)
            X_val.drop(columns=[col], inplace=True)
            X_tst.drop(columns=[col], inplace=True)
        model = TabM_D_Classifier(**{**params, 'random_state': seed, 'val_metric_name':'1-auc_ovr'})
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_tst)[:, 1] / n_splits

    return oof, test_pred, roc_auc_score(y, oof)

def train_cv_mlp(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cat_cols = [c for c in feature_cols if str(X[c].dtype) == 'category']

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()
        
        te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
        te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
        te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)

        X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
        X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])

        X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
        X_test_enc_10  = te_10.transform(X_tst[TE_COLS])

        X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
        X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
        X_test_enc_100  = te_100.transform(X_tst[TE_COLS])

        for i, col in enumerate(TE_COLS):
            X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
            X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
            X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')

            X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
            X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
            X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')

            X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
            X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
            X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')

            # Drop raw high-cardinality string columns
            X_tr.drop(columns=[col], inplace=True)
            X_val.drop(columns=[col], inplace=True)
            X_tst.drop(columns=[col], inplace=True)
            
        params['random_state'] = seed
        seed_everything(seed)
        model = RealMLP_TD_Classifier(**params)
        model.fit(X_tr, y_tr, X_val, y_val, cat_col_names=cat_cols, X_test=X_tst)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_tst)[:, 1] / n_splits

    return oof, test_pred, roc_auc_score(y, oof)

def train_cv_lr(X, y, X_test, feature_cols, params, TE_COLS, n_splits=5, seed=42, do_TE=True):
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_test))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    # Identify categorical columns that are NOT in TE_COLS 
    # (Logistic Regression requires OHE for these, or you can drop them if TE_COLS covers them all)
    cat_cols = [c for c in feature_cols if str(X[c].dtype) == 'category' and c not in TE_COLS]

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[tr_idx][feature_cols], y.iloc[tr_idx]
        X_val, y_val = X.iloc[val_idx][feature_cols], y.iloc[val_idx]
        X_tst = X_test[feature_cols].copy()
        
        if do_TE:
            te_auto = TargetEncoder(shuffle=True, cv=n_splits, smooth='auto', random_state=42)
            te_10   = TargetEncoder(shuffle=True, cv=n_splits, smooth=10.0, random_state=42)
            te_100  = TargetEncoder(shuffle=True, cv=n_splits, smooth=100.0, random_state=42)
    
            X_train_enc_auto = te_auto.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_auto = te_auto.transform(X_val[TE_COLS])
            X_test_enc_auto  = te_auto.transform(X_test[TE_COLS])
    
            X_train_enc_10 = te_10.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_10 = te_10.transform(X_val[TE_COLS])
            X_test_enc_10  = te_10.transform(X_tst[TE_COLS])
    
            X_train_enc_100 = te_100.fit_transform(X_tr[TE_COLS], y_tr)
            X_valid_enc_100 = te_100.transform(X_val[TE_COLS])
            X_test_enc_100  = te_100.transform(X_tst[TE_COLS])
    
            for i, col in enumerate(TE_COLS):
                X_tr[f"{col}_TE_auto"] = X_train_enc_auto[:, i].astype('float32')
                X_val[f"{col}_TE_auto"] = X_valid_enc_auto[:, i].astype('float32')
                X_tst[f"{col}_TE_auto"] = X_test_enc_auto[:, i].astype('float32')
    
                X_tr[f"{col}_TE_10"] = X_train_enc_10[:, i].astype('float32')
                X_val[f"{col}_TE_10"] = X_valid_enc_10[:, i].astype('float32')
                X_tst[f"{col}_TE_10"] = X_test_enc_10[:, i].astype('float32')
    
                X_tr[f"{col}_TE_100"] = X_train_enc_100[:, i].astype('float32')
                X_val[f"{col}_TE_100"] = X_valid_enc_100[:, i].astype('float32')
                X_tst[f"{col}_TE_100"] = X_test_enc_100[:, i].astype('float32')
    
                # Drop raw high-cardinality string columns
                X_tr.drop(columns=[col], inplace=True)
                X_val.drop(columns=[col], inplace=True)
                X_tst.drop(columns=[col], inplace=True)
        else:
            X_tr.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_val.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            X_tst.drop(columns=['Income_100_floor', 'Income_1000_floor'], inplace=True)
            
        # Drop any remaining non-numeric categorical columns (LR cannot process pandas categories)
        if cat_cols:
            X_tr.drop(columns=cat_cols, inplace=True)
            X_val.drop(columns=cat_cols, inplace=True)
            X_tst.drop(columns=cat_cols, inplace=True)

        # --- Imputation & Scaling (Required for Linear Models) ---
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()

        X_tr = scaler.fit_transform(imputer.fit_transform(X_tr))
        X_val = scaler.transform(imputer.transform(X_val))
        X_tst = scaler.transform(imputer.transform(X_tst))

        # --- Modeling ---
        model = LogisticRegression(**params, random_state=seed)
        model.fit(X_tr, y_tr)

        oof[val_idx] = model.predict_proba(X_val)[:, 1]
        test_pred += model.predict_proba(X_tst)[:, 1] / n_splits
        
        del model, X_tr, X_val, X_tst, imputer, scaler
        gc.collect()

    return oof, test_pred, roc_auc_score(y, oof)

def seed_average(train_fn, X, y, X_test, feature_cols, params, seeds, TE_COLS, **kwargs):
    oof_list, test_list = [], []
    for seed in seeds:
        oof, test_pred, auc = train_fn(X, y, X_test, feature_cols, params, seed=seed, TE_COLS=TE_COLS, **kwargs)
        oof_list.append(oof)
        test_list.append(test_pred)
        print(f'Seed {seed:>4} | AUC={auc:.6f}')

    avg_oof = np.mean(oof_list, axis=0)
    avg_test = np.mean(test_list, axis=0)
    print(f'Average OOF AUC: {roc_auc_score(y, avg_oof):.6f}')
    return avg_oof, avg_test

def rank_blend(oof_dict, test_dict, y):
    names = list(oof_dict.keys())
    ranked_oof = {k: rankdata(v) for k, v in oof_dict.items()}
    ranked_test = {k: rankdata(v) for k, v in test_dict.items()}

    def objective(weights):
        w = np.abs(weights)
        w = w / w.sum()
        blend = np.column_stack([ranked_oof[n] for n in names]) @ w
        return -roc_auc_score(y, blend)

    x0 = np.ones(len(names)) / len(names)
    result = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 5000})
    weights = np.abs(result.x)
    weights = weights / weights.sum()

    oof_blend = np.column_stack([ranked_oof[n] for n in names]) @ weights
    test_blend = np.column_stack([ranked_test[n] for n in names]) @ weights

    print('Optimal rank-blend weights:')
    for name, weight in zip(names, weights):
        print(f'  {name}: {weight:.4f}')
    print(f'Blend OOF AUC: {roc_auc_score(y, oof_blend):.6f}')
    return oof_blend, test_blend, dict(zip(names, weights))

def scale_group(group):
    # Handle edge case where all predictions in a group are identical
    if group.max() == group.min():
        return np.ones_like(group) * 0.5
    return (group - group.min()) / (group.max() - group.min())