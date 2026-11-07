# %% [code]
import math
import random
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, tfms):
        self._tfms = [t for t in tfms if t in ("median_center", "robust_scale", "smooth_clip", "l2_normalize")]

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if "median_center" in self._tfms or "robust_scale" in self._tfms:
            self._median = np.nanmedian(X, axis=0).astype(np.float32)
            q_diff = (np.nanquantile(X, 0.75, axis=0) - np.nanquantile(X, 0.25, axis=0)).astype(np.float32)
            zero_idx = q_diff == 0.0
            if zero_idx.any():
                q_diff[zero_idx] = 0.5 * (np.nanmax(X, axis=0)[zero_idx] - np.nanmin(X, axis=0)[zero_idx])
            self._iqr_factors = (1.0 / (q_diff + 1e-30)).astype(np.float32)
            self._iqr_factors[q_diff == 0.0] = 0.0
        return self

    def transform(self, X, y=None):
        X = np.asarray(X, dtype=np.float32).copy()
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        for tfm in self._tfms:
            if tfm == "median_center":
                X -= self._median[None, :]
            elif tfm == "robust_scale":
                X *= self._iqr_factors[None, :]
            elif tfm == "smooth_clip":
                X = X / np.sqrt(1 + (X / 3) ** 2)
            elif tfm == "l2_normalize":
                norms = np.linalg.norm(X, axis=1, keepdims=True)
                X /= np.where(norms == 0, 1.0, norms)
        return X.astype(np.float32)


class CategoricalFeatureLayer(nn.Module):
    def __init__(self, n_ens, cat_dims, embed_dim=8, onehot_thresh=8):
        super().__init__()
        self.n_ens = n_ens
        self.cat_dims = list(cat_dims)
        self.onehot_features = []
        self.embed_layers = nn.ModuleList()
        self._embed_feature_indices = []
        for i, dim in enumerate(self.cat_dims):
            if dim <= onehot_thresh:
                self.onehot_features.append(i)
            else:
                self.embed_layers.append(nn.ModuleList([nn.Embedding(dim, embed_dim) for _ in range(n_ens)]))
                self._embed_feature_indices.append(i)

    def forward(self, x):
        batch_size, n_ens, _ = x.shape
        features = []
        if self.onehot_features:
            onehot_x = x[:, :, self.onehot_features]
            onehot_dims = [self.cat_dims[i] for i in self.onehot_features]
            encoded = torch.zeros(batch_size, n_ens, sum(onehot_dims), device=x.device, dtype=torch.float32)
            start = 0
            for idx, dim in enumerate(onehot_dims):
                pos = onehot_x[:, :, idx:idx + 1].long().clamp(0, dim - 1)
                encoded.scatter_(2, pos + start, 1.0)
                start += dim
            features.append(encoded)
        for emb_list, feat_idx in zip(self.embed_layers, self._embed_feature_indices):
            dim = self.cat_dims[feat_idx]
            feat_embs = []
            for model_idx in range(self.n_ens):
                indices = x[:, model_idx, feat_idx:feat_idx + 1].long().clamp(0, dim - 1)
                feat_embs.append(emb_list[model_idx](indices))
            features.append(torch.cat(feat_embs, dim=1))
        if not features:
            return torch.empty(batch_size, n_ens, 0, device=x.device)
        return torch.cat(features, dim=2)


class ScalingLayer(nn.Module):
    def __init__(self, n_ens, n_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(n_ens, n_features))

    def forward(self, x):
        return x * self.scale[None, :, :]


class NTPLinear(nn.Module):
    def __init__(self, n_ens, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.randn(n_ens, in_features, out_features))
        self.bias = nn.Parameter(torch.randn(n_ens, out_features)) if bias else None

    def forward(self, x):
        x = torch.einsum("bki,kio->bko", x, self.weight) / math.sqrt(self.in_features)
        if self.bias is not None:
            x = x + self.bias
        return x


class PBLDEmbedding(nn.Module):
    def __init__(self, n_ens, n_features, hidden_dim=16, out_dim=4, freq_scale=0.1, activation=nn.GELU):
        super().__init__()
        self.out_dim = out_dim
        self.w1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim) * freq_scale)
        self.b1 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim))
        self.w2 = nn.Parameter(torch.randn(n_ens, n_features, hidden_dim, out_dim - 1) / math.sqrt(hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(n_ens, n_features, out_dim - 1))
        self.act = activation()
        nn.init.uniform_(self.b1, -math.pi, math.pi)

    def forward(self, x):
        periodic = torch.cos(2 * math.pi * (x.unsqueeze(-1) * self.w1.unsqueeze(0) + self.b1.unsqueeze(0)))
        transformed = self.act(torch.einsum("bkfh,kfhd->bkfd", periodic, self.w2) + self.b2.unsqueeze(0))
        return torch.cat([x.unsqueeze(-1), transformed], dim=-1).flatten(start_dim=2)


class RealMLP(nn.Module):
    def __init__(self, output_dim, cat_dims, n_numerical, cfg):
        super().__init__()
        n_ens = cfg["n_ens"]
        self.n_ens = n_ens
        self.cate = CategoricalFeatureLayer(
            n_ens=n_ens,
            cat_dims=cat_dims,
            embed_dim=cfg["embed_dim"],
            onehot_thresh=cfg["onehot_thresh"],
        )
        self.num_embed = PBLDEmbedding(
            n_ens=n_ens,
            n_features=n_numerical,
            hidden_dim=cfg["pbld_hidden_dim"],
            out_dim=cfg["pbld_out_dim"],
            freq_scale=cfg["pbld_freq_scale"],
            activation=cfg["pbld_activation"],
        )
        num_emb_dim = n_numerical * cfg["pbld_out_dim"]
        cat_emb_dim = sum(c if c <= cfg["onehot_thresh"] else cfg["embed_dim"] for c in cat_dims)
        total_dim = num_emb_dim + cat_emb_dim
        layers = []
        if cfg["add_front_scale"]:
            layers.append(ScalingLayer(n_ens=n_ens, n_features=total_dim))
        self._dropout_modules = []
        in_dim = total_dim
        for i, out_dim in enumerate(cfg["hidden_dims"]):
            linear = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=out_dim)
            if i == 0:
                self.first_linear = linear
            drop = nn.Dropout(cfg["dropout"])
            self._dropout_modules.append(drop)
            layers += [linear, cfg["activation"](), drop]
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)
        self.output_layer = NTPLinear(n_ens=n_ens, in_features=in_dim, out_features=output_dim)

    def forward(self, x_num, x_cat):
        x_num = x_num.unsqueeze(1).expand(-1, self.n_ens, -1)
        x_cat = x_cat.unsqueeze(1).expand(-1, self.n_ens, -1)
        combined = torch.cat([self.num_embed(x_num), self.cate(x_cat)], dim=2)
        return self.output_layer(self.hidden(combined))


def apply_schedule(init_value, progress, sched, flat_ratio=0.3):
    if sched == "constant":
        return init_value
    if sched == "cos":
        return init_value * (math.cos(math.pi * progress) + 1) / 2
    if sched == "flat_cos":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (math.cos(math.pi * t) + 1) / 2
    if sched == "flat_anneal":
        if progress < flat_ratio:
            return init_value
        t = (progress - flat_ratio) / (1 - flat_ratio)
        return init_value * (1 - t)
    if sched == "sqrt_cos":
        return init_value * math.sqrt((math.cos(math.pi * progress) + 1) / 2)
    if sched == "expm4t":
        return init_value * math.exp(-4 * progress)
    raise ValueError(f"Unknown schedule: {sched}")


def get_parameter_groups(model, p):
    first_linear_weight_id = id(model.first_linear.weight)
    scale_p, pbld_p, first_w_p, other_w_p, bias_p = [], [], [], [], []
    for name, param in model.named_parameters():
        if "num_embed" in name:
            pbld_p.append(param)
        elif "scale" in name:
            scale_p.append(param)
        elif id(param) == first_linear_weight_id:
            first_w_p.append(param)
        elif "bias" in name:
            bias_p.append(param)
        else:
            other_w_p.append(param)
    lr = p["lr"]
    wd = p["weight_decay"]
    return [
        {"params": scale_p, "lr": lr * p["lr_scale_mult"], "weight_decay": wd * p["wd_scale_mult"], "group": "scale"},
        {"params": pbld_p, "lr": lr * p["pbld_lr_factor"], "weight_decay": wd, "group": "pbld"},
        {"params": first_w_p, "lr": lr * p["first_layer_lr_factor"], "weight_decay": wd * p["first_layer_wd_factor"], "group": "first_w"},
        {"params": other_w_p, "lr": lr, "weight_decay": wd, "group": "other_w"},
        {"params": bias_p, "lr": lr * p["lr_bias_mult"], "weight_decay": wd * p["wd_bias_mult"], "group": "bias"},
    ]


def smooth_ce_loss_from_logits(y_true, logits, ls=0.0, class_weights=None, focal_gamma=0.0, logit_bias=None):
    n_classes = logits.size(1)
    if logit_bias is not None:
        logits = logits + logit_bias[None, :]
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    y_smooth = torch.full_like(log_probs, ls / n_classes)
    y_smooth.scatter_(1, y_true.unsqueeze(1), 1.0 - ls + ls / n_classes)
    per_sample_loss = -(y_smooth * log_probs).sum(dim=1)
    if focal_gamma > 0:
        pt = probs.gather(1, y_true.unsqueeze(1)).squeeze(1).clamp(1e-15, 1.0)
        per_sample_loss = per_sample_loss * torch.pow(1.0 - pt, focal_gamma)
    if class_weights is not None:
        sample_weights = class_weights[y_true]
        return (per_sample_loss * sample_weights).sum() / sample_weights.sum()
    return per_sample_loss.mean()


class RealMLP_TD_Classifier(BaseEstimator):
    def __init__(self, **kwargs):
        self.params = kwargs

    def fit(self, X_train, y_train, X_val, y_val, cat_col_names=None, X_test=None):
        p = self.params
        dev = torch.device(p["device"] if torch.cuda.is_available() else "cpu")
        verbose = p["verbosity"]
        cat_col_names = [c for c in (cat_col_names or []) if c in X_train.columns]
        num_col_names = [c for c in X_train.columns if c not in cat_col_names]

        x_tr_num = X_train[num_col_names].values.astype(np.float32)
        x_va_num = X_val[num_col_names].values.astype(np.float32)
        x_tr_cat = X_train[cat_col_names].astype("int32").values.astype(np.int64) if cat_col_names else np.zeros((len(X_train), 0), dtype=np.int64)
        x_va_cat = X_val[cat_col_names].astype("int32").values.astype(np.int64) if cat_col_names else np.zeros((len(X_val), 0), dtype=np.int64)
        y_tr = np.asarray(y_train, dtype=np.int64)
        y_va = np.asarray(y_val, dtype=np.int64)

        self.preprocessor_ = NumericalPreprocessor(p["tfms"])
        self.preprocessor_.fit(x_tr_num)
        x_tr_num = self.preprocessor_.transform(x_tr_num)
        x_va_num = self.preprocessor_.transform(x_va_num)
        self.cat_col_names_ = cat_col_names
        self.num_col_names_ = num_col_names

        if cat_col_names:
            all_cat = [x_tr_cat, x_va_cat]
            if X_test is not None:
                all_cat.append(X_test[cat_col_names].astype("int32").values.astype(np.int64))
            cat_dims = (np.concatenate(all_cat, axis=0).max(axis=0) + 1).clip(min=1).tolist()
            cat_max = np.array(cat_dims) - 1
            x_tr_cat = np.clip(x_tr_cat, 0, cat_max)
            x_va_cat = np.clip(x_va_cat, 0, cat_max)
        else:
            cat_dims = []
        self.cat_dims_ = cat_dims

        classes = np.unique(y_tr)
        self.classes_ = classes
        weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
        cw_power = float(p.get("class_weight_power", 1.0))
        if cw_power != 1.0:
            weights_np = np.power(weights_np, cw_power)
        cw_mult = p.get("class_weight_multipliers")
        if cw_mult is not None:
            weights_np = weights_np * np.asarray(cw_mult, dtype=np.float64)
        class_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=dev)

        logit_bias = None
        prior_power = float(p.get("loss_prior_power", 0.0))
        if prior_power != 0.0:
            counts = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
            priors = counts / counts.sum()
            centered = np.log(np.clip(priors, 1e-12, None)) - np.log(np.clip(priors, 1e-12, None)).mean()
            logit_bias = torch.as_tensor(centered * prior_power, dtype=torch.float32, device=dev)

        self.model_ = RealMLP(len(classes), cat_dims, x_tr_num.shape[1], p).to(dev)
        param_groups = get_parameter_groups(self.model_, p)
        for group in param_groups:
            group["lr_base"] = group["lr"]
        optimizer = torch.optim.AdamW(param_groups, betas=(p["mom"], p["sq_mom"]))

        xtn = torch.as_tensor(x_tr_num, dtype=torch.float32, device=dev)
        xtc = torch.as_tensor(x_tr_cat, dtype=torch.long, device=dev)
        ytt = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
        xvn = torch.as_tensor(x_va_num, dtype=torch.float32, device=dev)
        xvc = torch.as_tensor(x_va_cat, dtype=torch.long, device=dev)

        n_ens = p["n_ens"]
        train_bs = p["train_bs"]
        eval_bs = p["eval_bs"]
        epochs = p["epochs"]
        total_steps = max(1, epochs * len(y_tr))
        train_order = np.arange(len(y_tr))
        rng = np.random.default_rng(int(p.get("random_state", 0)) + 2027)
        sample_probs = None
        sw_power = float(p.get("sample_weight_power", 0.0))
        if sw_power > 0:
            counts = np.bincount(y_tr, minlength=len(classes)).astype(np.float64)
            sample_probs = np.power(1.0 / np.clip(counts[y_tr], 1.0, None), sw_power)
            sample_probs /= sample_probs.sum()

        best_score = -np.inf
        best_epoch = 0
        best_val_probs = None
        best_state = None
        ema_decay = float(p.get("ema_decay", 0.0))
        ema_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()} if ema_decay > 0 else None

        for epoch in range(epochs):
            self.model_.train()
            if sample_probs is not None:
                epoch_order = rng.choice(len(y_tr), size=len(y_tr), replace=True, p=sample_probs)
            else:
                epoch_order = train_order.copy()
                np.random.shuffle(epoch_order)
            for start in range(0, len(y_tr), train_bs):
                progress = (epoch * len(y_tr) + start) / total_steps
                idx = epoch_order[start:start + train_bs]
                for group in optimizer.param_groups:
                    group["lr"] = apply_schedule(group["lr_base"], progress, p["lr_sched"], p["flat_ratio"])
                optimizer.zero_grad(set_to_none=True)
                x_num_batch = xtn[idx]
                if float(p.get("numeric_noise_std", 0.0)) > 0:
                    x_num_batch = x_num_batch + torch.randn_like(x_num_batch) * float(p["numeric_noise_std"])
                logits = self.model_(x_num_batch, xtc[idx])
                ls_val = apply_schedule(p["ls_eps"], progress, p["ls_eps_sched"], p["flat_ratio"])
                drop_val = apply_schedule(p["dropout"], progress, p["p_drop_sched"], p["flat_ratio"])
                for drop in self.model_._dropout_modules:
                    drop.p = drop_val
                loss = smooth_ce_loss_from_logits(
                    ytt[idx].repeat_interleave(n_ens),
                    logits.reshape(-1, len(classes)),
                    ls=ls_val,
                    class_weights=class_weights,
                    focal_gamma=float(p.get("focal_gamma", 0.0)),
                    logit_bias=logit_bias,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), p["grad_clip"])
                optimizer.step()
                if ema_state is not None:
                    with torch.no_grad():
                        for key, value in self.model_.state_dict().items():
                            if torch.is_floating_point(value):
                                ema_state[key].mul_(ema_decay).add_(value.detach(), alpha=1.0 - ema_decay)
                            else:
                                ema_state[key].copy_(value)

            self.model_.eval()
            live_state = None
            if ema_state is not None:
                live_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
                self.model_.load_state_dict(ema_state, strict=True)
            with torch.no_grad():
                val_probs = np.concatenate([
                    F.softmax(self.model_(xvn[s:s + eval_bs], xvc[s:s + eval_bs]), dim=2).mean(dim=1).cpu().numpy()
                    for s in range(0, len(y_va), eval_bs)
                ], axis=0)
            if live_state is not None:
                self.model_.load_state_dict(live_state, strict=True)

            score_probs = val_probs
            eval_mult = p.get("eval_class_multipliers")
            if eval_mult is not None:
                eval_mult = np.asarray(eval_mult, dtype=np.float32)
                score_probs = val_probs * eval_mult[None, :]
                score_probs /= np.clip(score_probs.sum(axis=1, keepdims=True), 1e-12, None)
            epoch_score = balanced_accuracy_score(y_va, np.argmax(score_probs, axis=1))
            improved = epoch_score > best_score
            if improved:
                best_score = epoch_score
                best_epoch = epoch + 1
                best_val_probs = score_probs.copy()
                source = ema_state if ema_state is not None else self.model_.state_dict()
                best_state = {k: v.detach().clone() for k, v in source.items()}
            if verbose >= 2:
                print(f"  epoch {epoch + 1}/{epochs}  score={epoch_score:.5f}  best={best_score:.5f}  ls={ls_val:.4f}  drop={drop_val:.4f}" + (" *" if improved else ""))

        if best_state is not None:
            self.model_.load_state_dict(best_state, strict=True)
        self.best_score_ = best_score
        self.best_epoch_ = best_epoch
        self.best_val_probs_ = best_val_probs
        self._dev = dev
        if verbose >= 1:
            print(f"   best score: {best_score:.5f}  (epoch {best_epoch})")
        return self

    def predict_proba(self, X):
        eval_bs = self.params["eval_bs"]
        x_num = self.preprocessor_.transform(X[self.num_col_names_].values.astype(np.float32))
        if self.cat_col_names_:
            x_cat = X[self.cat_col_names_].astype("int32").values.astype(np.int64)
            x_cat = np.clip(x_cat, 0, np.array(self.cat_dims_) - 1)
        else:
            x_cat = np.zeros((len(X), 0), dtype=np.int64)
        x_num = torch.as_tensor(x_num, dtype=torch.float32, device=self._dev)
        x_cat = torch.as_tensor(x_cat, dtype=torch.long, device=self._dev)
        self.model_.eval()
        with torch.no_grad():
            return np.concatenate([
                F.softmax(self.model_(x_num[s:s + eval_bs], x_cat[s:s + eval_bs]), dim=2).mean(dim=1).cpu().numpy()
                for s in range(0, len(X), eval_bs)
            ], axis=0)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]