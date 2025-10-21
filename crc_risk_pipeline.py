
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRC multimodal risk pipeline (runnable scaffold)
- Late-fusion base learners (per modality)
- Discrete-time survival (person-period logistic) to produce S(t)
- Parallel present-time fused classifier with isotonic calibration
Dependencies: numpy, pandas, scikit-learn
Optional: xgboost (if available), but scaffold uses sklearn-only to run everywhere
Author: Your Team
"""
from __future__ import annotations
import os
import math
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedKFold

RNG = np.random.default_rng(42)


# ----------------------------- Utility transforms -----------------------------
def winsorize(x, lower_q=0.01, upper_q=0.99):
    lo, hi = np.nanquantile(x, [lower_q, upper_q])
    return np.clip(x, lo, hi)

def zscore(x, mean=None, std=None):
    if mean is None: mean = np.nanmean(x)
    if std is None: std = np.nanstd(x) if np.nanstd(x)>0 else 1.0
    return (x - mean) / std, mean, std

def add_pseudocount(x, eps=1e-6):
    return x + eps

def clr_transform(rel_abund_matrix: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Centered log-ratio transform for compositional data."""
    X = add_pseudocount(rel_abund_matrix, eps)
    gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return np.log(X / gm)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ----------------------------- Feature Featurizers -----------------------------

@dataclass
class ClinicalState:
    num_means: Dict[str, float]
    num_stds: Dict[str, float]
    numeric_cols: List[str]
    cat_cols: List[str]
    ohe_categories_: List[List[str]]

class ClinicalFeaturizer:
    """Featurize clinical metadata with z-scoring and one-hot encoding. Fit only on train to avoid leakage."""
    def __init__(self, numeric_cols: List[str], cat_cols: List[str]):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.state: Optional[ClinicalState] = None
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    def fit(self, df: pd.DataFrame):
        num_means = {}
        num_stds = {}
        Xn = []
        for c in self.numeric_cols:
            xw = winsorize(df[c].to_numpy())
            z, m, s = zscore(xw)
            num_means[c] = float(m); num_stds[c] = float(s)
            Xn.append(z.reshape(-1,1))
        Xn = np.hstack(Xn) if Xn else np.zeros((len(df),0))

        if self.cat_cols:
            self.ohe.fit(df[self.cat_cols].astype(str))
            ohe_categories_ = [list(c) for c in self.ohe.categories_]
        else:
            ohe_categories_ = []

        self.state = ClinicalState(num_means, num_stds, self.numeric_cols, self.cat_cols, ohe_categories_)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.state is not None, "ClinicalFeaturizer not fit"
        Xn = []
        for c in self.state.numeric_cols:
            x = winsorize(df[c].to_numpy())
            z, _, _ = zscore(x, self.state.num_means[c], self.state.num_stds[c])
            Xn.append(z.reshape(-1,1))
        Xn = np.hstack(Xn) if Xn else np.zeros((len(df),0))

        if self.state.cat_cols:
            Xc = self.ohe.transform(df[self.state.cat_cols].astype(str))
        else:
            Xc = np.zeros((len(df),0))

        return np.hstack([Xn, Xc])


@dataclass
class GermlineState:
    prs_means: Dict[str, float]
    prs_stds: Dict[str, float]
    cols: List[str]

class GermlineFeaturizer:
    """Simple z-scoring for PRS per-ancestry not implemented here; add ancestry-aware logic as needed."""
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.state: Optional[GermlineState] = None

    def fit(self, df: pd.DataFrame):
        prs_means = {}; prs_stds = {}
        for c in self.cols:
            z, m, s = zscore(df[c].to_numpy())
            prs_means[c] = float(m); prs_stds[c] = float(s)
        self.state = GermlineState(prs_means, prs_stds, self.cols)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.state is not None, "GermlineFeaturizer not fit"
        X = []
        for c in self.state.cols:
            z, _, _ = zscore(df[c].to_numpy(), self.state.prs_means[c], self.state.prs_stds[c])
            X.append(z.reshape(-1,1))
        return np.hstack(X) if X else np.zeros((len(df),0))


@dataclass
class SimpleScalerState:
    means: Dict[str, float]
    stds: Dict[str, float]
    cols: List[str]

class SimpleScaler:
    """Generic train-only z-scaler for numeric columns."""
    def __init__(self, cols: List[str]):
        self.cols = cols
        self.state: Optional[SimpleScalerState] = None

    def fit(self, df: pd.DataFrame):
        means = {}; stds = {}
        for c in self.cols:
            z, m, s = zscore(df[c].to_numpy())
            means[c] = float(m); stds[c] = float(s)
        self.state = SimpleScalerState(means, stds, self.cols)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        assert self.state is not None
        X = []
        for c in self.state.cols:
            z, _, _ = zscore(df[c].to_numpy(), self.state.means[c], self.state.stds[c])
            X.append(z.reshape(-1,1))
        return np.hstack(X) if X else np.zeros((len(df),0))


class MicrobiomeFeaturizer:
    """CLR transform for selected taxa/pathway columns. Includes simple prevalence filter at fit time."""
    def __init__(self, cols: List[str], prevalence_threshold: float = 0.1):
        self.cols = cols
        self.prev = prevalence_threshold
        self.selected_cols: List[str] = []

    def fit(self, df: pd.DataFrame):
        mat = df[self.cols].to_numpy()
        pres = (mat > 0).mean(axis=0)
        self.selected_cols = [c for c, p in zip(self.cols, pres) if p >= self.prev]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.selected_cols:
            return np.zeros((len(df),0))
        mat = df[self.selected_cols].to_numpy()
        mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)
        return clr_transform(mat)


# ----------------------------- Discrete-time survival -----------------------------

def make_time_bins(max_months: int = 180) -> np.ndarray:
    """Monthly bins up to max_months."""
    return np.arange(1, max_months+1)

def person_period(X_stack: np.ndarray,
                  t_event_months: np.ndarray,
                  event_crc: np.ndarray,
                  time_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand subject-level data to person-period rows for discrete-time logistic.
    y_bin = 1 if event occurs in bin k, else 0; rows are included while at risk.
    Returns: X_pp, y_pp, t_idx per-row (time index)
    """
    n, p = X_stack.shape
    rows = []
    y = []
    t_idx = []
    for i in range(n):
        t = int(min(t_event_months[i], time_bins[-1]))
        ev = int(event_crc[i])
        # at risk from month 1..t (if censored ev=0) or 1..t (event occurs at t)
        # If censored, last row has y=0; if event, last row has y=1
        for k, tk in enumerate(time_bins, start=1):
            if tk > t:
                break
            rows.append(np.hstack([X_stack[i], [tk]]))
            if (tk == t) and (ev == 1):
                y.append(1)
            else:
                y.append(0)
            t_idx.append(tk)
    X_pp = np.vstack(rows) if rows else np.zeros((0, p+1))
    y_pp = np.array(y, dtype=int)
    t_idx = np.array(t_idx, dtype=int)
    return X_pp, y_pp, t_idx

class DiscreteTimeSurvival:
    """
    Logistic regression over person-period data with time as a numeric covariate.
    Predicts hazard h(t|x); Survival S(t) = Π_{k<=t} (1 - h_k).
    """
    def __init__(self, C=1.0):
        self.C = C
        self.model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=200)

    def fit(self, Z_surv: np.ndarray, t_event_months: np.ndarray, event_crc: np.ndarray, time_bins: np.ndarray):
        X_pp, y_pp, _ = person_period(Z_surv, t_event_months, event_crc, time_bins)
        if X_pp.shape[0] == 0:
            raise ValueError("Empty person-period matrix; check data")
        self.model.fit(X_pp, y_pp)
        self.p = Z_surv.shape[1]
        self.time_bins = time_bins.copy()
        return self

    def predict_survival(self, Z_surv: np.ndarray) -> np.ndarray:
        """Return S(t) on the same time grid used for training."""
        n = Z_surv.shape[0]
        S = np.ones((n, len(self.time_bins)))
        for j, tk in enumerate(self.time_bins):
            X = np.hstack([Z_surv, np.full((n,1), tk)])
            h = self.model.predict_proba(X)[:,1]  # hazard at time tk
            if j == 0:
                S[:, j] = 1.0 - h
            else:
                S[:, j] = S[:, j-1] * (1.0 - h)
        return S


# ----------------------------- Base learners & fusion -----------------------------

class BaseLearners:
    """Train per-modality base learners: survival (produce risk scores) and present-time classifier logits."""
    def __init__(self):
        self.models_now: Dict[str, any] = {}
        self.models_surv: Dict[str, any] = {}

    def fit(self, Xmats: Dict[str, np.ndarray], y_present: np.ndarray,
            t_event_months: np.ndarray, event_crc: np.ndarray, time_bins: np.ndarray):
        # Present-time base learners
        for key in Xmats:
            X = Xmats[key]
            if X is None or X.shape[1] == 0: 
                continue
            clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            clf.fit(X, y_present)
            self.models_now[key] = clf

        # Survival base "risk encoders": here we use simple L2 logistic on event indicator at 36 months as proxy
        # (You can replace with Cox/RSF and map to risk score.)
        horizon = min(time_bins[-1], 36)
        y_h = (t_event_months <= horizon) & (event_crc == 1)
        for key in Xmats:
            X = Xmats[key]
            if X is None or X.shape[1] == 0:
                continue
            clf = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
            clf.fit(X, y_h.astype(int))
            self.models_surv[key] = clf
        return self

    def predict_stack(self, Xmats: Dict[str, np.ndarray], time_bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return Z_surv (stacked per-modality risk logits) and Z_now (stacked present-time logits)."""
        Z_now = []
        Z_surv = []
        for key in Xmats:
            X = Xmats[key]
            if X is None or X.shape[1] == 0: 
                continue
            if key in self.models_now:
                logit = self.models_now[key].decision_function(X).reshape(-1,1)
                Z_now.append(logit)
            if key in self.models_surv:
                # Repeat survival risk logit across time to mimic per-time features (simplified)
                logit = self.models_surv[key].decision_function(X).reshape(-1,1)
                Z_surv.append(logit)
        Z_now = np.concatenate(Z_now, axis=1) if Z_now else np.zeros((len(next(iter(Xmats.values()))), 0))
        # For survival, concatenate and (optionally) repeat across time in the discrete-time model
        if Z_surv:
            Zs = np.concatenate(Z_surv, axis=1)
        else:
            Zs = np.zeros((len(next(iter(Xmats.values()))), 0))
        return Zs, Z_now


class NowFusedClassifier:
    """Fused present-time classifier over stacked base logits with isotonic calibration."""
    def __init__(self):
        self.model = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500)
        self.iso = None

    def fit(self, Z_now: np.ndarray, y_now: np.ndarray):
        if Z_now.shape[1] == 0:
            # Degenerate: no modality; fallback to intercept-only
            self.model = LogisticRegression(fit_intercept=True).fit(np.zeros((len(y_now),1)), y_now)
            self.iso = None
            return self
        self.model.fit(Z_now, y_now)
        # Calibration on same data here (in CV you'll refit on train and apply to val)
        scores = self.model.decision_function(Z_now)
        # Isotonic requires sorted scores
        order = np.argsort(scores)
        self.iso = IsotonicRegression(out_of_bounds='clip').fit(scores[order], y_now[order])
        return self

    def predict_proba(self, Z_now: np.ndarray) -> np.ndarray:
        scores = self.model.decision_function(Z_now) if Z_now.shape[1] else self.model.decision_function(np.zeros((len(Z_now),1)))
        if self.iso is None:
            return sigmoid(scores)
        return self.iso.predict(scores)


# ----------------------------- CV, metrics, end-to-end -----------------------------

def time_dependent_auc_stub(t_event, event, risk, horizon_months: int) -> float:
    """Very simple AUC at fixed horizon using censoring-agnostic approximation for demo only."""
    y = ((t_event <= horizon_months) & (event == 1)).astype(int)
    try:
        return roc_auc_score(y, risk)
    except Exception:
        return np.nan

def integrate_brier_stub(y_surv, S, time_bins):
    """Toy IBS: mean brier across times vs naive event indicator (ignores censoring for demo)."""
    t_event, event = y_surv
    bs = []
    for j, t in enumerate(time_bins):
        y = ((t_event > t) | (event == 0)).astype(int)  # survived past t -> 1
        p = S[:, j]
        bs.append(np.mean((y - p)**2))
    return float(np.mean(bs))

def survival_at(S: np.ndarray, time_bins: np.ndarray, months: int) -> np.ndarray:
    idx = np.searchsorted(time_bins, months, side='right') - 1
    idx = np.clip(idx, 0, len(time_bins)-1)
    return S[:, idx]


def kfold_blocked_by_site(df: pd.DataFrame, n_splits=5, site_col='site_id'):
    """GroupKFold by site to avoid leakage across centers."""
    groups = df[site_col].values
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(df, groups=groups, groups=groups):
        yield tr, va


def run_cv(X: pd.DataFrame, y: pd.DataFrame,
           feature_cfg: Dict[str, Dict], time_bins: np.ndarray,
           site_col: str = 'site_id', verbose: bool = True) -> pd.DataFrame:
    """Outer CV with site-blocking; transforms fit only on train; evaluate on val."""
    metrics = []
    for fold, (tr, va) in enumerate(kfold_blocked_by_site(X, n_splits=5, site_col=site_col)):
        Xtr, Xva = X.iloc[tr].copy(), X.iloc[va].copy()
        ytr, yva = y.iloc[tr].copy(), y.iloc[va].copy()

        # --- Fit featurizers on TRAIN only ---
        clin_feat = ClinicalFeaturizer(feature_cfg['clin']['num'], feature_cfg['clin']['cat']).fit(Xtr)
        germ_feat = GermlineFeaturizer(feature_cfg['germ']['cols']).fit(Xtr)
        soma_feat = SimpleScaler(feature_cfg['somatic']['cols']).fit(Xtr) if feature_cfg['somatic']['cols'] else None
        meth_feat = SimpleScaler(feature_cfg['meth']['cols']).fit(Xtr) if feature_cfg['meth']['cols'] else None
        micr_feat = MicrobiomeFeaturizer(feature_cfg['micro']['cols'], prevalence_threshold=0.1).fit(Xtr) if feature_cfg['micro']['cols'] else None

        # --- Transform TRAIN and VALIDATION using TRAIN-state ---
        Xm_tr = {
            'clin': clin_feat.transform(Xtr),
            'germ': germ_feat.transform(Xtr),
            'somatic': (soma_feat.transform(Xtr) if soma_feat else np.zeros((len(Xtr),0))),
            'meth': (meth_feat.transform(Xtr) if meth_feat else np.zeros((len(Xtr),0))),
            'micro': (micr_feat.transform(Xtr) if micr_feat else np.zeros((len(Xtr),0))),
        }
        Xm_va = {
            'clin': clin_feat.transform(Xva),
            'germ': germ_feat.transform(Xva),
            'somatic': (soma_feat.transform(Xva) if soma_feat else np.zeros((len(Xva),0))),
            'meth': (meth_feat.transform(Xva) if meth_feat else np.zeros((len(Xva),0))),
            'micro': (micr_feat.transform(Xva) if micr_feat else np.zeros((len(Xva),0))),
        }

        # --- Base learners ---
        base = BaseLearners().fit(
            Xm_tr,
            ytr['y_present_crc'].to_numpy().astype(int),
            ytr['t_event_months'].to_numpy().astype(int),
            ytr['event_crc'].to_numpy().astype(int),
            time_bins
        )
        Zs_tr, Zn_tr = base.predict_stack(Xm_tr, time_bins)
        Zs_va, Zn_va = base.predict_stack(Xm_va, time_bins)

        # --- Discrete-time survival stacker ---
        dts = DiscreteTimeSurvival(C=1.0).fit(
            Zs_tr,
            ytr['t_event_months'].to_numpy().astype(int),
            ytr['event_crc'].to_numpy().astype(int),
            time_bins
        )
        S_va = dts.predict_survival(Zs_va)

        # --- Present-time fused classifier + calibration ---
        clf_now = NowFusedClassifier().fit(Zn_tr, ytr['y_present_crc'].to_numpy().astype(int))
        p_now = clf_now.predict_proba(Zn_va)

        # --- Derive risks at horizons ---
        risk1y = 1 - survival_at(S_va, time_bins, 12)
        risk3y = 1 - survival_at(S_va, time_bins, 36)
        riskLT = 1 - survival_at(S_va, time_bins, time_bins[-1])

        # --- Metrics (toy AUCs; replace with proper time-dependent AUC with censoring) ---
        auc1y = time_dependent_auc_stub(yva['t_event_months'].to_numpy(), yva['event_crc'].to_numpy(), risk1y, 12)
        auc3y = time_dependent_auc_stub(yva['t_event_months'].to_numpy(), yva['event_crc'].to_numpy(), risk3y, 36)
        ibs   = integrate_brier_stub((yva['t_event_months'].to_numpy(), yva['event_crc'].to_numpy()), S_va, time_bins)
        auc_now = roc_auc_score(yva['y_present_crc'].to_numpy().astype(int), p_now)

        metrics.append(dict(fold=fold, auc1y=float(auc1y), auc3y=float(auc3y), ibs=float(ibs), auc_now=float(auc_now)))

        if verbose:
            print(f"[fold {fold}] AUC(now)={auc_now:.3f} | AUC(1y)={auc1y:.3f} | AUC(3y)={auc3y:.3f} | IBS={ibs:.3f}")

    return pd.DataFrame(metrics)


# ----------------------------- Demo / synthetic run -----------------------------

def _make_synthetic(n=500, n_sites=5, seed=42):
    """Create synthetic multimodal data + labels with known signal for quick sanity checks."""
    rng = np.random.default_rng(seed)
    site = rng.integers(0, n_sites, size=n)
    age = rng.normal(60, 10, size=n)
    sex = rng.choice(['F','M'], size=n)
    bmi = rng.normal(27, 4, size=n)

    prs = rng.normal(0, 1, size=n)
    lynch = rng.binomial(1, 0.02, size=n)

    # somatic (optional signals)
    msi = rng.choice(['MSS','MSI-H'], p=[0.9,0.1], size=n)
    tmb = np.abs(rng.normal(5, 3, size=n))
    apc = rng.binomial(1, 0.6, size=n)
    kras = rng.binomial(1, 0.4, size=n)

    # methylation biomarkers
    meth_sept9 = np.abs(rng.normal(0.5, 0.3, size=n))
    meth_ndrg4 = np.abs(rng.normal(0.5, 0.3, size=n))

    # microbiome (compositional)
    taxa = rng.dirichlet(alpha=np.ones(10), size=n)  # 10 taxa
    fus = taxa[:,0]  # pretend index 0 is Fusobacterium

    # true latent risk
    lin_now = -6.0 + 0.05*(age-60) + 0.8*lynch + 0.6*meth_sept9 + 0.9*fus + 0.5*(mki := kras)
    p_now = 1/(1+np.exp(-lin_now))
    y_now = rng.binomial(1, p_now)

    # incident risk over time
    base_haz = 0.002 + 0.0005*(age-60) + 0.003*lynch + 0.002*fus + 0.002*meth_sept9
    t_event = rng.exponential(1/(base_haz+1e-6))
    t_event = np.clip(t_event, 1, 120)  # months
    event_crc = (rng.random(n) < 0.6) & (t_event < 120)
    t_event = np.where(event_crc, t_event, 120)

    df = pd.DataFrame(dict(
        subject_id=np.arange(n),
        site_id=site,
        age_years=age,
        sex=sex,
        bmi=bmi,
        prs_std=prs,
        lynch_pathvar=lynch,
        msi_status=msi,
        tmb=tmb,
        gene_APC=apc,
        gene_KRAS=kras,
        meth_SEPT9=meth_sept9,
        meth_NDRG4=meth_ndrg4,
        y_present_crc=y_now,
        t_event_months=t_event.astype(int),
        event_crc=event_crc.astype(int),
    ))
    # tack on microbiome columns
    for j in range(taxa.shape[1]):
        df[f"taxa_{j}"] = taxa[:,j]
    return df

def main_demo():
    df = _make_synthetic(n=600)
    feature_cfg = {
        'clin': {'num':['age_years','bmi'], 'cat':['sex','site_id']},
        'germ': {'cols':['prs_std','lynch_pathvar']},
        'somatic': {'cols':['tmb','gene_APC','gene_KRAS']},
        'meth': {'cols':['meth_SEPT9','meth_NDRG4']},
        'micro': {'cols':[c for c in df.columns if c.startswith('taxa_')]},
    }
    time_bins = make_time_bins(120)
    metrics = run_cv(df, df[['y_present_crc','t_event_months','event_crc']], feature_cfg, time_bins)
    print("\nSummary:")
    print(metrics.describe())

if __name__ == "__main__":
    main_demo()
