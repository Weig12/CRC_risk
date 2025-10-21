
# tests/test_leakage_and_calibration.py
# Run with: pytest -q
import numpy as np
import pandas as pd
from crc_risk_pipeline import ClinicalFeaturizer, _make_synthetic, make_time_bins, run_cv

def test_no_leakage_clinical_stats():
    df = _make_synthetic(n=200, seed=123)
    # Split manually
    msk = df['site_id'] < df['site_id'].median()
    train, val = df[msk], df[~msk]

    feat = ClinicalFeaturizer(numeric_cols=['age_years','bmi'], cat_cols=['sex','site_id']).fit(train)
    # Capture train means
    m_age = feat.state.num_means['age_years']
    s_age = feat.state.num_stds['age_years']

    # Transform val should use TRAIN stats, not recompute
    Xv = feat.transform(val)
    # Recompute naive z on val to ensure it's different from 0 mean, unit var generally
    zv = (val['age_years'] - val['age_years'].mean()) / (val['age_years'].std())
    # Check train-mean-based z has mean close to (val_mean - train_mean)/train_std, not ~0
    assert abs(Xv[:,0].mean() - ((val['age_years'].mean()-m_age)/s_age)) < 1e-6

def test_calibration_monotonicity_present_time():
    df = _make_synthetic(n=400, seed=7)
    # Simple single fold run to get probabilities
    feature_cfg = {
        'clin': {'num':['age_years','bmi'], 'cat':['sex','site_id']},
        'germ': {'cols':['prs_std','lynch_pathvar']},
        'somatic': {'cols':['tmb','gene_APC','gene_KRAS']},
        'meth': {'cols':['meth_SEPT9','meth_NDRG4']},
        'micro': {'cols':[c for c in df.columns if c.startswith('taxa_')]},
    }
    # Use 2 folds to keep fast
    from crc_risk_pipeline import kfold_blocked_by_site, ClinicalFeaturizer, GermlineFeaturizer, SimpleScaler, MicrobiomeFeaturizer, BaseLearners, DiscreteTimeSurvival, NowFusedClassifier, survival_at
    time_bins = make_time_bins(60)
    # take first fold
    folds = list(kfold_blocked_by_site(df, n_splits=2, site_col='site_id'))
    tr, va = folds[0]
    Xtr, Xva = df.iloc[tr], df.iloc[va]
    ytr, yva = df.iloc[tr][['y_present_crc','t_event_months','event_crc']], df.iloc[va][['y_present_crc','t_event_months','event_crc']]

    # Fit transforms on TRAIN only
    clin = ClinicalFeaturizer(['age_years','bmi'], ['sex','site_id']).fit(Xtr)
    germ = GermlineFeaturizer(['prs_std','lynch_pathvar']).fit(Xtr)
    soma = SimpleScaler(['tmb','gene_APC','gene_KRAS']).fit(Xtr)
    meth = SimpleScaler(['meth_SEPT9','meth_NDRG4']).fit(Xtr)
    micr = MicrobiomeFeaturizer([c for c in df.columns if c.startswith('taxa_')]).fit(Xtr)

    Xm_tr = {'clin':clin.transform(Xtr),'germ':germ.transform(Xtr),'somatic':soma.transform(Xtr),'meth':meth.transform(Xtr),'micro':micr.transform(Xtr)}
    Xm_va = {'clin':clin.transform(Xva),'germ':germ.transform(Xva),'somatic':soma.transform(Xva),'meth':meth.transform(Xva),'micro':micr.transform(Xva)}

    base = BaseLearners().fit(Xm_tr, ytr['y_present_crc'].to_numpy().astype(int),
                              ytr['t_event_months'].to_numpy().astype(int),
                              ytr['event_crc'].to_numpy().astype(int),
                              time_bins)
    _, Zn_va = base.predict_stack(Xm_va, time_bins)
    from sklearn.isotonic import IsotonicRegression
    # Fit fused classifier on train to ensure isotonic monotonicity
    _, Zn_tr = base.predict_stack(Xm_tr, time_bins)
    clf = NowFusedClassifier().fit(Zn_tr, ytr['y_present_crc'].to_numpy().astype(int))
    p = clf.predict_proba(Zn_va)
    # Monotonicity: probability should increase with model score (isotonic enforces non-decreasing)
    scores = clf.model.decision_function(Zn_va)
    # Sort by score
    order = np.argsort(scores)
    p_sorted = p[order]
    assert np.all(np.diff(p_sorted) >= -1e-9)
