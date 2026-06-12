"""OOF target-encoding for dept x role (x tier) interactions."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, '.')
from fe import engineer

CAT = ['department', 'university_tier', 'target_role', 'hobby',
       'preferred_social_media_platform']
DROP = ['student_id', 'career_success_score', 'mentor_feedback_text']

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
y = tr['career_success_score'].values
folds = list(KFold(5, shuffle=True, random_state=42).split(tr))
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_ / trp).values
def lb_est(p):
    return np.average((y - p) ** 2, weights=iw)

def oof_te(keys, smooth=30):
    """OOF target encoding train + full-fit test."""
    k_tr = tr[keys].astype(str).agg('|'.join, axis=1)
    k_te = te[keys].astype(str).agg('|'.join, axis=1)
    enc_tr = np.full(len(tr), np.nan)
    gmean = y.mean()
    for tr_i, va_i in folds:
        g = pd.Series(y[tr_i]).groupby(k_tr.iloc[tr_i].values)
        m, n = g.mean(), g.size()
        sm = (m * n + gmean * smooth) / (n + smooth)
        enc_tr[va_i] = k_tr.iloc[va_i].map(sm).fillna(gmean).values
    g = pd.Series(y).groupby(k_tr.values)
    m, n = g.mean(), g.size()
    sm = (m * n + gmean * smooth) / (n + smooth)
    enc_te = k_te.map(sm).fillna(gmean).values
    return enc_tr, enc_te

Str = np.load('cache_svd_tr.npy')
toof = np.load('cache_toof2.npy'); eoof = np.load('cache_eoof.npy')
ftoof = np.load('cache_ft_oof.npy')
impf = pd.read_csv('cache_impflags_train.csv')
X = engineer(tr).drop(columns=DROP).reset_index(drop=True)
X = pd.concat([X, pd.DataFrame(Str, columns=[f'svd_{i}' for i in range(64)]), impf], axis=1)
X['text_pred'] = toof; X['text_pred_emb'] = eoof; X['text_pred_ft'] = ftoof

for keys, name in [(['department','target_role'], 'te_dr'),
                   (['department','target_role','university_tier'], 'te_drt'),
                   (['target_role','university_tier'], 'te_rt')]:
    e_tr, _ = oof_te(keys)
    X[name] = e_tr
for c in CAT: X[c] = X[c].astype(str)

oof = np.zeros(len(X))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT, early_stopping_rounds=400,
                          random_seed=42, verbose=False)
    m.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X.iloc[va_i])
oc = np.clip(oof, 0, 100)
print('TargetEnc CatBoost: uniform=%.3f LB_est=%.3f' % (mean_squared_error(y, oc), lb_est(oc)))
print('REFERANS (ayni ozellikler TE\'siz, ft dahil degil): 86.868 | v2 cat (ft dahil): 87.631')
np.save('cache_oof_te.npy', oof)
