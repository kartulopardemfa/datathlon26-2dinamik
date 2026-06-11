"""Fold-clean pseudo-labeling: for each fold, pseudo labels come from a model
that never saw the validation fold (honest estimate of the LB gain)."""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

sys.path.insert(0, '.')
from fe import engineer

CAT_COLS = ['department', 'university_tier', 'target_role', 'hobby',
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

Str, Ste = np.load('cache_svd_tr.npy'), np.load('cache_svd_te.npy')
toof, tpred = np.load('cache_toof2.npy'), np.load('cache_tpred2.npy')
eoof, epred = np.load('cache_eoof.npy'), np.load('cache_epred.npy')
imp_tr = pd.read_csv('cache_impflags_train.csv')
imp_te = pd.read_csv('cache_impflags_test.csv')

def build(df, S_, impf, tf, em):
    f = engineer(df).drop(columns=[c for c in DROP if c in df.columns]).reset_index(drop=True)
    f = pd.concat([f, pd.DataFrame(S_, columns=[f'svd_{i}' for i in range(S_.shape[1])]),
                   impf.reset_index(drop=True)], axis=1)
    f['text_pred_tfidf'] = tf
    f['text_pred_emb'] = em
    return f

X = build(tr, Str, imp_tr, toof, eoof)
Xte = build(te, Ste, imp_te, tpred, epred)
for c in CAT_COLS:
    X[c] = X[c].astype(str); Xte[c] = Xte[c].astype(str)

def cat():
    return CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6,
                             l2_leaf_reg=3, cat_features=CAT_COLS,
                             early_stopping_rounds=400, random_seed=42, verbose=False)

oof = np.zeros(len(X))
for k, (tr_i, va_i) in enumerate(folds):
    base = cat()
    base.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[va_i], y[va_i]))
    yp = np.clip(base.predict(Xte), 0, 100)   # fold-clean pseudo labels
    Xa = pd.concat([X.iloc[tr_i], Xte], ignore_index=True)
    ya = np.r_[y[tr_i], yp]
    wa = np.r_[np.ones(len(tr_i)), np.full(len(Xte), 0.5)]
    m = cat()
    m.fit(Xa, ya, sample_weight=wa, eval_set=(X.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X.iloc[va_i])
    print(f'fold {k} done', flush=True)
oc = np.clip(oof, 0, 100)
print(f'CLEAN pseudo pw=0.5: uniform={mean_squared_error(y, oc):.3f} LB_est={lb_est(oc):.3f}',
      flush=True)
np.save('cache_oof_pseudo_clean.npy', oof)
print('referans: 86.868 | sizintili pseudo: 84.097')
