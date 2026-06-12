"""Exploit target-aware imputation: split each skill col into imputed/real regimes."""
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
IMP = ['sql_score','machine_learning_score','backend_score','frontend_score','cloud_score','devops_score']

tr = pd.read_csv('data/train.csv', encoding='utf-8-sig')
te = pd.read_csv('data/test.csv', encoding='utf-8-sig')
trs = pd.read_csv('data/train.csv', encoding='utf-8-sig', dtype=str)
y = tr['career_success_score'].values
folds = list(KFold(5, shuffle=True, random_state=42).split(tr))
tp_ = te.application_year.value_counts(normalize=True)
trp = tr.application_year.value_counts(normalize=True)
iw = tr.application_year.map(tp_ / trp).values
def lb_est(p):
    return np.average((y - p) ** 2, weights=iw)

def dl(s):
    s = str(s)
    return len(s.split('.')[1]) if '.' in s and s != 'nan' else 0

Str = np.load('cache_svd_tr.npy')
toof = np.load('cache_toof2.npy'); eoof = np.load('cache_eoof.npy')
impf = pd.read_csv('cache_impflags_train.csv')
X = engineer(tr).drop(columns=DROP).reset_index(drop=True)
X = pd.concat([X, pd.DataFrame(Str, columns=[f'svd_{i}' for i in range(64)]), impf], axis=1)
X['text_pred'] = toof; X['text_pred_emb'] = eoof

# regime-split columns
imp_vals = []
for c in IMP:
    f = (trs[c].map(dl) > 3).values
    X[f'{c}_if_imp'] = np.where(f, tr[c].values, np.nan)
    X[f'{c}_if_real'] = np.where(~f, tr[c].values, np.nan)
    imp_vals.append(np.where(f, tr[c].values, np.nan))
# aggregate: mean of imputed values (the strongest leak carrier)
IV = np.vstack(imp_vals)
X['imp_val_mean'] = np.nanmean(IV, axis=0)
X['imp_val_max'] = np.nanmax(IV, axis=0)
for c in CAT: X[c] = X[c].astype(str)

oof = np.zeros(len(X))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT, early_stopping_rounds=400,
                          random_seed=42, verbose=False)
    m.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X.iloc[va_i])
oc = np.clip(oof, 0, 100)
print('REGIME-SPLIT CatBoost: uniform=%.3f LB_est=%.3f' % (mean_squared_error(y, oc), lb_est(oc)))
print('REFERANS (ayni kurulum, split yok): uniform=76.821 LB_est=86.868')
np.save('cache_oof_leak.npy', oof)
