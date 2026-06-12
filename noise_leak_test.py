"""Does text leak the irreducible 'noise' (y - f(tabular))?"""
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
y = tr['career_success_score'].values
yr = tr.application_year.values
folds = list(KFold(5, shuffle=True, random_state=42).split(tr))
imp_tr = pd.read_csv('cache_impflags_train.csv')

# tabular ONLY (no text-derived features at all)
X = engineer(tr).drop(columns=DROP + ['text_len', 'text_words'], errors='ignore').reset_index(drop=True)
X = pd.concat([X, imp_tr], axis=1)
for c in CAT_COLS: X[c] = X[c].astype(str)
oof = np.zeros(len(X))
for tr_i, va_i in folds:
    m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                          cat_features=CAT_COLS, early_stopping_rounds=400,
                          random_seed=42, verbose=False)
    m.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[va_i], y[va_i]))
    oof[va_i] = m.predict(X.iloc[va_i])
np.save('cache_oof_tabonly.npy', oof)
oc = np.clip(oof, 0, 100)
res = y - oc
print('tabular-only uniform MSE:', mean_squared_error(y, oc), flush=True)

toof = np.load('cache_toof2.npy')
eoof = np.load('cache_eoof.npy')
ft = np.load('cache_ft_oof.npy')
print('\nNOISE-LEAK: corr(text_pred, tabular_residual)')
print('year   tfidf    emb     ft     res_std')
for yy in sorted(np.unique(yr)):
    m = yr == yy
    print(yy, ' %.3f  %.3f  %.3f   %.1f' % (
        np.corrcoef(toof[m], res[m])[0, 1], np.corrcoef(eoof[m], res[m])[0, 1],
        np.corrcoef(ft[m], res[m])[0, 1], res[m].std()), flush=True)
allc = np.corrcoef(np.vstack([toof, eoof, ft]).mean(0), res)[0, 1]
print('GENEL (3 metin ort.):', round(allc, 4))
