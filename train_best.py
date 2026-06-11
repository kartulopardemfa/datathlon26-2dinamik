"""Best-config models: engineered + svd64 + impflags + tfidf/emb text preds.
No emb-PCA, no FT feature (both hurt late-year generalization).
Saves OOF + test preds for blending."""
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
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

def run(model_name, seed):
    oof = np.zeros(len(X)); tep = np.zeros(len(Xte))
    if model_name == 'cat':
        Xc, Xtc = X.copy(), Xte.copy()
        for c in CAT_COLS:
            Xc[c] = Xc[c].astype(str); Xtc[c] = Xtc[c].astype(str)
        for tr_i, va_i in folds:
            m = CatBoostRegressor(iterations=10000, learning_rate=0.03, depth=6,
                                  l2_leaf_reg=3, cat_features=CAT_COLS,
                                  early_stopping_rounds=400, random_seed=seed, verbose=False)
            m.fit(Xc.iloc[tr_i], y[tr_i], eval_set=(Xc.iloc[va_i], y[va_i]))
            oof[va_i] = m.predict(Xc.iloc[va_i]); tep += m.predict(Xtc) / 5
    else:
        Xl, Xtl = X.copy(), Xte.copy()
        for c in CAT_COLS:
            Xl[c] = Xl[c].astype('category')
            Xtl[c] = Xtl[c].astype('category').cat.set_categories(Xl[c].cat.categories)
        for tr_i, va_i in folds:
            if model_name == 'lgb':
                m = lgb.LGBMRegressor(n_estimators=6000, learning_rate=0.02, num_leaves=63,
                                      colsample_bytree=0.7, subsample=0.8, subsample_freq=1,
                                      min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
                                      random_state=seed, verbose=-1)
                m.fit(Xl.iloc[tr_i], y[tr_i], eval_set=[(Xl.iloc[va_i], y[va_i])],
                      callbacks=[lgb.early_stopping(300, verbose=False)])
            else:
                m = xgb.XGBRegressor(n_estimators=6000, learning_rate=0.02, max_depth=6,
                                     colsample_bytree=0.7, subsample=0.8, min_child_weight=5,
                                     reg_alpha=0.1, reg_lambda=1.0, enable_categorical=True,
                                     tree_method='hist', early_stopping_rounds=300,
                                     random_state=seed)
                m.fit(Xl.iloc[tr_i], y[tr_i], eval_set=[(Xl.iloc[va_i], y[va_i])], verbose=False)
            oof[va_i] = m.predict(Xl.iloc[va_i]); tep += m.predict(Xtl) / 5
    return oof, tep

for name in ['cat', 'lgb', 'xgb']:
    for seed in (42, 2026):
        o, t = run(name, seed)
        np.save(f'cache_best_{name}_{seed}_oof.npy', o)
        np.save(f'cache_best_{name}_{seed}_te.npy', t)
        print(f'{name}_{seed}: uniform={mean_squared_error(y, np.clip(o,0,100)):.3f} '
              f'LB_est={lb_est(np.clip(o,0,100)):.3f}', flush=True)
print('done')
